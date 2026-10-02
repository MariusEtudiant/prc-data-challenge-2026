"""N05: the regime mixture, per airport, against the composition that now counts.

The error budget on the rebuilt mirror is unambiguous: 975 flights out of 344,412 -
0.28% - carry 36% of the error, and predicting everything above two hours perfectly
would take us from 294.38 to 254.44. The ordinary traffic, by contrast, is nearly
exhausted: N04 found only 2.5 points of residual signal inside each month.

The mixture (classifier for P(y > 7200) plus two conditional regressions, Platt-
calibrated on the ES window) was our champion on 2026-09-01 at 337.3 when the plain
three-GBDT ensemble sat at 353.4. It was dropped when the pipeline moved to
per-airport models and was NEVER recombined with them - so it has never been tried in
the architecture we actually ship, nor on a test that is now 56% July.

Three things measured here, per airport, on the honest odd/even day split:
  A  our current four-source blend           (the reference)
  B  the mixture alone
  C  blend and mixture combined by NNLS

If C beats A on both months, the tail is finally worth attacking.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import nnls

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
PROC, EXP = ROOT / "data" / "processed", ROOT / "experiments"
from dc26.validation import rmse
import dc26.mixture as mixmod
from dc26.mixture import fit_mixture, predict_mixture

# At 7200s most airports hold no positives at all (EDDF 0, EDDM 0, EHAM 1): the
# extreme regime is essentially a Rome phenomenon. 3600s keeps a fittable population
# everywhere and still covers the 1-2h band, which alone carries 10.8% of the error.
mixmod.THR = 3600

CAP, MIN_TRAIN = 86400, 20000
FILES = ["X_v2_mirror_t.parquet", "flightctx_train2025.parquet", "depregime_train2025.parquet",
         "aobtq_train2025.parquet", "anomreg_train2025.parquet"]

X = pd.concat([pd.read_parquet(PROC / f) for f in FILES], axis=1)
meta = pd.read_parquet(PROC / "meta_train2025.parquet")
y = meta["y"].astype("float64")
tr, va, es = meta["tr"].to_numpy(), meta["va"].to_numpy(), meta["es"].to_numpy()
ok = (y > 0).to_numpy() & (y < 200000).to_numpy()
art = np.zeros(len(y), bool)
for k in (1, 2, 3):
    art |= (y - k * 86400).abs().to_numpy() < 3600
ok &= ~art
cat = [c for c in X.columns if X[c].dtype.name == "category"]
apt = X["AIRPORT"].astype(str).to_numpy()
apt_va, yv, clean = apt[va], y[va].to_numpy(), ~art[va]
mvt = pd.read_parquet(PROC / "movements_2025.parquet", columns=["PHASE_mvt", "MVT_TIME_UTC_mvt"])
t = mvt[mvt["PHASE_mvt"] == "DEP"]["MVT_TIME_UTC_mvt"].reset_index(drop=True)
day, mon = t.dt.day.to_numpy()[va], t.dt.month.to_numpy()[va]

n39 = np.load(EXP / "m39_preds.npz")
ng = np.clip(n39["nn_global"], 60, CAP)
na = np.clip(np.where(np.isfinite(n39["nn_airport"]), n39["nn_airport"], n39["nn_global"]), 60, CAP)
V = np.vstack([np.load(EXP / "m96_reference.npy"), np.load(EXP / "v14_cb_mirror.npy"), ng, na])
W = json.loads((EXP / "v15_weights.json").read_text())
blend = np.full(len(yv), np.nan)
for a, w in W.items():
    m = apt_va == a
    if m.any():
        blend[m] = np.clip(np.array(w) @ np.nan_to_num(V[:, m]), 60, CAP)

t0 = time.time()
mix = np.full(len(yv), np.nan)
print("=== ajustement du melange de regimes, par aeroport ===", flush=True)
for a in np.unique(apt):
    f = ok & tr & (apt == a)
    sv = apt_va == a
    if f.sum() < MIN_TRAIN or not sv.any():
        continue
    n_ext = int((y[f] > mixmod.THR).sum())
    if n_ext < 40:
        print("  [%s] seulement %d extremes -> melange non ajuste" % (a, n_ext), flush=True)
        continue
    mdl = fit_mixture(X[f], y[f], es[f], cat)
    # predict_mixture returns (prediction, p_extreme); we want the prediction
    mix[sv] = np.clip(predict_mixture(mdl, X[va][sv])[0], 60, CAP)
    print("  [%s] %d extremes sur %d (%.2f%%)  (%.0f min)" % (
        a, n_ext, f.sum(), 100 * n_ext / f.sum(), (time.time() - t0) / 60), flush=True)

np.save(EXP / "n05_mixture_va.npy", mix)
have = np.isfinite(mix) & np.isfinite(blend) & clean
print("\ncouverture du melange: %.1f%% du holdout" % (100 * have.mean()))

print("\n=== comparaison, ajustee jours IMPAIRS / mesuree jours PAIRS ===")
print("  %-8s %10s %10s %10s %8s %8s" % ("aero", "A melange", "B mixture", "C combine", "w_mix", "gain C-A"))
tA = tB = tC = n = 0.0
for a in sorted(set(apt_va)):
    mo = have & (apt_va == a) & (day % 2 == 1)
    me = have & (apt_va == a) & (day % 2 == 0)
    if mo.sum() < 500 or me.sum() < 500:
        continue
    M = np.vstack([blend, mix])
    w, _ = nnls(M[:, mo].T, yv[mo])
    w = w / w.sum() if w.sum() > 0 else np.array([1.0, 0.0])
    pc = np.clip(w @ M[:, me], 60, CAP)
    rA, rB, rC = rmse(yv[me], blend[me]), rmse(yv[me], mix[me]), rmse(yv[me], pc)
    tA += ((yv[me] - blend[me]) ** 2).sum()
    tB += ((yv[me] - mix[me]) ** 2).sum()
    tC += ((yv[me] - pc) ** 2).sum()
    n += me.sum()
    print("  %-8s %10.1f %10.1f %10.1f %8.2f %+8.1f" % (a, rA, rB, rC, w[1], rC - rA))
print("\n  GLOBAL jours pairs : A=%.2f  B=%.2f  C=%.2f  (C-A = %+.2f)" % (
    np.sqrt(tA / n), np.sqrt(tB / n), np.sqrt(tC / n), np.sqrt(tC / n) - np.sqrt(tA / n)))

for lab, sel in (("janvier", mon == 1), ("juillet", mon == 7)):
    m = have & sel & (day % 2 == 0)
    print("  %-10s A=%.2f  B=%.2f" % (lab, rmse(yv[m], blend[m]), rmse(yv[m], mix[m])))
