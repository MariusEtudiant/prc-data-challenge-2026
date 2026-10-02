"""M81: the networks are trained on a target clipped at 7200s, so they collapse on
heavy-tailed airports - LIRF nn_global 757.9 vs tree 376.7. The NNLS weights were
fitted on ungated predictions, so that tail damage drove their weight to zero at
exactly the two airports where the sources disagree most (LIRF tree/net error
correlation 0.569 vs 0.94 elsewhere) - i.e. where blending should pay most.

Test: refit the per-airport weights on gated predictions - networks contribute only
where the trees say the flight is ordinary. Weights fitted on JANUARY, measured on
JULY, exactly as M42 did, so the comparison is honest.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import nnls

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
PROC, EXP = ROOT / "data" / "processed", ROOT / "experiments"
from dc26.validation import rmse

CAP, GATE = 86400, 7200
X = pd.read_parquet(PROC / "X_v2_mirror_t.parquet", columns=["AIRPORT"])
meta = pd.read_parquet(PROC / "meta_train2025.parquet")
y = meta["y"].astype("float64").to_numpy()
va = meta["va"].to_numpy()
art = np.zeros(len(y), bool)
for k in (1, 2, 3):
    art |= np.abs(y - k * 86400) < 3600
yv, apt, clean = y[va], X["AIRPORT"].astype(str).to_numpy()[va], ~art[va]
vp = meta.loc[va, "val_part"].to_numpy()
jan, jul = vp == "jan", vp == "jul"

t = np.load(EXP / "m42_preds.npz")
n39 = np.load(EXP / "m39_preds.npz")
names = ["air_lgb", "air_cb", "glob_lgb", "nn_global", "nn_airport"]
V = np.vstack([
    np.clip(t["air_lgb_va"], 60, CAP), np.clip(t["air_cb_va"], 60, CAP),
    np.clip(t["glob_lgb_va"], 60, CAP), np.clip(n39["nn_global"], 60, CAP),
    np.clip(np.where(np.isfinite(n39["nn_airport"]), n39["nn_airport"], n39["nn_global"]), 60, CAP)])

# gate: outside the ordinary regime the networks are replaced by the tree consensus
tree_mean = np.nanmean(V[:2], axis=0)
ordinary = tree_mean <= GATE
Vg = V.copy()
for i in (3, 4):
    Vg[i] = np.where(ordinary, V[i], tree_mean)
print("part des vols en regime ordinaire: %.2f%%" % (100 * np.mean(ordinary[clean])))

print("\n=== poids reajustes sur janvier, mesures sur juillet ===")
print("  %-7s %9s %9s %9s %28s" % ("aero", "M42 jan", "gate jan", "gate jul", "poids reseaux (gate)"))
old = json.loads((EXP / "m42_weights.json").read_text())
new = {}
tot_o = tot_n = 0.0
for a in sorted(set(apt)):
    m_all = (apt == a) & clean
    m_fit = m_all & jan
    m_tst = m_all & jul
    if m_fit.sum() < 200:
        continue
    for tag, M in (("old", V), ("new", Vg)):
        A = np.where(np.isfinite(M[:, m_fit].T), M[:, m_fit].T, 0.0)
        w, _ = nnls(A, yv[m_fit])
        w = w / w.sum() if w.sum() > 0 else np.array([1.0, 0, 0, 0, 0])
        if tag == "new":
            new[a] = [float(x) for x in w]
            wn = w
    wo = np.array(old.get(a, [1, 0, 0, 0, 0]), float)
    po = np.clip(wo @ np.nan_to_num(V[:, m_all]), 60, CAP)
    pn = np.clip(wn @ np.nan_to_num(Vg[:, m_all]), 60, CAP)
    ro_j, rn_j = rmse(yv[m_all & jan[m_all.nonzero()[0] * 0 + True] if False else m_fit], po[jan[m_all]]), None
    ro_j = rmse(yv[m_fit], po[jan[m_all]])
    rn_j = rmse(yv[m_fit], pn[jan[m_all]])
    rn_u = rmse(yv[m_tst], pn[jul[m_all]]) if m_tst.sum() > 100 else np.nan
    print("  %-7s %9.1f %9.1f %9.1f   %s" % (
        a, ro_j, rn_j, rn_u, " ".join("%s=%.2f" % (n, x) for n, x in zip(names, wn) if x > 0.01)))
    tot_o += ((yv[m_all] - po) ** 2).sum()
    tot_n += ((yv[m_all] - pn) ** 2).sum()
    if a in ("LIRF", "LTFM"):
        print("        ancien: %s" % " ".join("%s=%.2f" % (n, x) for n, x in zip(names, wo) if x > 0.01))

n = int(clean.sum())
print("\nGLOBAL nettoye : M42 %.2f -> gate %.2f (%+.2f)" % (
    np.sqrt(tot_o / n), np.sqrt(tot_n / n), np.sqrt(tot_n / n) - np.sqrt(tot_o / n)))
(EXP / "m81_weights.json").write_text(json.dumps(new))
