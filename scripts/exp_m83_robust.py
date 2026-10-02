"""M83: is the LIRF gain from gating robust, or an artefact of the odd/even split?
Four independent partitions of January, plus a check that the gate threshold is not
knife-edged.
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.optimize import nnls

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
PROC, EXP = ROOT / "data" / "processed", ROOT / "experiments"
from dc26.validation import rmse

CAP = 86400
X = pd.read_parquet(PROC / "X_v2_mirror_t.parquet", columns=["AIRPORT", "doy"])
meta = pd.read_parquet(PROC / "meta_train2025.parquet")
y = meta["y"].astype("float64").to_numpy()
va = meta["va"].to_numpy()
art = np.zeros(len(y), bool)
for k in (1, 2, 3):
    art |= np.abs(y - k * 86400) < 3600
yv = y[va]
apt = X["AIRPORT"].astype(str).to_numpy()[va]
doy = X["doy"].to_numpy()[va]
vp = meta.loc[va, "val_part"].to_numpy()
jan = (vp == "jan") & ~art[va]

t = np.load(EXP / "m42_preds.npz")
n39 = np.load(EXP / "m39_preds.npz")
V = np.vstack([
    np.clip(t["air_lgb_va"], 60, CAP), np.clip(t["air_cb_va"], 60, CAP),
    np.clip(t["glob_lgb_va"], 60, CAP), np.clip(n39["nn_global"], 60, CAP),
    np.clip(np.where(np.isfinite(n39["nn_airport"]), n39["nn_airport"], n39["nn_global"]), 60, CAP)])
tree = np.nanmean(V[:2], axis=0)


def gated(g):
    W = V.copy()
    for i in (3, 4):
        W[i] = np.where(tree <= g, V[i], tree)
    return W


def run(M, mf, mt):
    A = np.where(np.isfinite(M[:, mf].T), M[:, mf].T, 0.0)
    w, _ = nnls(A, yv[mf])
    w = w / w.sum() if w.sum() > 0 else np.array([1.0, 0, 0, 0, 0])
    return rmse(yv[mt], np.clip(w @ np.nan_to_num(M[:, mt]), 60, CAP))


SPLITS = {
    "pair/impair": (doy % 2 == 1, doy % 2 == 0),
    "impair/pair": (doy % 2 == 0, doy % 2 == 1),
    "1re/2e quinzaine": (doy <= 15, doy > 15),
    "2e/1re quinzaine": (doy > 15, doy <= 15),
}
print("=== LIRF : gain du gate sur quatre partitions independantes de janvier ===")
print("  %-20s %10s %10s %9s" % ("partition", "A actuelle", "C gate", "gain"))
G = gated(7200)
for lab, (a_, b_) in SPLITS.items():
    mf, mt = jan & a_ & (apt == "LIRF"), jan & b_ & (apt == "LIRF")
    if mf.sum() < 300 or mt.sum() < 300:
        continue
    ra, rc = run(V, mf, mt), run(G, mf, mt)
    print("  %-20s %10.1f %10.1f %+9.1f" % (lab, ra, rc, rc - ra))

print("\n=== sensibilite au seuil (partition pair/impair, LIRF) ===")
mf, mt = jan & (doy % 2 == 1) & (apt == "LIRF"), jan & (doy % 2 == 0) & (apt == "LIRF")
print("  %-12s %10s" % ("seuil", "RMSE"))
print("  %-12s %10.1f" % ("aucun gate", run(V, mf, mt)))
for g in (3600, 5400, 7200, 10800, 14400, 21600):
    print("  %-12d %10.1f" % (g, run(gated(g), mf, mt)))

print("\n=== effet global sur les dix aeroports, moyenne des quatre partitions ===")
for lab, (a_, b_) in SPLITS.items():
    ta = tc = 0.0
    n = 0
    for ap in sorted(set(apt)):
        mf, mt = jan & a_ & (apt == ap), jan & b_ & (apt == ap)
        if mf.sum() < 300 or mt.sum() < 300:
            continue
        ta += run(V, mf, mt) ** 2 * mt.sum()
        tc += run(G, mf, mt) ** 2 * mt.sum()
        n += mt.sum()
    print("  %-20s A=%.2f  C=%.2f  (%+.2f)" % (lab, np.sqrt(ta / n), np.sqrt(tc / n),
                                               np.sqrt(tc / n) - np.sqrt(ta / n)))
