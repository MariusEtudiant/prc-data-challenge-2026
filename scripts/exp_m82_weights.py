"""M82: honest test of how the blend weights are fitted.

M81 showed a handful of flights dominate the NNLS objective and drive the network
weights to zero at LIRF - but measured that in-sample. Proper protocol here:

  fit on JANUARY ODD days, measure on JANUARY EVEN days  (all ten airports)
  fit on JANUARY,           measure on JULY              (the three July airports)

Three ways of fitting the same five sources:
  A  as today: NNLS on every clean January row
  B  NNLS on ordinary rows only (y < 7200) - the tail cannot dominate the fit
  C  NNLS on gated predictions - networks replaced by the tree consensus outside
     the ordinary regime
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
clean = ~art[va]
vp = meta.loc[va, "val_part"].to_numpy()
jan, jul = (vp == "jan") & clean, (vp == "jul") & clean

t = np.load(EXP / "m42_preds.npz")
n39 = np.load(EXP / "m39_preds.npz")
names = ["air_lgb", "air_cb", "glob_lgb", "nn_global", "nn_airport"]
V = np.vstack([
    np.clip(t["air_lgb_va"], 60, CAP), np.clip(t["air_cb_va"], 60, CAP),
    np.clip(t["glob_lgb_va"], 60, CAP), np.clip(n39["nn_global"], 60, CAP),
    np.clip(np.where(np.isfinite(n39["nn_airport"]), n39["nn_airport"], n39["nn_global"]), 60, CAP)])
tree = np.nanmean(V[:2], axis=0)
Vg = V.copy()
for i in (3, 4):
    Vg[i] = np.where(tree <= GATE, V[i], tree)


def fit(M, mask):
    A = np.where(np.isfinite(M[:, mask].T), M[:, mask].T, 0.0)
    w, _ = nnls(A, yv[mask])
    return w / w.sum() if w.sum() > 0 else np.array([1.0, 0, 0, 0, 0])


def apply(M, w, mask):
    return np.clip(w @ np.nan_to_num(M[:, mask]), 60, CAP)


VAR = {"A actuelle": (V, False), "B sans la queue": (V, True), "C gate": (Vg, False)}
odd, even = jan & (doy % 2 == 1), jan & (doy % 2 == 0)

print("=== ajuste sur janvier impair, mesure sur janvier PAIR (hors echantillon) ===")
print("  %-7s %7s %11s %13s %8s" % ("aero", "n pair", "A actuelle", "B sans queue", "C gate"))
tot = {k: 0.0 for k in VAR}
nn = 0
for a in sorted(set(apt)):
    mf, mt = odd & (apt == a), even & (apt == a)
    if mf.sum() < 300 or mt.sum() < 300:
        continue
    out = {}
    for k, (M, ordonly) in VAR.items():
        f = mf & (yv < GATE) if ordonly else mf
        w = fit(M, f)
        p = apply(M, w, mt)
        out[k] = rmse(yv[mt], p)
        tot[k] += ((yv[mt] - p) ** 2).sum()
    nn += mt.sum()
    print("  %-7s %7d %11.1f %13.1f %8.1f" % (a, mt.sum(), out["A actuelle"],
                                              out["B sans la queue"], out["C gate"]))
print("  %-7s %7d %11.2f %13.2f %8.2f" % ("GLOBAL", nn, *[np.sqrt(tot[k] / nn) for k in
                                                          ("A actuelle", "B sans la queue", "C gate")]))

print("\n=== ajuste sur janvier entier, mesure sur JUILLET (3 aeroports) ===")
print("  %-7s %7s %11s %13s %8s" % ("aero", "n juil", "A actuelle", "B sans queue", "C gate"))
tot = {k: 0.0 for k in VAR}
nn = 0
for a in sorted(set(apt[jul])):
    mf, mt = jan & (apt == a), jul & (apt == a)
    if mf.sum() < 300 or mt.sum() < 300:
        continue
    out = {}
    for k, (M, ordonly) in VAR.items():
        f = mf & (yv < GATE) if ordonly else mf
        w = fit(M, f)
        p = apply(M, w, mt)
        out[k] = rmse(yv[mt], p)
        tot[k] += ((yv[mt] - p) ** 2).sum()
    nn += mt.sum()
    print("  %-7s %7d %11.1f %13.1f %8.1f" % (a, mt.sum(), out["A actuelle"],
                                              out["B sans la queue"], out["C gate"]))
print("  %-7s %7d %11.2f %13.2f %8.2f" % ("GLOBAL", nn, *[np.sqrt(tot[k] / nn) for k in
                                                          ("A actuelle", "B sans la queue", "C gate")]))
