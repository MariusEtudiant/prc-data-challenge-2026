"""M80: the NNLS blend gives the neural networks zero weight at LIRF and LTFM -
our two weakest airports - while the three strongest draw half or more from them.
Are the networks actually bad there, or merely redundant with the trees?
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
PROC, EXP = ROOT / "data" / "processed", ROOT / "experiments"
from dc26.validation import rmse

CAP = 86400
X = pd.read_parquet(PROC / "X_v2_mirror_t.parquet", columns=["AIRPORT"])
meta = pd.read_parquet(PROC / "meta_train2025.parquet")
y = meta["y"].astype("float64").to_numpy()
va = meta["va"].to_numpy()
art = np.zeros(len(y), bool)
for k in (1, 2, 3):
    art |= np.abs(y - k * 86400) < 3600
yv, apt, clean = y[va], X["AIRPORT"].astype(str).to_numpy()[va], ~art[va]

t = np.load(EXP / "m42_preds.npz")
n39 = np.load(EXP / "m39_preds.npz")
S = {
    "air_lgb": np.clip(t["air_lgb_va"], 60, CAP),
    "air_cb": np.clip(t["air_cb_va"], 60, CAP),
    "glob_lgb": np.clip(t["glob_lgb_va"], 60, CAP),
    "nn_global": np.clip(n39["nn_global"], 60, CAP),
    "nn_airport": np.clip(np.where(np.isfinite(n39["nn_airport"]), n39["nn_airport"],
                                   n39["nn_global"]), 60, CAP),
}
blend = np.load(EXP / "m49_blend_va.npy")

print("=== RMSE de chaque source, par aeroport (holdout nettoye) ===")
print("  %-7s %9s %9s %9s %10s %11s %9s %10s" % (
    "aero", "air_lgb", "air_cb", "glob_lgb", "nn_global", "nn_airport", "melange", "meilleur"))
for a in sorted(set(apt)):
    m = clean & (apt == a) & np.isfinite(blend)
    if m.sum() < 500:
        continue
    r = {k: rmse(yv[m], v[m]) for k, v in S.items() if np.isfinite(v[m]).all()}
    if not r:
        continue
    best = min(r, key=r.get)
    print("  %-7s %9.1f %9.1f %9.1f %10.1f %11.1f %9.1f %10s" % (
        a, r.get("air_lgb", np.nan), r.get("air_cb", np.nan), r.get("glob_lgb", np.nan),
        r.get("nn_global", np.nan), r.get("nn_airport", np.nan),
        rmse(yv[m], blend[m]), best))

print("\n=== correlation des erreurs entre sources (diversite du melange) ===")
print("  %-7s %22s %24s" % ("aero", "corr(err air_lgb, air_cb)", "corr(err arbres, reseaux)"))
for a in sorted(set(apt)):
    m = clean & (apt == a) & np.isfinite(blend)
    if m.sum() < 500:
        continue
    e = {k: yv[m] - v[m] for k, v in S.items()}
    tree = (e["air_lgb"] + e["air_cb"]) / 2
    net = (e["nn_global"] + e["nn_airport"]) / 2
    print("  %-7s %22.3f %24.3f" % (a, np.corrcoef(e["air_lgb"], e["air_cb"])[0, 1],
                                    np.corrcoef(tree, net)[0, 1]))
