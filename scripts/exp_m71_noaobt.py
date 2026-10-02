"""M71: AOBT_3 is our dominant feature. Where it is missing, the problem is a
different one - and one model currently serves both populations, dominated by the
98.5% that have it. Is that where LFPG/EGLL/LIRF lose their ordinary traffic?
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
PROC, EXP = ROOT / "data" / "processed", ROOT / "experiments"
from dc26.validation import rmse

X = pd.read_parquet(PROC / "X_v2_mirror_t.parquet", columns=["AIRPORT", "d_aobt3"])
Xr = pd.read_parquet(PROC / "X_v2_ranking_t.parquet", columns=["AIRPORT", "d_aobt3"])
meta = pd.read_parquet(PROC / "meta_train2025.parquet")
y = meta["y"].astype("float64").to_numpy()
va = meta["va"].to_numpy()
p = np.load(EXP / "m49_blend_va.npy")
yv = y[va]
apt = X["AIRPORT"].astype(str).to_numpy()[va]
has = np.isfinite(X["d_aobt3"].to_numpy(dtype="float64"))[va]
art = np.zeros(len(y), bool)
for k in (1, 2, 3):
    art |= np.abs(y - k * 86400) < 3600
g = np.isfinite(p) & ~art[va] & (yv < 3600)   # ordinary traffic only
sq = (yv - p) ** 2

print("=== trafic ordinaire : avec vs sans AOBT_3, par aeroport ===")
print("  %-7s %8s %8s %10s %10s %13s" % ("aero", "n avec", "n sans", "RMSE avec", "RMSE sans", "part err2 sans"))
tot = sq[g].sum()
for a in sorted(set(apt)):
    m = g & (apt == a)
    w, wo = m & has, m & ~has
    if wo.sum() < 30:
        print("  %-7s %8d %8d %10.1f %10s %13s" % (a, w.sum(), wo.sum(), rmse(yv[w], p[w]), "-", "-"))
        continue
    print("  %-7s %8d %8d %10.1f %10.1f %12.1f%%" % (
        a, w.sum(), wo.sum(), rmse(yv[w], p[w]), rmse(yv[wo], p[wo]), 100 * sq[wo].sum() / tot))

w, wo = g & has, g & ~has
print("\n  GLOBAL  avec=%d (RMSE %.1f) | sans=%d (RMSE %.1f) -> les %.1f%% sans AOBT_3 font %.1f%% de l erreur" % (
    w.sum(), rmse(yv[w], p[w]), wo.sum(), rmse(yv[wo], p[wo]),
    100 * wo.sum() / g.sum(), 100 * sq[wo].sum() / tot))

hr = np.isfinite(Xr["d_aobt3"].to_numpy(dtype="float64"))
print("\n=== part sans AOBT_3 dans le jeu de test ===")
s = pd.Series(~hr).groupby(Xr["AIRPORT"].astype(str)).mean().sort_values(ascending=False)
print("  " + "  ".join("%s=%.1f%%" % (k, 100 * v) for k, v in s.items()))
print("  ensemble du test: %.1f%%" % (100 * (~hr).mean()))
