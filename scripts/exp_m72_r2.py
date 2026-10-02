"""M72: normalise the per-airport error by that airport's own variability.

RMSE 314.8 at LIRF vs 140.6 at LEBL means nothing on its own: if Rome's taxi-out
is intrinsically twice as variable, both models are equally good. What compares is
the share of variance explained. A uniformly high R2 means the heterogeneity is
intrinsic; a low one names an airport where we underperform our own standard.
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
meta = pd.read_parquet(PROC / "meta_train2025.parquet")
y = meta["y"].astype("float64").to_numpy()
va = meta["va"].to_numpy()
p = np.load(EXP / "m49_blend_va.npy")
yv = y[va]
apt = X["AIRPORT"].astype(str).to_numpy()[va]
d = X["d_aobt3"].to_numpy(dtype="float64")[va]
art = np.zeros(len(y), bool)
for k in (1, 2, 3):
    art |= np.abs(y - k * 86400) < 3600
g = np.isfinite(p) & ~art[va] & (yv < 3600)

print("=== trafic ordinaire, normalise par la variabilite propre de l aeroport ===")
print("  %-7s %8s %9s %9s %8s %12s %12s" % ("aero", "n", "ecart-y", "RMSE", "R2", "RMSE/ecart", "AOBT_3 seul"))
rows = []
for a in sorted(set(apt)):
    m = g & (apt == a)
    if m.sum() < 500:
        continue
    sd = yv[m].std()
    r = rmse(yv[m], p[m])
    r2 = 1 - (r / sd) ** 2
    ma = m & np.isfinite(d)
    raw = rmse(yv[ma], np.clip(d[ma], 60, 86400)) if ma.sum() > 100 else np.nan
    rows.append((a, m.sum(), sd, r, r2, r / sd, raw))
for a, n, sd, r, r2, ratio, raw in sorted(rows, key=lambda x: x[4]):
    print("  %-7s %8d %9.1f %9.1f %8.4f %12.3f %12.1f" % (a, n, sd, r, r2, ratio, raw))

print("\n=== gain si les 4 plus faibles atteignaient le R2 median ===")
med = float(np.median([x[4] for x in rows]))
print("  R2 median = %.4f" % med)
rk = pd.read_parquet(ROOT / "data" / "ranking.parquet", columns=["ADEP_mvt", "PHASE_mvt"])
share = rk[rk["PHASE_mvt"] == "DEP"]["ADEP_mvt"].astype(str).value_counts(normalize=True)
cur = sum(float(share.get(a, 0)) * r ** 2 for a, _, _, r, _, _, _ in rows)
new = 0.0
for a, n, sd, r, r2, ratio, raw in rows:
    rr = sd * np.sqrt(1 - max(r2, med)) if r2 < med else r
    new += float(share.get(a, 0)) * rr ** 2
    if r2 < med:
        print("  %-7s R2 %.4f -> %.4f : RMSE %.1f -> %.1f" % (a, r2, med, r, rr))
w = sum(float(share.get(a, 0)) for a, *_ in rows)
print("  RMSE ordinaire pondere par le test : %.1f -> %.1f  (%+.1f)" % (
    np.sqrt(cur / w), np.sqrt(new / w), np.sqrt(new / w) - np.sqrt(cur / w)))
