"""M87: check the three factors the Istanbul literature names, against our data.
  1. extreme variance (the paper reports 10 to 140 minutes)
  2. arrival traffic interfering with taxiing aircraft
  3. gate-to-threshold distance, which is enormous at IST
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
PROC = ROOT / "data" / "processed"

X = pd.concat([pd.read_parquet(PROC / f) for f in
               ["X_v2_mirror_t.parquet", "depregime_train2025.parquet"]], axis=1)
meta = pd.read_parquet(PROC / "meta_train2025.parquet")
y = meta["y"].astype("float64").to_numpy()
apt = X["AIRPORT"].astype(str).to_numpy()
ok = (y > 0) & (y < 200000)
art = np.zeros(len(y), bool)
for k in (1, 2, 3):
    art |= np.abs(y - k * 86400) < 3600
ok &= ~art

print("=== 1. etendue reelle du roulage, en minutes ===")
print("  %-7s %8s %6s %6s %6s %6s %6s %8s %9s" % ("aero", "n", "p1", "p10", "p50", "p90", "p99", "p99.9", "max"))
for a in sorted(set(apt)):
    v = y[ok & (apt == a)] / 60
    print("  %-7s %8d %6.0f %6.0f %6.0f %6.0f %6.0f %8.0f %9.0f" % (
        a, len(v), *[np.percentile(v, q) for q in (1, 10, 50, 90, 99, 99.9)], v.max()))

print("\n=== 2. variance expliquee par la reference poste x piste (proxy de distance) ===")
r = X["ref_txot_p10"].to_numpy(dtype="float64")
print("  %-7s %12s %14s %16s" % ("aero", "corr(y,ref)", "R2 de ref seule", "etendue de ref (s)"))
for a in sorted(set(apt)):
    m = ok & (apt == a) & np.isfinite(r) & (y < 3600)
    if m.sum() < 500:
        continue
    c = np.corrcoef(r[m], y[m])[0, 1]
    print("  %-7s %12.3f %14.3f %16.0f" % (
        a, c, c ** 2, np.percentile(r[m], 95) - np.percentile(r[m], 5)))

print("\n=== 3. proxys d interference des arrivees deja presents ===")
cands = [c for c in X.columns if any(k in c for k in ("arr", "dep_proxy", "q_", "gap_"))]
print("  variables candidates:", ", ".join(cands[:12]))
print("\n  %-32s %10s %10s %10s" % ("variable", "LTFM", "EDDM", "LFPG"))
for c in cands[:12]:
    v = X[c].to_numpy(dtype="float64")
    out = []
    for a in ("LTFM", "EDDM", "LFPG"):
        m = ok & (apt == a) & np.isfinite(v) & (y < 3600)
        out.append(np.corrcoef(v[m], y[m])[0, 1] if m.sum() > 500 else np.nan)
    print("  %-32s %10.3f %10.3f %10.3f" % (c, *out))
