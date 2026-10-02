"""M73: why is LIRF at R2 0.26 when every other airport is at 0.61-0.86?

Three candidate causes, each testable:
  1. noisy target - the airport's own BLOCK_TIME disagrees with NM's AOBT_3 more
     than elsewhere, which caps any model
  2. starved model - too few training rows for a per-airport model
  3. missing inputs - the features that carry the signal elsewhere are absent here
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
PROC = ROOT / "data" / "processed"

X = pd.concat([pd.read_parquet(PROC / f) for f in
               ["X_v2_mirror_t.parquet", "flightctx_train2025.parquet",
                "depregime_train2025.parquet", "aobtq_train2025.parquet"]], axis=1)
meta = pd.read_parquet(PROC / "meta_train2025.parquet")
y = meta["y"].astype("float64").to_numpy()
tr = meta["tr"].to_numpy()
apt = X["AIRPORT"].astype(str).to_numpy()
d = X["d_aobt3"].to_numpy(dtype="float64")
ok = (y > 0) & (y < 200000) & (y < 3600)
art = np.zeros(len(y), bool)
for k in (1, 2, 3):
    art |= np.abs(y - k * 86400) < 3600
ok &= ~art

print("=== 1. bruit de la cible : desaccord entre BLOCK_TIME et AOBT_3 ===")
print("  %-7s %10s %12s %12s %12s" % ("aero", "n", "ecart-type r", "IQR r", "part |r|>300s"))
for a in sorted(set(apt)):
    m = ok & (apt == a) & np.isfinite(d)
    r = (y - d)[m]
    r = r[np.abs(r) < 3600]
    if len(r) < 500:
        continue
    print("  %-7s %10d %12.1f %12.1f %11.1f%%" % (
        a, len(r), r.std(), np.percentile(r, 75) - np.percentile(r, 25),
        100 * np.mean(np.abs(r) > 300)))

print("\n=== 2. volume d entrainement par aeroport ===")
print("  %-7s %12s %12s" % ("aero", "lignes train", "vs LEBL"))
cnt = {a: int((ok & tr & (apt == a)).sum()) for a in sorted(set(apt))}
ref = cnt.get("LEBL", 1)
for a, n in sorted(cnt.items(), key=lambda kv: kv[1]):
    print("  %-7s %12d %11.2fx" % (a, n, n / ref))

print("\n=== 3. disponibilite des variables cles a LIRF vs ailleurs ===")
key = ["d_aobt3", "d_eobt1", "d_lobt", "d_iobt", "taxi_upper_bound_s", "inbound_taxiin_s",
       "turnaround_avail_s", "q_dep_aobt_to_takeoff_runway", "implied_add"]
key = [c for c in key if c in X.columns]
print("  %-32s %12s %12s" % ("variable", "LIRF absent", "autres absent"))
mi = ok & (apt == "LIRF")
mo = ok & (apt != "LIRF")
for c in key:
    print("  %-32s %11.1f%% %11.1f%%" % (c, 100 * X.loc[mi, c].isna().mean(),
                                         100 * X.loc[mo, c].isna().mean()))

print("\n=== pouvoir predictif brut des variables a LIRF (correlation avec y) ===")
print("  %-32s %10s %10s" % ("variable", "LIRF", "EDDM"))
for c in key:
    v = X[c].to_numpy(dtype="float64")
    out = []
    for a in ("LIRF", "EDDM"):
        m = ok & (apt == a) & np.isfinite(v)
        out.append(np.corrcoef(v[m], y[m])[0, 1] if m.sum() > 500 else np.nan)
    print("  %-32s %10.3f %10.3f" % (c, out[0], out[1]))
