"""M66: is the AOBT_3 error structured rather than random?

d_aobt3 = MVT - AOBT_3 is Network Manager's view of the taxi time; the target is
MVT - BLOCK_TIME, the airport's own. The residual r = y - d_aobt3 has std ~355s
and dominates everything else. If the two sources define off-block differently
per operator or per stand (brake release vs pushback start), r carries a
systematic component that a tree can only approximate through splits.

Measured strictly out-of-sample: offsets fitted on the training months, evaluated
on the mirror holdout.
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
PROC = ROOT / "data" / "processed"

X = pd.read_parquet(PROC / "X_v2_mirror_t.parquet",
                    columns=["d_aobt3", "AIRPORT", "AIRCRAFT_OPERATOR_flt", "STAND", "RUNWAY"])
meta = pd.read_parquet(PROC / "meta_train2025.parquet")
y = meta["y"].astype("float64").to_numpy()
tr, va = meta["tr"].to_numpy(), meta["va"].to_numpy()
ok = (y > 0) & (y < 200000)
art = np.zeros(len(y), bool)
for k in (1, 2, 3):
    art |= np.abs(y - k * 86400) < 3600
ok &= ~art
d = X["d_aobt3"].to_numpy(dtype="float64")
r = y - d
base = ok & np.isfinite(d) & (np.abs(r) < 3600)
print("residu r = y - d_aobt3 sur %d vols : moyenne=%.1f  ecart-type=%.1f  median=%.1f" % (
    base.sum(), r[base].mean(), r[base].std(), np.median(r[base])))

X["op"] = X["AIRPORT"].astype(str) + "|" + X["AIRCRAFT_OPERATOR_flt"].astype(str)
print("\n=== le residu varie-t-il systematiquement selon l entite ? ===")
print("  %-12s %8s %12s %12s %14s" % ("entite", "groupes", "ecart-type", "part variance", "gain theorique"))
for key in ("op", "STAND", "RUNWAY"):
    k = X[key].astype(str).to_numpy()
    s = pd.Series(r[base]).groupby(k[base])
    cnt, mu = s.count(), s.mean()
    keep = cnt >= 200
    if keep.sum() < 5:
        continue
    w = cnt[keep] / cnt[keep].sum()
    between = np.sqrt(((mu[keep] - (mu[keep] * w).sum()) ** 2 * w).sum())
    tot = r[base].std()
    print("  %-12s %8d %12.1f %11.1f%% %13.1fs" % (
        key, int(keep.sum()), between, 100 * (between / tot) ** 2, between))

print("\n=== gain reel : appliquer l offset appris sur l entrainement au holdout ===")
from dc26.validation import rmse
for key in ("op", "STAND"):
    k = X[key].astype(str).to_numpy()
    f = base & tr
    s = pd.Series(r[f]).groupby(k[f])
    cnt, mu = s.count(), s.mean()
    off = mu[cnt >= 200]
    v = base & va
    o = pd.Series(k[v]).map(off).to_numpy(dtype="float64")
    o = np.where(np.isfinite(o), o, 0.0)
    cov = 100 * np.mean(o != 0)
    print("  %-8s couverture %.1f%% | RMSE de d_aobt3 seul : %.1f -> %.1f (%+.1f) avec offset" % (
        key, cov, rmse(y[v], d[v]), rmse(y[v], d[v] + o), rmse(y[v], d[v] + o) - rmse(y[v], d[v])))
