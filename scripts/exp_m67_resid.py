"""M67: does OUR model's residual still carry per-entity structure?

If the blend already absorbed the AOBT_3 offset per stand/operator, its own
residual y - pred should show none. Whatever structure remains is a gain we have
not taken. Offsets fitted on TRAINING rows, measured on the holdout.
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
PROC, EXP = ROOT / "data" / "processed", ROOT / "experiments"
from dc26.validation import rmse

X = pd.read_parquet(PROC / "X_v2_mirror_t.parquet",
                    columns=["d_aobt3", "AIRPORT", "AIRCRAFT_OPERATOR_flt", "STAND", "RUNWAY"])
meta = pd.read_parquet(PROC / "meta_train2025.parquet")
y = meta["y"].astype("float64").to_numpy()
tr, va = meta["tr"].to_numpy(), meta["va"].to_numpy()
_p = np.load(EXP / "m49_blend_va.npy")
pred = np.full(len(meta), np.nan)
pred[np.where(meta["va"].to_numpy())[0]] = _p
art = np.zeros(len(y), bool)
for k in (1, 2, 3):
    art |= np.abs(y - k * 86400) < 3600
X["op"] = X["AIRPORT"].astype(str) + "|" + X["AIRCRAFT_OPERATOR_flt"].astype(str)

# the blend only exists on the holdout, so entity offsets must come from the
# residual of a model of the same family; use y - d_aobt3 on training rows as the
# proxy the model had access to, then check the blend residual on the holdout.
d = X["d_aobt3"].to_numpy(dtype="float64")
ok = (y > 0) & (y < 200000) & ~art
v = va & ok & np.isfinite(pred)
res = np.full(len(y), np.nan)
res[v] = y[v] - pred[v]
print("residu du modele sur le holdout : moyenne=%.1f  ecart-type=%.1f  n=%d" % (
    np.nanmean(res[v]), np.nanstd(res[v]), v.sum()))

print("\n=== structure residuelle par entite (holdout, apres notre modele) ===")
print("  %-10s %8s %14s %16s %14s" % ("entite", "groupes", "ecart-type", "part variance", "shrinkage"))
for key in ("op", "STAND", "RUNWAY"):
    k = X[key].astype(str).to_numpy()
    s = pd.Series(res[v]).groupby(k[v])
    cnt, mu = s.count(), s.mean()
    keep = cnt >= 200
    if keep.sum() < 5:
        continue
    w = cnt[keep] / cnt[keep].sum()
    gm = (mu[keep] * w).sum()
    between = np.sqrt(((mu[keep] - gm) ** 2 * w).sum())
    # sampling noise inflates 'between': subtract the expected within-group noise
    sw = np.nanstd(res[v])
    noise = np.sqrt((sw ** 2 / cnt[keep] * w).sum())
    true = np.sqrt(max(between ** 2 - noise ** 2, 0))
    print("  %-10s %8d %14.1f %15.2f%% %13.1f (vrai, hors bruit)" % (
        key, int(keep.sum()), between, 100 * (true / sw) ** 2, true))

print("\n=== gain reel : offset appris sur l entrainement, applique au holdout ===")
print("  %-10s %10s %12s %12s %10s" % ("entite", "min n", "couverture", "RMSE apres", "gain"))
r_tr = np.where(np.isfinite(d), y - d, np.nan)
f = tr & ok & np.isfinite(r_tr) & (np.abs(r_tr) < 3600)
base = rmse(y[v], pred[v])
print("  reference du modele actuel : %.2f" % base)
for key in ("op", "STAND"):
    k = X[key].astype(str).to_numpy()
    for mn in (200, 1000):
        s = pd.Series(r_tr[f]).groupby(k[f])
        cnt, mu = s.count(), s.mean()
        off = (mu * cnt / (cnt + 300.0))[cnt >= mn]   # empirical-Bayes shrinkage
        o = pd.Series(k[v]).map(off).to_numpy(dtype="float64")
        o = np.where(np.isfinite(o), o, 0.0)
        r = rmse(y[v], pred[v] + o)
        print("  %-10s %10d %11.1f%% %12.2f %+10.2f" % (key, mn, 100 * np.mean(o != 0), r, r - base))
