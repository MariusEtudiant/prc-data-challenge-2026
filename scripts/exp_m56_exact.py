"""M56: on the odd flights, truth equals d_sched almost EXACTLY (56633 vs 56636,
41766 vs 41763, 39837 vs 39838...). That means BLOCK_TIME == SCHED_TIME for them.
The model approximates this relation; using it directly would be exact.
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
PROC, EXP = ROOT / "data" / "processed", ROOT / "experiments"
from dc26.validation import rmse

X = pd.read_parquet(PROC / "X_v2_mirror_t.parquet", columns=["d_sched", "has_fpl"])
meta = pd.read_parquet(PROC / "meta_train2025.parquet")
y = meta["y"].astype("float64").to_numpy()
tr, va = meta["tr"].to_numpy(), meta["va"].to_numpy()
ok = (y > 0) & (y < 200000)
ds, fp = X["d_sched"].to_numpy(), X["has_fpl"].to_numpy()
eq = np.isfinite(ds) & (np.abs(y - ds) < 60)

print("=== P(verite == d_sched a 60s pres) ===")
print("  %-34s %9s %9s" % ("population", "n", "P(egal)"))
for lab, m in [("tous", ok & np.isfinite(ds)),
               ("sans plan de vol", ok & np.isfinite(ds) & (fp == 0)),
               ("sans FPL & d_sched>2h", ok & np.isfinite(ds) & (fp == 0) & (ds > 7200)),
               ("sans FPL & d_sched>6h", ok & np.isfinite(ds) & (fp == 0) & (ds > 21600)),
               ("avec FPL & d_sched>2h", ok & np.isfinite(ds) & (fp == 1) & (ds > 7200))]:
    print("  %-34s %9d %8.1f%%" % (lab, m.sum(), 100 * eq[m].mean() if m.sum() else 0))

print("\n=== regle : si pas de plan de vol et d_sched > seuil, predire d_sched ===")
pred = np.load(EXP / "m49_blend_va.npy")
y_va, ds_va, fp_va = y[va], ds[va], fp[va]
g = np.isfinite(pred)
base = rmse(y_va[g], pred[g])
print("  RMSE holdout de reference = %.2f" % base)
for thr in (3600, 7200, 14400, 21600, 43200):
    d_tr = ok & tr & np.isfinite(ds) & (fp == 0) & (ds > thr)
    d_va = np.isfinite(ds_va) & (fp_va == 0) & (ds_va > thr) & g
    if d_va.sum() < 2:
        continue
    c = pred.copy()
    c[d_va] = np.clip(ds_va[d_va], 60, 86400)
    r = rmse(y_va[g], c[g])
    print("  seuil %6ds : %4d cand. entrainement (P(egal)=%4.1f%%) | %3d dans holdout -> RMSE %.2f (%+.2f)" % (
        thr, d_tr.sum(), 100 * eq[d_tr].mean(), d_va.sum(), r, r - base))
