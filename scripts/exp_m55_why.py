"""M55: why did the correction fail? Because the model is NOT blind to these flights."""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
PROC, EXP = ROOT / "data" / "processed", ROOT / "experiments"

X = pd.read_parquet(PROC / "X_v2_mirror_t.parquet", columns=["d_sched", "has_fpl"])
meta = pd.read_parquet(PROC / "meta_train2025.parquet")
y = meta["y"].astype("float64").to_numpy()
va = meta["va"].to_numpy()
pred = np.load(EXP / "m49_blend_va.npy")
y_va, ds_va, fp_va = y[va], X["d_sched"].to_numpy()[va], X["has_fpl"].to_numpy()[va]

print("=== tous les vols du holdout dont la verite depasse 2h ===")
m = np.where((y_va > 7200) & np.isfinite(pred))[0]
m = m[np.argsort(-y_va[m])]
print("  %10s %10s %10s %8s" % ("verite", "prediction", "d_sched", "fpl"))
for i in m[:14]:
    print("  %10.0f %10.0f %10s %8d" % (y_va[i], pred[i],
          "%.0f" % ds_va[i] if np.isfinite(ds_va[i]) else "absent", fp_va[i]))
print("  ... %d vols au total > 2h" % len(m))

print("\n=== le modele reagit-il deja a d_sched ? ===")
for lo, hi in [(0, 7200), (7200, 43200), (43200, 86400), (86400, 1e9)]:
    s = np.isfinite(ds_va) & (ds_va >= lo) & (ds_va < hi) & np.isfinite(pred)
    if s.sum() < 3:
        continue
    print("  d_sched %6.0f-%-8.0f n=%6d | prediction moyenne=%8.0fs | verite moyenne=%8.0fs" % (
        lo, hi, s.sum(), pred[s].mean(), y_va[s].mean()))
