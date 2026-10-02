"""M84: the same defect may sit in early stopping.

The gate fix worked because a least-squares objective on heavy-tailed data is
dominated by a few flights. Early stopping uses RMSE on the ES window, so if that
window holds a couple of 80,000s flights, `best_iteration` is chosen largely by
how the model happens to score on them - i.e. by noise.
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
PROC = ROOT / "data" / "processed"

X = pd.read_parquet(PROC / "X_v2_mirror_t.parquet", columns=["AIRPORT"])
meta = pd.read_parquet(PROC / "meta_train2025.parquet")
y = meta["y"].astype("float64").to_numpy()
tr, es = meta["tr"].to_numpy(), meta["es"].to_numpy()
ok = (y > 0) & (y < 200000)
art = np.zeros(len(y), bool)
for k in (1, 2, 3):
    art |= np.abs(y - k * 86400) < 3600
ok &= ~art
apt = X["AIRPORT"].astype(str).to_numpy()

print("=== domination de la fenetre d arret anticipe par ses valeurs extremes ===")
print("  %-7s %8s %8s %9s %13s %16s" % ("aero", "n ES", ">2h", ">5h",
                                        "part err2 du", "part err2 des"))
print("  %-7s %8s %8s %9s %13s %16s" % ("", "", "", "", "pire vol", "10 pires"))
for a in sorted(set(apt)):
    m = ok & tr & es & (apt == a)
    if m.sum() < 200:
        continue
    v = y[m]
    # error of a constant median predictor: shows how the tail dominates any L2 metric
    r = (v - np.median(v)) ** 2
    o = np.sort(r)[::-1]
    print("  %-7s %8d %8d %9d %12.1f%% %15.1f%%" % (
        a, m.sum(), (v > 7200).sum(), (v > 18000).sum(),
        100 * o[0] / r.sum(), 100 * o[:10].sum() / r.sum()))

print("\n  -> une part elevee signifie que la RMSE d arret est decidee par ces vols,")
print("     donc que l iteration retenue est en grande partie du hasard.")
