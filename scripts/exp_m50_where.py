"""M50: where does the squared error actually sit, and can recalibration recover any?"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
PROC, EXP = ROOT / "data" / "processed", ROOT / "experiments"

X = pd.concat([pd.read_parquet(PROC / f) for f in
               ["X_v2_mirror_t.parquet", "flightctx_train2025.parquet"]], axis=1)
meta = pd.read_parquet(PROC / "meta_train2025.parquet")
y = meta["y"].astype("float64")
va = meta["va"].to_numpy()
y_va = y[va].to_numpy()
pred = np.load(EXP / "m49_blend_va.npy")
vp = meta.loc[va, "val_part"].to_numpy()
g = np.isfinite(pred)
sq = (y_va - pred) ** 2
tot = sq[g].sum()

print("=== ou est l erreur quadratique (holdout miroir, n=%d) ===" % g.sum())
print("  %-22s %8s %8s %12s %8s" % ("tranche de verite", "n", "part n", "part err2", "RMSE si parfait"))
edges = [0, 3600, 7200, 18000, 60000, 1e9]
labs = ["< 1h (normal)", "1h - 2h", "2h - 5h", "5h - 16h", "> 16h (artefacts)"]
for i, lab in enumerate(labs):
    m = g & (y_va >= edges[i]) & (y_va < edges[i + 1])
    if not m.any():
        continue
    part = sq[m].sum() / tot
    rest = np.sqrt((tot - sq[m].sum()) / g.sum())
    print("  %-22s %8d %7.3f%% %11.1f%% %8.1f" % (lab, m.sum(), 100 * m.sum() / g.sum(),
                                                  100 * part, rest))
print("  RMSE reel = %.1f" % np.sqrt(tot / g.sum()))

# --- isotonic recalibration, fitted on January, measured on July (honest)
from sklearn.isotonic import IsotonicRegression
from dc26.validation import rmse
jan, jul = (vp == "jan") & g, (vp == "jul") & g
iso = IsotonicRegression(out_of_bounds="clip").fit(pred[jan], y_va[jan])
adj = iso.predict(pred[jul])
print("\n=== recalibration isotone (ajustee sur janvier, mesuree sur juillet) ===")
print("  juillet brut       = %.2f" % rmse(y_va[jul], pred[jul]))
print("  juillet recalibre  = %.2f  (%+.2f)" % (rmse(y_va[jul], adj), rmse(y_va[jul], adj) - rmse(y_va[jul], pred[jul])))
b = y_va[jan].mean() - pred[jan].mean()
print("  juillet + biais constant (%+.1fs) = %.2f  (%+.2f)" % (
    b, rmse(y_va[jul], pred[jul] + b), rmse(y_va[jul], pred[jul] + b) - rmse(y_va[jul], pred[jul])))
