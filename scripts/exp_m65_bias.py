"""M65: on the 2025 holdout, is the model biased on disrupted days?
Free test - reuses the stored holdout predictions, no retraining.
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
PROC, EXP = ROOT / "data" / "processed", ROOT / "experiments"
from dc26.data import load_movements_2025
from dc26.validation import rmse

meta = pd.read_parquet(PROC / "meta_train2025.parquet")
X = pd.read_parquet(PROC / "X_v2_mirror_t.parquet", columns=["AIRPORT"])
y = meta["y"].astype("float64").to_numpy()
va = meta["va"].to_numpy()
pred = np.load(EXP / "m49_blend_va.npy")
mv = load_movements_2025()

dep = mv[mv["PHASE_mvt"] == "DEP"].reset_index(drop=True)
assert len(dep) == len(meta), "alignement: %d vs %d" % (len(dep), len(meta))
day = dep["MVT_TIME_UTC_mvt"].dt.date.to_numpy()

arr = mv[mv["PHASE_mvt"] == "ARR"].copy()
arr["day"] = arr["MVT_TIME_UTC_mvt"].dt.date
t = arr["TAXITIME_SEC_mvt"].astype(float)
arr = arr[np.isfinite(t) & (t > 0)]
p90 = arr.groupby(["AIRPORT", "day"], observed=True)["TAXITIME_SEC_mvt"].quantile(.9).rename("p90").reset_index()
norm = p90.groupby("AIRPORT", observed=True)["p90"].median().rename("normal")
p90 = p90.merge(norm, on="AIRPORT"); p90["ratio"] = p90["p90"] / p90["normal"]

key = pd.DataFrame({"AIRPORT": X["AIRPORT"].astype(str).to_numpy(), "day": day})
ratio = key.merge(p90[["AIRPORT", "day", "ratio"]], on=["AIRPORT", "day"], how="left")["ratio"].to_numpy()

art = np.zeros(len(y), bool)
for k in (1, 2, 3):
    art |= np.abs(y - k * 86400) < 3600
rv, yv, av = ratio[va], y[va], art[va]
g = np.isfinite(pred) & ~av & np.isfinite(rv)

print("=== biais et erreur du modele selon la severite du jour (holdout 2025) ===")
print("  %-20s %8s %12s %12s %10s %9s" % ("severite", "n", "pred. moy", "verite moy", "biais", "RMSE"))
for lo, hi, lab in [(0, 1.2, "normal (<1,2x)"), (1.2, 1.5, "1,2x a 1,5x"),
                    (1.5, 2.0, "1,5x a 2x"), (2.0, 99, "au-dela de 2x")]:
    m = g & (rv >= lo) & (rv < hi)
    if m.sum() < 20:
        continue
    print("  %-20s %8d %12.0f %12.0f %+10.0f %9.1f" % (
        lab, m.sum(), pred[m].mean(), yv[m].mean(), yv[m].mean() - pred[m].mean(), rmse(yv[m], pred[m])))

m = g & (rv >= 1.5)
if m.sum() > 20:
    b = yv[m].mean() - pred[m].mean()
    r0, r1 = rmse(yv[m], pred[m]), rmse(yv[m], pred[m] + b)
    print("\n  correction du biais sur les jours >=1,5x : RMSE %.1f -> %.1f (%+.1f) sur %d vols" % (
        r0, r1, r1 - r0, m.sum()))
    print("  a l echelle du test (2469 vols concernes) : gain global estime = %.3f" % (
        (np.sqrt((215876 * 260.13 ** 2 - 2469 * (r0 ** 2 - r1 ** 2)) / 215876)) - 260.13))
