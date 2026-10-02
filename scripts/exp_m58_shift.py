"""M58: is arr_anom_n_30d cold-started in 2026? If the 30-day window has no
December 2025 history, the count ramps up through January and the feature means
something different at test time than during training. Same check for the PRU
monthly variable.
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
PROC = ROOT / "data" / "processed"

anom_t = pd.read_parquet(PROC / "anomreg_train2025.parquet")
anom_r = pd.read_parquet(PROC / "anomreg_ranking2026.parquet")
doy_t = pd.read_parquet(PROC / "X_v2_full_t.parquet", columns=["doy", "AIRPORT"])
doy_r = pd.read_parquet(PROC / "X_v2_ranking_t.parquet", columns=["doy", "AIRPORT"])
print("colonnes anomalie:", list(anom_t.columns))

c = "arr_anom_n_30d"
print("\n=== %s par tranche de jours ===" % c)
print("  %-22s %8s %12s" % ("periode", "n", "moyenne"))
for lab, d, dd, lo, hi in [("2025 jours 1-10", anom_t, doy_t, 1, 11),
                           ("2025 jours 11-20", anom_t, doy_t, 11, 21),
                           ("2025 jours 21-31", anom_t, doy_t, 21, 32),
                           ("2025 jours 60-90", anom_t, doy_t, 60, 91),
                           ("2026 jours 1-10", anom_r, doy_r, 1, 11),
                           ("2026 jours 11-20", anom_r, doy_r, 11, 21),
                           ("2026 jours 21-31", anom_r, doy_r, 21, 32)]:
    m = (dd["doy"] >= lo) & (dd["doy"] < hi)
    v = d.loc[m.to_numpy(), c].to_numpy(dtype="float64")
    print("  %-22s %8d %12.0f" % (lab, m.sum(), np.nanmean(v)))

print("\n=== toutes les variables d anomalie: janvier 2025 vs janvier 2026 ===")
jt = (doy_t["doy"] < 32).to_numpy()
jr = (doy_r["doy"] < 32).to_numpy()
print("  %-28s %14s %14s %10s" % ("variable", "jan 2025", "jan 2026", "rapport"))
for col in anom_t.columns:
    if anom_t[col].dtype.name == "category":
        continue
    a = np.nanmean(anom_t.loc[jt, col].to_numpy(dtype="float64"))
    b = np.nanmean(anom_r.loc[jr, col].to_numpy(dtype="float64"))
    print("  %-28s %14.4f %14.4f %10s" % (col, a, b, "%.2f" % (b / a) if a else "-"))

pru = PROC / "pru_month_features.parquet"
print("\n=== variable PRU mensuelle ===")
X = pd.read_parquet(PROC / "X_v2_full_t.parquet", columns=["doy"])
for f, lab in [("X_v2_full_t.parquet", "2025"), ("X_v2_ranking_t.parquet", "2026")]:
    try:
        v = pd.read_parquet(PROC / f, columns=["atxot_month_add_min"])["atxot_month_add_min"]
        print("  %s: moyenne=%.3f  ecart-type=%.3f  manquant=%.1f%%  valeurs distinctes=%d" % (
            lab, v.mean(), v.std(), 100 * v.isna().mean(), v.nunique()))
    except Exception as e:
        print("  %s: %s" % (lab, e))
