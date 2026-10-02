"""M62: is the LFPG 2026 anomaly spike a real operational regime or corrupted records?
The rate counts arrivals with taxi-in above one hour. Look at the actual values.
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from dc26.data import load_movements_2025, load_ranking_2026

mv, rk = load_movements_2025(), load_ranking_2026()
print("=== arrivees a LFPG : janvier 2025 vs janvier 2026 ===")
print("  %-14s %8s %10s %9s %9s %8s %8s %8s" % ("periode", "n", "median", ">30min", ">1h", ">3h", ">6h", "max"))
for lab, df in [("jan 2025", mv), ("test 2026", rk)]:
    a = df[(df["PHASE_mvt"] == "ARR") & (df["AIRPORT"] == "LFPG")].copy()
    if lab == "jan 2025":
        a = a[a["MVT_TIME_UTC_mvt"].dt.month == 1]
    t = a["TAXITIME_SEC_mvt"].to_numpy(dtype=float)
    t = t[np.isfinite(t) & (t > 0)]
    print("  %-14s %8d %10.0f %8d %9d %8d %8d %8.0f" % (
        lab, len(t), np.median(t), (t > 1800).sum(), (t > 3600).sum(),
        (t > 10800).sum(), (t > 21600).sum(), t.max()))

a = rk[(rk["PHASE_mvt"] == "ARR") & (rk["AIRPORT"] == "LFPG")]
t = a["TAXITIME_SEC_mvt"].to_numpy(dtype=float)
big = a[np.isfinite(t) & (t > 3600)]
print("\n=== les %d arrivees LFPG 2026 au-dela d une heure ===" % len(big))
v = np.sort(big["TAXITIME_SEC_mvt"].to_numpy(dtype=float))
print("  valeurs:", " ".join("%.0f" % x for x in v[:25]), "..." if len(v) > 25 else "")
d = big["MVT_TIME_UTC_mvt"].dt.date.value_counts().sort_values(ascending=False)
print("  dates les plus touchees:", ", ".join("%s (%d)" % (k, n) for k, n in d.head(5).items()))
print("  reparties sur %d jours distincts" % len(d))
