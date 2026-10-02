"""M61: is the 2026 test inside the training envelope, per airport?

January 2026 shows 4-6x the arrival-anomaly rate of January 2025. That is only a
problem if 2025 never reached those levels: a model interpolates safely inside
the range it saw, and extrapolates blindly outside it.
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
PROC = ROOT / "data" / "processed"

C = "arr_anom_rate_3600s_30d"
at = pd.read_parquet(PROC / "anomreg_train2025.parquet", columns=[C])[C].to_numpy(dtype="float64")
ar = pd.read_parquet(PROC / "anomreg_ranking2026.parquet", columns=[C])[C].to_numpy(dtype="float64")
mt = pd.read_parquet(PROC / "X_v2_full_t.parquet", columns=["AIRPORT", "doy", "month"])
mr = pd.read_parquet(PROC / "X_v2_ranking_t.parquet", columns=["AIRPORT", "doy", "month"])

print("=== taux d anomalie d arrivee (fenetre 30j) : 2025 vs test 2026, par aeroport ===")
print("  %-8s %10s %10s %10s %12s %14s" % ("aero", "2025 med", "2025 p95", "2025 max",
                                           "2026 median", "centile 2025"))
flag = []
for a in sorted(mt["AIRPORT"].astype(str).unique()):
    st = at[(mt["AIRPORT"].astype(str) == a).to_numpy()]
    sr = ar[(mr["AIRPORT"].astype(str) == a).to_numpy()]
    st, sr = st[np.isfinite(st)], sr[np.isfinite(sr)]
    if not len(st) or not len(sr):
        continue
    med = np.median(sr)
    pct = 100 * (st < med).mean()
    print("  %-8s %10.5f %10.5f %10.5f %12.5f %13.1f%%" % (
        a, np.median(st), np.percentile(st, 95), st.max(), med, pct))
    if pct > 95:
        flag.append((a, pct, med, st.max()))

print("\n=== aeroports dont le regime de test depasse le 95e centile de 2025 ===")
if flag:
    for a, p, med, mx in flag:
        print("  %-8s median 2026=%.5f est au %.1f centile de 2025 (max 2025=%.5f)" % (a, med, p, mx))
else:
    print("  aucun - le test reste dans l enveloppe de 2025")

thr99 = np.nanpercentile(at, 99)
frac = 100 * np.nanmean(ar > thr99)
print("\npart des lignes de test au-dela du 99e centile de 2025 : %.1f%%" % frac)
