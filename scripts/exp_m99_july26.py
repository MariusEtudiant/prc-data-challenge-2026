"""M99: v14 scored 343.31 while the mirror predicted ~294, and the mirror is
pessimistic by ~7 points - so July 2026 must be harder than July 2025, not easier.

Arrivals keep their taxi times in the ranking file, so the 2026 regime is directly
measurable and comparable with 2025 on the same quantity.
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
PROC = ROOT / "data" / "processed"

mv = pd.read_parquet(PROC / "movements_2025.parquet",
                     columns=["AIRPORT", "PHASE_mvt", "MVT_TIME_UTC_mvt", "TAXITIME_SEC_mvt"])
rk = pd.read_parquet(PROC / "ranking_2026.parquet",
                     columns=["ADEP_mvt", "ADES_mvt", "PHASE_mvt", "MVT_TIME_UTC_mvt", "TAXITIME_SEC_mvt"])
rk["AIRPORT"] = np.where(rk["PHASE_mvt"] == "DEP", rk["ADEP_mvt"], rk["ADES_mvt"])


def arr(df, month):
    a = df[(df["PHASE_mvt"] == "ARR") & (df["MVT_TIME_UTC_mvt"].dt.month == month)].copy()
    t = a["TAXITIME_SEC_mvt"].astype(float)
    return a[np.isfinite(t) & (t > 0) & (t < 200000)]


print("=== arrivees : juillet 2025 contre juillet 2026, par aeroport ===")
print("  %-7s %8s %8s %9s %9s %9s %9s %9s %9s" % (
    "aero", "n 25", "n 26", "med 25", "med 26", "p95 25", "p95 26", ">1h 25", ">1h 26"))
a25, a26 = arr(mv, 7), arr(rk, 7)
for ap in sorted(a26["AIRPORT"].astype(str).unique()):
    s25 = a25.loc[a25["AIRPORT"].astype(str) == ap, "TAXITIME_SEC_mvt"].astype(float)
    s26 = a26.loc[a26["AIRPORT"].astype(str) == ap, "TAXITIME_SEC_mvt"].astype(float)
    if len(s25) < 500 or len(s26) < 500:
        continue
    print("  %-7s %8d %8d %9.0f %9.0f %9.0f %9.0f %8.2f%% %8.2f%%" % (
        ap, len(s25), len(s26), s25.median(), s26.median(),
        np.percentile(s25, 95), np.percentile(s26, 95),
        100 * (s25 > 3600).mean(), 100 * (s26 > 3600).mean()))

print("\n=== janvier, comme controle (meme jeu qu'avant, score connu) ===")
print("  %-7s %9s %9s %9s %9s" % ("aero", "med 25", "med 26", ">1h 25", ">1h 26"))
j25, j26 = arr(mv, 1), arr(rk, 1)
for ap in sorted(j26["AIRPORT"].astype(str).unique()):
    s25 = j25.loc[j25["AIRPORT"].astype(str) == ap, "TAXITIME_SEC_mvt"].astype(float)
    s26 = j26.loc[j26["AIRPORT"].astype(str) == ap, "TAXITIME_SEC_mvt"].astype(float)
    if len(s25) < 500 or len(s26) < 500:
        continue
    print("  %-7s %9.0f %9.0f %8.2f%% %8.2f%%" % (
        ap, s25.median(), s26.median(), 100 * (s25 > 3600).mean(), 100 * (s26 > 3600).mean()))

print("\n=== ce que notre modele predit en juillet 2026 vs la realite 2025 des DEPARTS ===")
sub = pd.read_parquet(ROOT / "submissions" / "reliable-hamburger_v14.parquet")
d26 = rk[(rk["PHASE_mvt"] == "DEP") & (rk["MVT_TIME_UTC_mvt"].dt.month == 7)]
ids = pd.read_parquet(ROOT / "data" / "ranking.parquet", columns=["MVT_ID_mvt", "PHASE_mvt", "MVT_TIME_UTC_mvt", "ADEP_mvt"])
ids = ids[(ids["PHASE_mvt"] == "DEP") & (ids["MVT_TIME_UTC_mvt"].dt.month == 7)].merge(sub, on="MVT_ID_mvt")
d25 = mv[(mv["PHASE_mvt"] == "DEP") & (mv["MVT_TIME_UTC_mvt"].dt.month == 7)]
t25 = d25["TAXITIME_SEC_mvt"].astype(float)
d25 = d25[np.isfinite(t25) & (t25 > 0) & (t25 < 7200)]
print("  %-7s %14s %16s %10s" % ("aero", "reel jul 2025", "predit jul 2026", "ecart"))
for ap in sorted(ids["ADEP_mvt"].astype(str).unique()):
    r = d25.loc[d25["AIRPORT"].astype(str) == ap, "TAXITIME_SEC_mvt"].astype(float).mean()
    p = ids.loc[ids["ADEP_mvt"].astype(str) == ap, "TAXITIME_SEC_mvt"].mean()
    print("  %-7s %14.0f %16.0f %+10.0f" % (ap, r, p, p - r))
