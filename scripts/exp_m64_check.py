"""M64: what does our submitted model actually predict on the disrupted test days?"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from dc26.data import load_ranking_2026

rk = load_ranking_2026()
sub = pd.read_parquet(ROOT / "submissions" / "reliable-hamburger_v11.parquet")
dep = rk[rk["PHASE_mvt"] == "DEP"][["MVT_ID_mvt", "AIRPORT", "MVT_TIME_UTC_mvt"]].copy()
dep["day"] = dep["MVT_TIME_UTC_mvt"].dt.date
d = dep.merge(sub, on="MVT_ID_mvt", how="left")

arr = rk[rk["PHASE_mvt"] == "ARR"].copy()
arr["day"] = arr["MVT_TIME_UTC_mvt"].dt.date
t = arr["TAXITIME_SEC_mvt"].astype(float)
arr = arr[np.isfinite(t) & (t > 0)]

print("=== nos predictions sur les journees perturbees ===")
print("  %-6s %-12s %7s %12s %14s %14s" % ("aero", "jour", "n", "notre pred.", "pred. normale", "arrivees p90"))
cases = [("EHAM", "2026-01-05"), ("EHAM", "2026-01-04"), ("EHAM", "2026-01-03"),
         ("LFPG", "2026-01-07"), ("LSZH", "2026-01-22"), ("EGLL", "2026-01-04")]
for ap, day in cases:
    dd = pd.to_datetime(day).date()
    m = (d["AIRPORT"] == ap) & (d["day"] == dd)
    ref = (d["AIRPORT"] == ap) & (d["day"] != dd)
    a = arr[(arr["AIRPORT"] == ap) & (arr["day"] == dd)]["TAXITIME_SEC_mvt"].astype(float)
    print("  %-6s %-12s %7d %12.0f %14.0f %14.0f" % (
        ap, day, m.sum(), d.loc[m, "TAXITIME_SEC_mvt"].mean(),
        d.loc[ref, "TAXITIME_SEC_mvt"].mean(), np.percentile(a, 90) if len(a) else -1))

print("\n=== reaction globale du modele a la perturbation ambiante ===")
p90 = arr.groupby(["AIRPORT", "day"], observed=True)["TAXITIME_SEC_mvt"].quantile(.9).rename("p90").reset_index()
norm = p90.groupby("AIRPORT", observed=True)["p90"].median().rename("normal")
p90 = p90.merge(norm, on="AIRPORT"); p90["ratio"] = p90["p90"] / p90["normal"]
d2 = d.merge(p90[["AIRPORT", "day", "ratio"]], on=["AIRPORT", "day"], how="left")
print("  %-18s %8s %14s" % ("severite du jour", "n", "pred. moyenne"))
for lo, hi, lab in [(0, 1.2, "normal (<1,2x)"), (1.2, 1.5, "1,2x a 1,5x"),
                    (1.5, 2.0, "1,5x a 2x"), (2.0, 99, "au-dela de 2x")]:
    m = (d2["ratio"] >= lo) & (d2["ratio"] < hi)
    if m.sum():
        print("  %-18s %8d %14.0f" % (lab, m.sum(), d2.loc[m, "TAXITIME_SEC_mvt"].mean()))
