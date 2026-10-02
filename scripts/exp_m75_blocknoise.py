"""M75: is BLOCK_TIME itself noisy at LIRF/LTFM/EGLL?

Departure taxi-out = MVT - BLOCK and arrival taxi-in = BLOCK - MVT both depend on
the airport's BLOCK_TIME. If that timestamp is noisy at an airport, BOTH become
less predictable there. Arrivals are fully labelled in train and test, so their
predictability is measurable directly.

Predictor deliberately simple and identical everywhere: median taxi time per
(stand, runway, hour-of-day), fitted on 11 months, scored on the twelfth. Whatever
it cannot explain is either genuine operational variance or timestamp noise - and
comparing the two phases separates an airport-wide timestamp problem from a
departure-specific modelling gap.
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from dc26.data import load_movements_2025

mv = load_movements_2025()
mv = mv[mv["TAXITIME_SEC_mvt"].notna()].copy()
mv["t"] = mv["TAXITIME_SEC_mvt"].astype(float)
mv = mv[(mv["t"] > 0) & (mv["t"] < 3600)]
mv["hour"] = mv["MVT_TIME_UTC_mvt"].dt.hour
mv["month"] = mv["MVT_TIME_UTC_mvt"].dt.month

print("=== predictibilite des deux phases, meme predicteur partout ===")
print("  %-7s %9s %8s %8s %9s %8s %8s %10s" % (
    "aero", "n dep", "R2 dep", "sd dep", "n arr", "R2 arr", "sd arr", "desaccord"))
rows = []
for a in sorted(mv["AIRPORT"].astype(str).unique()):
    out = {}
    for ph in ("DEP", "ARR"):
        d = mv[(mv["AIRPORT"] == a) & (mv["PHASE_mvt"] == ph)]
        if len(d) < 5000:
            out[ph] = (np.nan, np.nan, 0)
            continue
        fit, test = d[d["month"] < 12], d[d["month"] == 12]
        if len(test) < 500:
            out[ph] = (np.nan, np.nan, 0)
            continue
        k = ["STAND_mvt", "RUNWAY_mvt", "hour"]
        med = fit.groupby(k, observed=True)["t"].median().rename("p")
        g = test.join(med, on=k)
        p = g["p"].fillna(fit["t"].median()).to_numpy()
        yv = g["t"].to_numpy()
        out[ph] = (1 - ((yv - p) ** 2).mean() / yv.var(), yv.std(), len(test))
    # AOBT_3 vs BLOCK disagreement, departures
    dd = mv[(mv["AIRPORT"] == a) & (mv["PHASE_mvt"] == "DEP")]
    r = (dd["MVT_TIME_UTC_mvt"] - dd["AOBT_3_flt"]).dt.total_seconds() - dd["t"]
    r = r[np.isfinite(r) & (r.abs() < 3600)]
    rows.append((a, out["DEP"], out["ARR"], r.std()))
    print("  %-7s %9d %8.3f %8.1f %9d %8.3f %8.1f %10.1f" % (
        a, out["DEP"][2], out["DEP"][0], out["DEP"][1],
        out["ARR"][2], out["ARR"][0], out["ARR"][1], r.std()))

print("\n=== correlations entre les trois indicateurs (10 aeroports) ===")
dep_r2 = np.array([x[1][0] for x in rows]); arr_r2 = np.array([x[2][0] for x in rows])
dis = np.array([x[3] for x in rows])
m = np.isfinite(dep_r2) & np.isfinite(arr_r2) & np.isfinite(dis)
print("  R2 depart  vs R2 arrivee   : %+.3f" % np.corrcoef(dep_r2[m], arr_r2[m])[0, 1])
print("  R2 depart  vs desaccord    : %+.3f" % np.corrcoef(dep_r2[m], dis[m])[0, 1])
print("  R2 arrivee vs desaccord    : %+.3f" % np.corrcoef(arr_r2[m], dis[m])[0, 1])
