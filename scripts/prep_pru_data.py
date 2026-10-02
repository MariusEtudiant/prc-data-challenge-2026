"""Convert PRU xlsx open data to filtered parquet for our 10 airports.
Outputs in data/external/:
  pru_atfm_daily.parquet   - daily arrival ATFM delay by airport & cause (2025..2026-07)
  pru_atxot_monthly.parquet- monthly additional taxi-out per airport
  pru_traffic_daily.parquet- daily dep/arr counts per airport
"""
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from dc26.data import AIRPORTS

EXT = ROOT / "data" / "external"

print("parsing ATXOT...", flush=True)
a = pd.read_excel(EXT / "Taxi-Out_Additional_Time.xlsx", sheet_name="DATA")
a = a[a["APT_ICAO"].isin(AIRPORTS) & (a["YEAR"] >= 2024)].copy()
a["add_min_per_dep"] = a["TOTAL_ADD_TIME_MIN"] / a["VALID_FL"]
a["ref_min_per_dep"] = a["TOTAL_REF_TIME_MIN"] / a["TOTAL_REF_NB_FL"]
a[["YEAR", "MONTH_NUM", "APT_ICAO", "VALID_FL", "add_min_per_dep", "ref_min_per_dep"]].to_parquet(
    EXT / "pru_atxot_monthly.parquet", index=False)
print(f"  atxot: {len(a)} rows, {a['YEAR'].min()}-{a['YEAR'].max()}", flush=True)

print("parsing ATFM (big)...", flush=True)
d = pd.read_excel(EXT / "Airport_Arrival_ATFM_Delay.xlsx", sheet_name="DATA")
d = d[d["APT_ICAO"].isin(AIRPORTS) & (d["YEAR"] >= 2024)].copy()
d["FLT_DATE"] = pd.to_datetime(d["FLT_DATE"])
out = pd.DataFrame({
    "date": d["FLT_DATE"], "airport": d["APT_ICAO"],
    "arr_flights": d["FLT_ARR_1"],
    "dly_total_min": d["DLY_APT_ARR_1"],
    "dly_weather_min": d["DLY_APT_ARR_W_1"].fillna(0) + d["DLY_APT_ARR_D_1"].fillna(0),
    "dly_capacity_min": d["DLY_APT_ARR_C_1"].fillna(0) + d["DLY_APT_ARR_S_1"].fillna(0)
                        + d["DLY_APT_ARR_G_1"].fillna(0),
    "n_delayed_15": d["FLT_ARR_1_DLY_15"],
})
out.to_parquet(EXT / "pru_atfm_daily.parquet", index=False)
print(f"  atfm: {len(out)} rows, {out['date'].min().date()} .. {out['date'].max().date()}", flush=True)

print("parsing traffic (big)...", flush=True)
t = pd.read_excel(EXT / "Airport_Traffic.xlsx", sheet_name="DATA")
t = t[t["APT_ICAO"].isin(AIRPORTS) & (t["YEAR"] >= 2024)].copy()
t["FLT_DATE"] = pd.to_datetime(t["FLT_DATE"])
t[["FLT_DATE", "APT_ICAO", "FLT_DEP_1", "FLT_ARR_1", "FLT_TOT_1"]].rename(
    columns={"FLT_DATE": "date", "APT_ICAO": "airport"}).to_parquet(
    EXT / "pru_traffic_daily.parquet", index=False)
print(f"  traffic: rows for our airports, {t['FLT_DATE'].min().date()} .. {t['FLT_DATE'].max().date()}", flush=True)
print("done", flush=True)
