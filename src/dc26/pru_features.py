"""Features from PRU open data: daily arrival ATFM delays by cause (covers Jan+Jul 2026)
and monthly additional taxi-out. Joined on (airport, UTC date of takeoff)."""
from pathlib import Path

import numpy as np
import pandas as pd

from .data import ROOT

EXT = ROOT / "data" / "external"

ATFM_COLS = ["atfm_dly_per_arr", "atfm_wx_per_arr", "atfm_cap_per_arr",
             "atfm_share_dly15", "atfm_dly_per_arr_prevday"]
ATXOT_COLS = ["atxot_month_add_min"]


def build_pru_features(df: pd.DataFrame) -> pd.DataFrame:
    atfm = pd.read_parquet(EXT / "pru_atfm_daily.parquet")
    atfm["date"] = pd.to_datetime(atfm["date"]).dt.date
    arr = atfm["arr_flights"].replace(0, np.nan)
    # NaN in the PRU delay columns means "no delay recorded" -> 0
    atfm["atfm_dly_per_arr"] = atfm["dly_total_min"].fillna(0) / arr
    atfm["atfm_wx_per_arr"] = atfm["dly_weather_min"] / arr
    atfm["atfm_cap_per_arr"] = atfm["dly_capacity_min"] / arr
    atfm["atfm_share_dly15"] = atfm["n_delayed_15"].fillna(0) / arr
    prev = atfm[["date", "airport", "atfm_dly_per_arr"]].copy()
    prev["date"] = (pd.to_datetime(prev["date"]) + pd.Timedelta(days=1)).dt.date
    prev = prev.rename(columns={"atfm_dly_per_arr": "atfm_dly_per_arr_prevday"})
    atfm = atfm.merge(prev, on=["date", "airport"], how="left")

    atx = pd.read_parquet(EXT / "pru_atxot_monthly.parquet")
    atx = atx.rename(columns={"APT_ICAO": "airport"})
    atx["atxot_month_add_min"] = atx["add_min_per_dep"]

    key = pd.DataFrame({
        "airport": df["AIRPORT"].astype(str).to_numpy(),
        "date": df["MVT_TIME_UTC_mvt"].dt.date.to_numpy(),
        "YEAR": df["MVT_TIME_UTC_mvt"].dt.year.to_numpy(),
        "MONTH_NUM": df["MVT_TIME_UTC_mvt"].dt.month.to_numpy(),
    }, index=df.index)
    out = key.merge(atfm[["date", "airport"] + ATFM_COLS], on=["date", "airport"], how="left")
    out = out.merge(atx[["YEAR", "MONTH_NUM", "airport", "atxot_month_add_min"]],
                    on=["YEAR", "MONTH_NUM", "airport"], how="left")
    out.index = df.index
    return out[ATFM_COLS + ATXOT_COLS].astype("float32")
