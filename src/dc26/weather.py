"""METAR weather features. Joins the latest observation at/before takeoff time
(per airport), plus short-history aggregates. Stale obs (>3h) are masked."""
from pathlib import Path

import numpy as np
import pandas as pd

from .data import ROOT

METAR_PATH = ROOT / "data" / "external" / "metar.parquet"
_EPOCH = pd.Timestamp(0, tz="UTC")


def _sec(ts: pd.Series) -> np.ndarray:
    return ((ts - _EPOCH) / pd.Timedelta(seconds=1)).to_numpy(dtype=float)


def load_metar() -> pd.DataFrame:
    met = pd.read_parquet(METAR_PATH)
    met = met.sort_values(["station", "valid"]).reset_index(drop=True)
    met["tmpc"] = (met["tmpf"] - 32) / 1.8
    met["dwpc"] = (met["dwpf"] - 32) / 1.8
    met["spread"] = met["tmpc"] - met["dwpc"]
    met["wind_kt"] = met["sknt"]
    met["gust_kt"] = met["gust"]
    met["vsby_mi"] = met["vsby"]
    wx = met["wxcodes"].fillna("")
    met["wx_snow"] = wx.str.contains("SN").astype("int8")
    met["wx_freezing"] = wx.str.contains("FZ").astype("int8")
    met["wx_rain"] = (wx.str.contains("RA") | wx.str.contains("DZ")).astype("int8")
    met["wx_ts"] = wx.str.contains("TS").astype("int8")
    met["wx_fog"] = (wx.str.contains("FG") | wx.str.contains("BR")).astype("int8")
    met["ceiling_ft"] = np.where(met["skyc1"].isin(["BKN", "OVC", "VV"]), met["skyl1"], np.nan)
    met["deice_proxy"] = ((met["tmpc"] <= 3) &
                          ((met["wx_snow"] + met["wx_freezing"] + met["wx_rain"]) > 0)).astype("int8")
    # 3h rolling history per station (on the obs table, time-based)
    met = met.set_index("valid")
    g = met.groupby("station", group_keys=False)
    met["snow_3h"] = g["wx_snow"].apply(lambda s: s.rolling("3h").max())
    met["freezing_3h"] = g["wx_freezing"].apply(lambda s: s.rolling("3h").max())
    met["min_vsby_3h"] = g["vsby_mi"].apply(lambda s: s.rolling("3h").min())
    return met.reset_index()


WX_COLS = ["tmpc", "spread", "wind_kt", "gust_kt", "vsby_mi", "ceiling_ft",
           "wx_snow", "wx_freezing", "wx_rain", "wx_ts", "wx_fog", "deice_proxy",
           "snow_3h", "freezing_3h", "min_vsby_3h"]


def build_weather_features(df: pd.DataFrame, met: pd.DataFrame | None = None) -> pd.DataFrame:
    if met is None:
        met = load_metar()
    out = pd.DataFrame(index=df.index, columns=WX_COLS, dtype=float)
    for ap, tgt in df.groupby("AIRPORT", observed=True):
        m = met[met["station"] == ap]
        if m.empty:
            continue
        mt = _sec(m["valid"])
        t = _sec(tgt["MVT_TIME_UTC_mvt"])
        idx = np.searchsorted(mt, t, side="right") - 1
        ok = idx >= 0
        idx_c = np.clip(idx, 0, len(m) - 1)
        stale = np.zeros(len(t), dtype=bool)
        stale[ok] = (t[ok] - mt[idx_c[ok]]) > 3 * 3600
        vals = m[WX_COLS].to_numpy(dtype=float)[idx_c]
        vals[~ok | stale] = np.nan
        out.loc[tgt.index, WX_COLS] = vals
    return out.astype(float)
