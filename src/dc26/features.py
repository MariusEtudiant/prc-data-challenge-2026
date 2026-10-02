"""Static (per-row) feature builder — v1 = temporal + FPL deltas + categoricals + has_fpl."""
import pandas as pd


def build_features_v1(df: pd.DataFrame) -> pd.DataFrame:
    X = pd.DataFrame(index=df.index)
    t = df["MVT_TIME_UTC_mvt"]
    X["hour"] = t.dt.hour + t.dt.minute / 60
    X["dow"] = t.dt.dayofweek.astype("int8")
    X["month"] = t.dt.month.astype("int8")
    X["doy"] = t.dt.dayofyear.astype("int16")
    for c, ts in [("aobt3", "AOBT_3_flt"), ("eobt1", "EOBT_1_flt"),
                  ("lobt", "LOBT_flt"), ("iobt", "IOBT_flt"), ("sched", "SCHED_TIME_UTC_mvt")]:
        X[f"d_{c}"] = (t - df[ts]).dt.total_seconds()
    X["d_aobt3_vs_eobt1"] = (df["AOBT_3_flt"] - df["EOBT_1_flt"]).dt.total_seconds()
    X["has_fpl"] = df["FLIGHT_ID_mvt"].notna().astype("int8")
    for c in ["AIRPORT", "AIRCRAFT_TYPE_mvt", "WK_TBL_CAT_flt", "MARKET_SEGMENT_flt",
              "AIRCRAFT_OPERATOR_flt", "FLIGHT_RULE_mvt"]:
        X[c] = df[c].astype("category")
    X["RUNWAY"] = (df["AIRPORT"].astype(str) + "_" + df["RUNWAY_mvt"].astype(str)).astype("category")
    X["STAND"] = (df["AIRPORT"].astype(str) + "_" + df["STAND_mvt"].astype(str)).astype("category")
    X["ADES_REGION"] = df["ADES_mvt"].astype(str).str[:2].astype("category")
    return X
