"""Flight-specific context — columns and relations never exploited so far.

The AOBT_3 error is ~80% flight-specific, so this targets the individual flight's
own story rather than the airport state (which is saturated). Everything here is
known at prediction time in the ranking file.

1. Flight-plan revision history: how much the off-block estimate was revised
   (IOBT -> EOBT_1 -> LOBT -> AOBT_3). A heavily revised plan signals a disrupted
   operation, which is exactly when AOBT_3 drifts.
2. Unused columns: FLIGHT_TYPE_flt, ADES_FILED (diversion), ARVT_1/ARVT_3 (planned
   vs M3 flight duration), AIRCRAFT_TYPE_flt vs _mvt mismatch (record-matching
   quality), callsign prefix.
3. Within-period operator/airport punctuality, built only from timestamps that are
   given for departures in the ranking file (MVT_TIME and SCHED_TIME), never from
   the blanked taxi times.
"""
import numpy as np
import pandas as pd

COLS = ["rev_lobt_iobt", "rev_eobt_iobt", "rev_aobt_lobt", "rev_span",
        "dur_m3", "dur_filed", "dur_diff",
        "ades_filed_differs", "actype_mismatch", "fltid_missing",
        "op_day_late_mean", "op_day_late_n", "apt_hour_late_mean"]


def build_flight_context(df: pd.DataFrame) -> pd.DataFrame:
    X = pd.DataFrame(index=df.index)
    s = lambda a, b: (df[a] - df[b]).dt.total_seconds()

    # 1) flight-plan revision chain
    X["rev_lobt_iobt"] = s("LOBT_flt", "IOBT_flt")
    X["rev_eobt_iobt"] = s("EOBT_1_flt", "IOBT_flt")
    X["rev_aobt_lobt"] = s("AOBT_3_flt", "LOBT_flt")
    X["rev_span"] = (pd.concat([df["IOBT_flt"], df["EOBT_1_flt"], df["LOBT_flt"]], axis=1).max(axis=1)
                     - pd.concat([df["IOBT_flt"], df["EOBT_1_flt"], df["LOBT_flt"]], axis=1).min(axis=1)
                     ).dt.total_seconds()

    # 2) previously unused columns
    X["dur_m3"] = s("ARVT_3_flt", "AOBT_3_flt")
    X["dur_filed"] = s("ARVT_1_flt", "EOBT_1_flt")
    X["dur_diff"] = X["dur_m3"] - X["dur_filed"]
    X["ades_filed_differs"] = (df["ADES_FILED_flt"].astype(str)
                               != df["ADES_mvt"].astype(str)).astype("int8")
    X["actype_mismatch"] = (df["AIRCRAFT_TYPE_flt"].astype(str)
                            != df["AIRCRAFT_TYPE_mvt"].astype(str)).astype("int8")
    X["fltid_missing"] = df["FLIGHT_ID_mvt"].isna().astype("int8")

    # 3) punctuality context from timestamps that are given for ranking departures
    late = (df["MVT_TIME_UTC_mvt"] - df["SCHED_TIME_UTC_mvt"]).dt.total_seconds()
    late = late.where(late.abs() < 6 * 3600)
    key_op = (df["AIRPORT"].astype(str) + "_" + df["AIRCRAFT_OPERATOR_flt"].astype(str)
              + "_" + df["MVT_TIME_UTC_mvt"].dt.floor("D").astype(str))
    g = late.groupby(key_op)
    X["op_day_late_mean"] = g.transform("mean")
    X["op_day_late_n"] = g.transform("size")
    key_ah = (df["AIRPORT"].astype(str) + "_"
              + df["MVT_TIME_UTC_mvt"].dt.floor("h").astype(str))
    X["apt_hour_late_mean"] = late.groupby(key_ah).transform("mean")

    for c in ["FLIGHT_TYPE_flt"]:
        X[c] = df[c].astype("category")
    X["callsign_prefix"] = df["CALLSIGN_flt"].astype(str).str[:3].astype("category")

    for c in X.columns:
        if X[c].dtype == "float64":
            X[c] = X[c].astype("float32")
    return X
