"""Per-entity history features — the flight's own story rather than the airport's state.

The AOBT_3 error is 80% flight-specific (see journal EDA), so the remaining signal
should live in what happened to *this* aircraft / *this* operator / *this* flight
number recently, not in ambient traffic.

All features are strictly backward-looking (expanding means shifted by one) so no
future information leaks, and they are computed identically on train and ranking.

  op_recent_dev      : operator's recent mean deviation from its own taxi baseline
  fltnum_prev_taxi   : same flight number's previous observed taxi-out at this airport
  fltnum_mean_taxi   : expanding mean for that flight number
  stand_prev_taxi    : previous departure from the same stand
  op_day_delay       : operator's mean departure lateness so far that day
  ac_type_dev        : aircraft type's recent deviation
"""
import numpy as np
import pandas as pd

COLS = ["op_recent_dev", "fltnum_prev_taxi", "fltnum_mean_taxi",
        "stand_prev_taxi", "op_day_delay", "ac_type_dev"]


def _expanding_prev_mean(df: pd.DataFrame, by, value: str) -> pd.Series:
    """Mean of `value` over all previous rows of the group (no leakage)."""
    g = df.groupby(by, observed=True)[value]
    cs = g.cumsum() - df[value].fillna(0)
    cnt = g.cumcount()
    return (cs / cnt.replace(0, np.nan)).astype("float32")


def build_history_features(target: pd.DataFrame, movements: pd.DataFrame) -> pd.DataFrame:
    """`target` must be departures; `movements` supplies the observed taxi times.

    On the ranking file departure taxi times are blank, so the history is carried by
    whatever IS observed there (arrivals) plus the 2025 priors passed in `movements`.
    """
    dep = movements[movements["PHASE_mvt"] == "DEP"].copy()
    dep = dep.sort_values("MVT_TIME_UTC_mvt")
    dep["taxi"] = dep["TAXITIME_SEC_mvt"].where(dep["TAXITIME_SEC_mvt"].between(60, 7200))

    apt = dep["AIRPORT"].astype(str)
    dep["_op"] = apt + "_" + dep["AIRCRAFT_OPERATOR_flt"].astype(str)
    dep["_fn"] = apt + "_" + dep["FLIGHT_mvt"].astype(str)
    dep["_st"] = apt + "_" + dep["STAND_mvt"].astype(str)
    dep["_ac"] = apt + "_" + dep["AIRCRAFT_TYPE_mvt"].astype(str)
    dep["_day"] = dep["MVT_TIME_UTC_mvt"].dt.floor("D")

    # deviation from the flight's own stand x runway reference is unavailable here,
    # so use the airport-hour baseline as the comparison point
    base = dep.groupby([apt, dep["MVT_TIME_UTC_mvt"].dt.hour], observed=True)["taxi"].transform("median")
    dep["dev"] = dep["taxi"] - base

    out = pd.DataFrame(index=dep.index)
    out["op_recent_dev"] = _expanding_prev_mean(dep, "_op", "dev")
    out["ac_type_dev"] = _expanding_prev_mean(dep, "_ac", "dev")
    out["fltnum_mean_taxi"] = _expanding_prev_mean(dep, "_fn", "taxi")
    out["fltnum_prev_taxi"] = dep.groupby("_fn", observed=True)["taxi"].shift(1).astype("float32")
    out["stand_prev_taxi"] = dep.groupby("_st", observed=True)["taxi"].shift(1).astype("float32")

    late = (dep["MVT_TIME_UTC_mvt"] - dep["SCHED_TIME_UTC_mvt"]).dt.total_seconds()
    dep["late"] = late.where(late.abs() < 6 * 3600)
    out["op_day_delay"] = _expanding_prev_mean(dep, ["_op", "_day"], "late")

    return out.reindex(target.index)[COLS].astype("float32")
