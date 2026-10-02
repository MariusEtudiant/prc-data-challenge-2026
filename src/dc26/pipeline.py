"""Unified feature pipeline (v2) — identical code path for train and ranking.

v2 = v1 static + airport state/queue + METAR + PRU unimpeded reference (P10 per
stand x runway, official ATXOT methodology) + PRU daily ATFM / monthly ATXOT.

The reference times are FITTED on a training sample (`fit_reference_times`) and
then APPLIED to any set of rows, so mirror experiments and the final submission
use the same code with different fitting scopes.
"""
import numpy as np
import pandas as pd

from .features import build_features_v1
from .state_features import get_state_features, build_state_features
from .weather import build_weather_features
from .pru_features import build_pru_features

MIN_SAMPLE = 100          # min flights per group for a valid reference (PRU: >=10 below P10)
REF_Q = 0.10              # PRU unimpeded reference = 10th percentile
MAX_TAXI_REF = 7200       # PRU filter: taxi-out > 120 min excluded from the reference


def _keys(df: pd.DataFrame):
    ap = df["AIRPORT"].astype(str)
    combo = ap + "_" + df["STAND_mvt"].astype(str) + "_" + df["RUNWAY_mvt"].astype(str)
    rwy = ap + "_" + df["RUNWAY_mvt"].astype(str)
    return combo, rwy, ap


def fit_reference_times(df: pd.DataFrame, y: pd.Series, sample_mask=None) -> dict:
    """PRU-style unimpeded reference taxi-out times, fitted on `df[sample_mask]`."""
    combo, rwy, ap = _keys(df)
    m = y.between(60, MAX_TAXI_REF).to_numpy()
    if sample_mask is not None:
        m = m & np.asarray(sample_mask)

    def q10(keys):
        g = y[m].groupby(keys[m])
        ref = g.quantile(REF_Q)
        return ref[g.size() >= MIN_SAMPLE]

    return {"combo": q10(combo), "rwy": q10(rwy), "apt": q10(ap)}


def apply_reference_times(df: pd.DataFrame, refs: dict) -> pd.Series:
    combo, rwy, ap = _keys(df)
    ref = combo.map(refs["combo"])
    ref = ref.fillna(rwy.map(refs["rwy"]))
    ref = ref.fillna(ap.map(refs["apt"]))
    return ref.astype("float32")


def build_features_v2(df: pd.DataFrame, movements: pd.DataFrame, refs: dict,
                      met=None, state_cache: str | None = None,
                      with_turnaround: bool = False) -> pd.DataFrame:
    """Full v2 matrix for the departure rows `df` within period `movements`."""
    state = (get_state_features(state_cache, df, movements) if state_cache
             else build_state_features(df, movements))
    blocks = [build_features_v1(df), state, build_weather_features(df, met),
              build_pru_features(df)]
    if with_turnaround:
        from .turnaround import build_turnaround_features
        blocks.append(build_turnaround_features(df, movements))
    X = pd.concat(blocks, axis=1)
    ref = apply_reference_times(df, refs)
    X["ref_txot_p10"] = ref.to_numpy()
    X["implied_add"] = (X["d_aobt3"].to_numpy() - ref.to_numpy()).astype("float32")
    for c in X.columns:
        if X[c].dtype == "float64":
            X[c] = X[c].astype("float32")
    return X


def align_categories(X_target: pd.DataFrame, X_ref: pd.DataFrame) -> pd.DataFrame:
    """Make categorical columns of X_target share X_ref's categories (LightGBM needs this)."""
    out = X_target.copy()
    for c in X_ref.columns:
        if X_ref[c].dtype.name == "category":
            out[c] = pd.Categorical(out[c], categories=X_ref[c].cat.categories)
    return out[X_ref.columns]
