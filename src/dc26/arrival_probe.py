"""Surface-condition probe built from ARRIVALS.

The ranking file keeps arrival taxi-in times unmasked: they are 215k *measurements*
of how the surface actually behaved in Jan/Jul 2026. Raw means are confounded by
which runway/stand each arrival used, so we compare every arrival with its own
reference taxi-in (P50 per airport x runway x stand, fitted on 2025 arrivals) and
aggregate the resulting anomaly in rolling windows.

  arr_anom_30m / _60m / _180m : mean taxi-in anomaly recently observed (seconds)
  arr_anom_n_60m              : how many arrivals back that estimate
  arr_anom_day                : same-day airport-level anomaly
"""
import numpy as np
import pandas as pd

_EPOCH = pd.Timestamp(0, tz="UTC")
WINDOWS = [1800, 3600, 10800]
COLS = [f"arr_anom_{w}s" for w in WINDOWS] + ["arr_anom_n_3600s", "arr_anom_day"]
MIN_SAMPLE = 30


def _sec(ts: pd.Series) -> np.ndarray:
    return ((ts - _EPOCH) / pd.Timedelta(seconds=1)).to_numpy(dtype=float)


def _keys(df: pd.DataFrame) -> pd.Series:
    return (df["AIRPORT"].astype(str) + "_" + df["RUNWAY_mvt"].astype(str)
            + "_" + df["STAND_mvt"].astype(str))


def fit_arrival_reference(movements: pd.DataFrame) -> dict:
    """Median taxi-in per airport x runway x stand, from arrivals (2025 scope)."""
    a = movements[movements["PHASE_mvt"] == "ARR"]
    t = a["TAXITIME_SEC_mvt"]
    m = t.between(30, 3600)
    k = _keys(a)[m]
    g = t[m].groupby(k)
    ref = g.median()
    ref = ref[g.size() >= MIN_SAMPLE]
    ap = t[m].groupby(a["AIRPORT"].astype(str)[m]).median()
    return {"combo": ref, "apt": ap}


def build_arrival_probe(target: pd.DataFrame, movements: pd.DataFrame,
                        arr_refs: dict) -> pd.DataFrame:
    a = movements[movements["PHASE_mvt"] == "ARR"].copy()
    k = _keys(a)
    ref = k.map(arr_refs["combo"]).fillna(a["AIRPORT"].astype(str).map(arr_refs["apt"]))
    anom = (a["TAXITIME_SEC_mvt"] - ref).to_numpy(dtype=float)
    valid = np.isfinite(anom) & a["TAXITIME_SEC_mvt"].between(30, 3600).to_numpy()

    out = pd.DataFrame(index=target.index, columns=COLS, dtype=float)
    for ap, tgt in target.groupby("AIRPORT", observed=True):
        sel = (a["AIRPORT"] == ap).to_numpy() & valid
        if sel.sum() < MIN_SAMPLE:
            continue
        at = _sec(a["MVT_TIME_UTC_mvt"])[sel]
        av = anom[sel]
        o = np.argsort(at)
        at, av = at[o], av[o]
        cs = np.concatenate([[0.0], np.cumsum(av)])
        t = _sec(tgt["MVT_TIME_UTC_mvt"])
        hi = np.searchsorted(at, t, "left")
        for w in WINDOWS:
            lo = np.searchsorted(at, t - w, "left")
            n = hi - lo
            with np.errstate(invalid="ignore", divide="ignore"):
                mean = np.where(n > 0, (cs[hi] - cs[lo]) / np.maximum(n, 1), np.nan)
            out.loc[tgt.index, f"arr_anom_{w}s"] = mean
            if w == 3600:
                out.loc[tgt.index, "arr_anom_n_3600s"] = n
        # same-day airport anomaly (uses the whole day: legal, arrivals are given)
        day0 = (t // 86400) * 86400
        lo_d = np.searchsorted(at, day0, "left")
        hi_d = np.searchsorted(at, day0 + 86400, "left")
        nd = hi_d - lo_d
        out.loc[tgt.index, "arr_anom_day"] = np.where(nd > 0, (cs[hi_d] - cs[lo_d]) / np.maximum(nd, 1), np.nan)
    return out.astype("float32")
