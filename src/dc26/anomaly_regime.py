"""Ambient anomaly regime, measured on ARRIVALS - which are labelled in both sets.

Extreme taxi values are partly episodic: a recording problem at an airport during a
period produces many of them at once (LIRF Feb 2025: 21.6% of unmatched departures
extreme vs ~4% normally; EHAM Jan 2026: arrival anomaly rate 61x the 2025 level).

Departure taxi times are blanked in the ranking file, but ARRIVAL ones are not, so
the local rate of anomalous arrivals is observable in the test period itself and is
computed identically on train and ranking. Strictly backward+local windows.

Distinct from the earlier arrival probe (M17c, failed): that measured the *mean*
taxi-in anomaly (ordinary congestion); this measures the *rate of extreme values*
(the recording regime).
"""
import numpy as np
import pandas as pd

_EPOCH = pd.Timestamp(0, tz="UTC")
THRESHOLDS = [1800, 3600]
WINDOWS = [7 * 86400, 30 * 86400]
COLS = ([f"arr_anom_rate_{t}s_{w // 86400}d" for t in THRESHOLDS for w in WINDOWS]
        + ["arr_anom_rate_month", "arr_anom_n_30d"])


def _sec(ts: pd.Series) -> np.ndarray:
    return ((ts - _EPOCH) / pd.Timedelta(seconds=1)).to_numpy(dtype=float)


def build_anomaly_regime(target: pd.DataFrame, movements: pd.DataFrame) -> pd.DataFrame:
    arr = movements[movements["PHASE_mvt"] == "ARR"]
    out = pd.DataFrame(index=target.index, columns=COLS, dtype=float)

    for ap, tgt in target.groupby("AIRPORT", observed=True):
        a = arr[arr["AIRPORT"] == ap]
        if len(a) < 500:
            continue
        at = _sec(a["MVT_TIME_UTC_mvt"])
        taxi = a["TAXITIME_SEC_mvt"].to_numpy(dtype=float)
        o = np.argsort(at)
        at, taxi = at[o], taxi[o]
        t = _sec(tgt["MVT_TIME_UTC_mvt"])

        for thr in THRESHOLDS:
            flag = (taxi > thr).astype(float)
            cs = np.concatenate([[0.0], np.cumsum(flag)])
            for w in WINDOWS:
                hi = np.searchsorted(at, t, "left")
                lo = np.searchsorted(at, t - w, "left")
                n = hi - lo
                with np.errstate(invalid="ignore"):
                    out.loc[tgt.index, f"arr_anom_rate_{thr}s_{w // 86400}d"] = np.where(
                        n > 30, (cs[hi] - cs[lo]) / np.maximum(n, 1), np.nan)
                if thr == 3600 and w == 30 * 86400:
                    out.loc[tgt.index, "arr_anom_n_30d"] = n

        # whole-month rate (uses the full month: legal, arrivals are given in both sets)
        m_arr = pd.to_datetime(a["MVT_TIME_UTC_mvt"].to_numpy()[o]).to_period("M")
        rate_m = pd.Series((taxi > 3600).astype(float)).groupby(m_arr).mean()
        out.loc[tgt.index, "arr_anom_rate_month"] = (
            tgt["MVT_TIME_UTC_mvt"].dt.to_period("M").map(rate_m).to_numpy(dtype=float))
    return out.astype("float32")
