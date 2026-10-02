"""Stand-chaining / turnaround features.

Each departure is linked to the most recent arrival that blocked in at the SAME
stand before it. That inbound aircraft is (almost always) the one departing, which
gives, without any external data:
  - available turnaround time (scheduled departure - inbound in-block),
  - inbound lateness (a late inbound compresses the turnaround),
  - the inbound's measured taxi-in time at that stand (a direct, same-location
    probe of current surface conditions),
  - stand occupancy since the inbound arrived.

All inputs are known at prediction time in the ranking file: arrivals keep their
BLOCK_TIME and TAXITIME there; only departures are blanked.
"""
import numpy as np
import pandas as pd

_EPOCH = pd.Timestamp(0, tz="UTC")
COLS = ["turnaround_avail_s", "inbound_taxiin_s", "inbound_late_s",
        "taxi_upper_bound_s", "inbound_gap_ok"]


def _sec(ts: pd.Series) -> np.ndarray:
    return ((ts - _EPOCH) / pd.Timedelta(seconds=1)).to_numpy(dtype=float)


def build_turnaround_features(target: pd.DataFrame, movements: pd.DataFrame) -> pd.DataFrame:
    arr = movements[movements["PHASE_mvt"] == "ARR"]
    out = pd.DataFrame(index=target.index, columns=COLS, dtype=float)

    for (ap, stand), tgt in target.groupby(["AIRPORT", "STAND_mvt"], observed=True):
        a = arr[(arr["AIRPORT"] == ap) & (arr["STAND_mvt"] == stand)]
        if a.empty:
            continue
        a = a.sort_values("BLOCK_TIME_UTC_mvt")
        a_block = _sec(a["BLOCK_TIME_UTC_mvt"])
        a_taxi = a["TAXITIME_SEC_mvt"].to_numpy(dtype=float)
        a_sched = _sec(a["SCHED_TIME_UTC_mvt"])
        a_mvt = _sec(a["MVT_TIME_UTC_mvt"])

        # reference instant = own scheduled off-block (always known), fall back to takeoff
        t_ref = _sec(tgt["SCHED_TIME_UTC_mvt"])
        t_off = _sec(tgt["MVT_TIME_UTC_mvt"])
        t_ref = np.where(np.isnan(t_ref), t_off, t_ref)

        idx = np.searchsorted(a_block, t_ref, side="right") - 1
        ok = idx >= 0
        i = np.clip(idx, 0, len(a) - 1)

        occ = np.where(ok, t_ref - a_block[i], np.nan)
        # a plausible turnaround is under ~12h; beyond that the chain is broken
        plausible = ok & (occ > 0) & (occ < 12 * 3600)
        out.loc[tgt.index, "turnaround_avail_s"] = np.where(plausible, occ, np.nan)
        # hard physical bound: taxi-out cannot exceed the time since the inbound
        # blocked in at this very stand (both timestamps known at prediction time)
        out.loc[tgt.index, "taxi_upper_bound_s"] = np.where(plausible, t_off - a_block[i], np.nan)
        out.loc[tgt.index, "inbound_taxiin_s"] = np.where(plausible, a_taxi[i], np.nan)
        out.loc[tgt.index, "inbound_late_s"] = np.where(plausible, a_mvt[i] - a_sched[i], np.nan)
        out.loc[tgt.index, "inbound_gap_ok"] = plausible.astype(float)
    return out.astype("float32")
