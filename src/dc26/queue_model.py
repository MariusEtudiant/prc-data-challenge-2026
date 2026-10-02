"""Physical departure-queue reconstruction.

With the PRU unimpeded reference we can estimate when each aircraft *reaches the
runway queue*:   t_join = AOBT_3 + ref_txot .
Everything else (takeoff times of all departures) is known, so the queue can be
rebuilt exactly rather than approximated by time-window counts:

  q_ahead_at_join   aircraft that joined before us and had not taken off yet
  rwy_gap_median    current runway service interval (median inter-departure gap)
  pred_queue_wait   q_ahead_at_join * rwy_gap_median  -> physical wait estimate
  pred_taxi_total   ref_txot + pred_queue_wait        -> physical taxi estimate
  seq_delta         takeoff order minus join order within the day (overtaking)
"""
import numpy as np
import pandas as pd

_EPOCH = pd.Timestamp(0, tz="UTC")
COLS = ["t_join_valid", "q_ahead_at_join", "rwy_gap_median", "pred_queue_wait",
        "pred_taxi_total", "seq_delta", "join_to_takeoff"]


def _sec(ts: pd.Series) -> np.ndarray:
    return ((ts - _EPOCH) / pd.Timedelta(seconds=1)).to_numpy(dtype=float)


def build_queue_features(target: pd.DataFrame, movements: pd.DataFrame,
                         ref_txot: pd.Series) -> pd.DataFrame:
    dep_all = movements[movements["PHASE_mvt"] == "DEP"]
    out = pd.DataFrame(index=target.index, columns=COLS, dtype=float)

    # queue-join estimate for every departure of the period (needed as context)
    ctx_ref = ref_txot.reindex(dep_all.index)
    ctx_join = _sec(dep_all["AOBT_3_flt"]) + ctx_ref.to_numpy(dtype=float)
    ctx_off = _sec(dep_all["MVT_TIME_UTC_mvt"])
    ctx_rwy = (dep_all["AIRPORT"].astype(str) + "_" + dep_all["RUNWAY_mvt"].astype(str)).to_numpy()

    tgt_rwy_all = (target["AIRPORT"].astype(str) + "_" + target["RUNWAY_mvt"].astype(str))
    t_off_all = _sec(target["MVT_TIME_UTC_mvt"])
    t_join_all = _sec(target["AOBT_3_flt"]) + ref_txot.reindex(target.index).to_numpy(dtype=float)

    for rwy in pd.unique(tgt_rwy_all):
        sel = (tgt_rwy_all == rwy).to_numpy()
        idx = target.index[sel]
        t_off = t_off_all[sel]
        t_join = t_join_all[sel]

        c = ctx_rwy == rwy
        cj = np.sort(ctx_join[c][~np.isnan(ctx_join[c])])
        co = np.sort(ctx_off[c])

        # backlog at our own join instant: joined before us, not yet airborne
        tj = np.where(np.isnan(t_join), t_off, t_join)
        q = np.searchsorted(cj, tj, "left") - np.searchsorted(co, tj, "right")
        out.loc[idx, "q_ahead_at_join"] = np.maximum(q, 0)
        out.loc[idx, "t_join_valid"] = (~np.isnan(t_join)).astype(float)
        out.loc[idx, "join_to_takeoff"] = t_off - tj

        # runway service interval: median gap over the 10 departures before us
        pos = np.searchsorted(co, t_off, "left")
        gaps = np.diff(co, prepend=co[0] if len(co) else 0.0)
        k = 10
        lo = np.maximum(pos - k, 1)
        med = np.full(len(t_off), np.nan)
        for j in range(len(t_off)):
            if pos[j] > lo[j]:
                med[j] = np.median(gaps[lo[j]:pos[j]])
        med = np.clip(med, 20, 1800)
        out.loc[idx, "rwy_gap_median"] = med
        wait = np.maximum(q, 0) * med
        out.loc[idx, "pred_queue_wait"] = wait
        out.loc[idx, "pred_taxi_total"] = ref_txot.reindex(idx).to_numpy(dtype=float) + wait

    # overtaking: takeoff order vs join order, within runway and UTC day
    day = target["MVT_TIME_UTC_mvt"].dt.floor("D")
    grp = pd.DataFrame({"rwy": tgt_rwy_all.to_numpy(), "day": day.to_numpy(),
                        "join": t_join_all, "off": t_off_all}, index=target.index)
    g = grp.groupby(["rwy", "day"], observed=True)
    out["seq_delta"] = (g["off"].rank(method="first") - g["join"].rank(method="first")).to_numpy()
    return out.astype("float32")
