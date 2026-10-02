"""Airport-state / queue features, computable identically on train and ranking.

Only uses information available at prediction time in the ranking file:
- other flights' takeoff/landing times (MVT_TIME, given for all movements),
- other flights' AOBT_3 (given for ~98.5% of departures),
- arrivals' block times and taxi-in times (given),
- runway / stand of every movement.
Never uses departures' BLOCK_TIME or TAXITIME (blanked in ranking).
"""
import numpy as np
import pandas as pd

WINDOWS_PAST = [600, 1800, 3600]      # 10, 30, 60 min
WINDOWS_FUT = [600, 1800]             # 10, 30 min (transductive: takeoff times are known)

_EPOCH = pd.Timestamp(0, tz="UTC")


def _sec(ts: pd.Series) -> np.ndarray:
    """Epoch seconds as float, unit-safe (data ships as datetime64[us, UTC]). NaT -> NaN."""
    return ((ts - _EPOCH) / pd.Timedelta(seconds=1)).to_numpy(dtype=float)


def build_state_features(target: pd.DataFrame, movements: pd.DataFrame) -> pd.DataFrame:
    """target: departure rows to featurize. movements: all movements of the same period."""
    dep_all = movements[movements["PHASE_mvt"] == "DEP"]
    arr_all = movements[movements["PHASE_mvt"] == "ARR"]

    out = pd.DataFrame(index=target.index, dtype=float)
    cols = (
        ["q_dep_aobt_to_takeoff_airport", "q_dep_aobt_to_takeoff_runway", "q_backlog_at_takeoff"]
        + [f"deps_last_{w}s" for w in WINDOWS_PAST]
        + [f"arrs_last_{w}s" for w in WINDOWS_PAST]
        + [f"deps_next_{w}s" for w in WINDOWS_FUT]
        + [f"push_last_{w}s" for w in WINDOWS_PAST[:2]]
        + ["gap_prev_rwy", "gap_next_rwy", "gap_prev_airport",
           "arr_taxi_mean_3600s", "arr_taxi_n_3600s", "arr_taxiing_now"]
    )
    for c in cols:
        out[c] = np.nan

    for ap, tgt in target.groupby("AIRPORT", observed=True):
        ti = tgt.index
        t = _sec(tgt["MVT_TIME_UTC_mvt"])           # own takeoff time (known in test)
        b = _sec(tgt["AOBT_3_flt"])                 # own off-block proxy (known in test)
        b_ok = tgt["AOBT_3_flt"].notna().to_numpy()

        d = dep_all[dep_all["AIRPORT"] == ap]
        a = arr_all[arr_all["AIRPORT"] == ap]

        dep_t = np.sort(_sec(d["MVT_TIME_UTC_mvt"]))
        has_aobt = d["AOBT_3_flt"].notna()
        dep_aobt = np.sort(_sec(d.loc[has_aobt, "AOBT_3_flt"]))
        # takeoff times of the same subpopulation, so the backlog difference is unbiased
        dep_t_aobt = np.sort(_sec(d.loc[has_aobt, "MVT_TIME_UTC_mvt"]))
        arr_t_raw = _sec(a["MVT_TIME_UTC_mvt"])
        order = np.argsort(arr_t_raw)
        arr_t = arr_t_raw[order]
        arr_taxi = a["TAXITIME_SEC_mvt"].to_numpy(dtype=float)[order]
        arr_taxi_ok = ~np.isnan(arr_taxi)
        arr_taxi_cum = np.concatenate([[0.0], np.cumsum(np.where(arr_taxi_ok, arr_taxi, 0.0))])
        arr_n_cum = np.concatenate([[0], np.cumsum(arr_taxi_ok.astype(np.int64))])
        arr_inblock = np.sort(_sec(a.loc[a["BLOCK_TIME_UTC_mvt"].notna(), "BLOCK_TIME_UTC_mvt"]))

        ss = np.searchsorted

        # queue between own off-block and own takeoff (exclusive of self: [b, t) )
        n_before_t = ss(dep_t, t, side="left")
        q_air = n_before_t - ss(dep_t, b, side="left")
        out.loc[ti, "q_dep_aobt_to_takeoff_airport"] = np.where(b_ok, q_air, np.nan)

        # backlog: pushed back but not yet airborne at own takeoff time
        backlog = ss(dep_aobt, t, side="right") - ss(dep_t_aobt, t, side="right")
        out.loc[ti, "q_backlog_at_takeoff"] = backlog

        # past/future departure and arrival counts
        for w in WINDOWS_PAST:
            out.loc[ti, f"deps_last_{w}s"] = n_before_t - ss(dep_t, t - w, side="left")
            out.loc[ti, f"arrs_last_{w}s"] = ss(arr_t, t, side="left") - ss(arr_t, t - w, side="left")
        for w in WINDOWS_FUT:
            out.loc[ti, f"deps_next_{w}s"] = ss(dep_t, t + w, side="right") - ss(dep_t, t, side="right")
        for w in WINDOWS_PAST[:2]:
            out.loc[ti, f"push_last_{w}s"] = ss(dep_aobt, t, side="left") - ss(dep_aobt, t - w, side="left")

        # airport-level gap to previous takeoff
        idx_prev = ss(dep_t, t, side="left")
        prev_ok = idx_prev > 0
        gp = np.full(len(t), np.nan)
        gp[prev_ok] = t[prev_ok] - dep_t[idx_prev[prev_ok] - 1]
        out.loc[ti, "gap_prev_airport"] = gp

        # arrivals: recent taxi-in congestion + currently taxiing-in
        hi, lo = ss(arr_t, t, side="left"), ss(arr_t, t - 3600, side="left")
        n_arr = arr_n_cum[hi] - arr_n_cum[lo]
        s_arr = arr_taxi_cum[hi] - arr_taxi_cum[lo]
        out.loc[ti, "arr_taxi_n_3600s"] = n_arr
        with np.errstate(invalid="ignore"):
            out.loc[ti, "arr_taxi_mean_3600s"] = np.where(n_arr > 0, s_arr / np.maximum(n_arr, 1), np.nan)
        out.loc[ti, "arr_taxiing_now"] = ss(arr_t, t, side="right") - ss(arr_inblock, t, side="right")

        # runway-level: queue between off-block and takeoff + takeoff gaps on own runway
        rwy_tgt = tgt["RUNWAY_mvt"].astype(str)
        d_rwy = d["RUNWAY_mvt"].astype(str)
        for rwy, tgt_r in tgt.groupby(rwy_tgt, observed=True):
            ri = tgt_r.index
            pos = tgt.index.get_indexer(ri)
            tr, br, bokr = t[pos], b[pos], b_ok[pos]
            rt = np.sort(_sec(d.loc[(d_rwy == rwy).to_numpy(), "MVT_TIME_UTC_mvt"]))
            n_bt = ss(rt, tr, side="left")
            qr = n_bt - ss(rt, br, side="left")
            out.loc[ri, "q_dep_aobt_to_takeoff_runway"] = np.where(bokr, qr, np.nan)
            gpn = np.full(len(tr), np.nan)
            ok = n_bt > 0
            gpn[ok] = tr[ok] - rt[n_bt[ok] - 1]
            out.loc[ri, "gap_prev_rwy"] = gpn
            nxt = ss(rt, tr, side="right")
            gnn = np.full(len(tr), np.nan)
            ok2 = nxt < len(rt)
            gnn[ok2] = rt[nxt[ok2]] - tr[ok2]
            out.loc[ri, "gap_next_rwy"] = gnn

    # cap gaps (month boundaries in the ranking file create artificial huge gaps)
    for c in ["gap_prev_rwy", "gap_next_rwy", "gap_prev_airport"]:
        out[c] = out[c].clip(upper=7200)
    return out


def build_runway_config_features(target: pd.DataFrame, movements: pd.DataFrame) -> pd.DataFrame:
    """Runway-configuration state at takeoff time: how is the airport using its runways?"""
    dep_all = movements[movements["PHASE_mvt"] == "DEP"]
    arr_all = movements[movements["PHASE_mvt"] == "ARR"]
    W = 1800  # 30 min window

    out = pd.DataFrame(index=target.index, dtype=float)
    for c in ["rwy_dep_share_30m", "arrs_own_rwy_30m", "n_dep_rwys_30m", "dep_rank_on_rwy_today"]:
        out[c] = np.nan

    ss = np.searchsorted
    for ap, tgt in target.groupby("AIRPORT", observed=True):
        t = _sec(tgt["MVT_TIME_UTC_mvt"])
        d = dep_all[dep_all["AIRPORT"] == ap]
        a = arr_all[arr_all["AIRPORT"] == ap]
        dep_t_all = np.sort(_sec(d["MVT_TIME_UTC_mvt"]))
        deps_30 = ss(dep_t_all, t, side="left") - ss(dep_t_all, t - W, side="left")

        # per-runway sorted takeoff/landing times
        rwy_tgt = tgt["RUNWAY_mvt"].astype(str)
        d_rwy = d["RUNWAY_mvt"].astype(str)
        a_rwy = a["RUNWAY_mvt"].astype(str)
        rwy_dep_times = {r: np.sort(_sec(g["MVT_TIME_UTC_mvt"])) for r, g in d.groupby(d_rwy, observed=True)}
        rwy_arr_times = {r: np.sort(_sec(g["MVT_TIME_UTC_mvt"])) for r, g in a.groupby(a_rwy, observed=True)}

        # n distinct dep runways active in last 30 min: count runways with >=1 takeoff
        all_rwys = list(rwy_dep_times)
        act = np.zeros(len(t))
        for r in all_rwys:
            rt = rwy_dep_times[r]
            act += (ss(rt, t, side="left") - ss(rt, t - W, side="left")) > 0
        out.loc[tgt.index, "n_dep_rwys_30m"] = act

        for rwy, tgt_r in tgt.groupby(rwy_tgt, observed=True):
            pos = tgt.index.get_indexer(tgt_r.index)
            tr = t[pos]
            rt = rwy_dep_times.get(rwy, np.array([]))
            own = ss(rt, tr, side="left") - ss(rt, tr - W, side="left")
            tot = deps_30[pos]
            out.loc[tgt_r.index, "rwy_dep_share_30m"] = np.where(tot > 0, own / np.maximum(tot, 1), np.nan)
            at = rwy_arr_times.get(rwy, np.array([]))
            out.loc[tgt_r.index, "arrs_own_rwy_30m"] = ss(at, tr, side="left") - ss(at, tr - W, side="left")
            # rank of this takeoff on its runway since 04:00 local-ish (UTC day start ok)
            day_start = (tr // 86400) * 86400
            out.loc[tgt_r.index, "dep_rank_on_rwy_today"] = ss(rt, tr, side="left") - ss(rt, day_start, side="left")
    return out


def get_state_features(cache_name: str, target: pd.DataFrame, movements: pd.DataFrame) -> pd.DataFrame:
    """Parquet-cached wrapper (cache under data/processed/)."""
    from .data import PROCESSED
    path = PROCESSED / f"state_features_{cache_name}.parquet"
    if path.exists():
        cached = pd.read_parquet(path)
        if len(cached) == len(target):
            cached.index = target.index
            return cached
    sf = build_state_features(target, movements)
    sf.reset_index(drop=True).to_parquet(path, index=False)
    return sf
