"""Departure-side regime: what the OTHER departures are currently experiencing.

The earlier surface probe used arrival taxi-in, which is the wrong sensor: de-icing,
ATFM slot holding and runway queueing hit departures only. The right sensor is the
recent taxi-out of other departures — blanked in the ranking file, but reconstructible
from two columns that ARE given for every departure:

    proxy_taxi = MVT_TIME_UTC - AOBT_3_flt

The same proxy is used on train and on ranking, so the feature means the same thing
in both. Self is always excluded from its own neighbourhood.

  dep_proxy_mean_{w}      mean proxy taxi-out of other departures over the last w seconds
  dep_proxy_med_60m       median (robust to a single stuck flight)
  dep_proxy_anom_60m      same, minus the airport's usual level for that hour-of-day
  dep_proxy_rwy_30m       restricted to our own runway
  dep_long_share_60m      share of recent departures with proxy > 2x the airport norm
                          (the de-icing / flow-restriction signature)
  rwy_config_changed_1h   the mix of active departure runways changed in the last hour
"""
import numpy as np
import pandas as pd

_EPOCH = pd.Timestamp(0, tz="UTC")
WINDOWS = [1800, 3600, 10800]
COLS = ([f"dep_proxy_mean_{w}s" for w in WINDOWS]
        + ["dep_proxy_med_3600s", "dep_proxy_anom_3600s", "dep_proxy_n_3600s",
           "dep_proxy_rwy_1800s", "dep_long_share_3600s", "rwy_config_changed_1h"])


def _sec(ts: pd.Series) -> np.ndarray:
    return ((ts - _EPOCH) / pd.Timedelta(seconds=1)).to_numpy(dtype=float)


def _roll_mean_excl_self(t_ctx, v_ctx, t, w, self_val=None):
    """Mean of v_ctx over (t-w, t), excluding the row itself when it is in context."""
    cs = np.concatenate([[0.0], np.cumsum(v_ctx)])
    hi = np.searchsorted(t_ctx, t, "right")
    lo = np.searchsorted(t_ctx, t - w, "left")
    n = hi - lo
    s = cs[hi] - cs[lo]
    if self_val is not None:
        ok = np.isfinite(self_val)
        s = s - np.where(ok, np.nan_to_num(self_val), 0.0)
        n = n - ok.astype(int)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(n > 0, s / np.maximum(n, 1), np.nan), n


def build_dep_regime(target: pd.DataFrame, movements: pd.DataFrame) -> pd.DataFrame:
    dep = movements[movements["PHASE_mvt"] == "DEP"]
    proxy_ctx = (_sec(dep["MVT_TIME_UTC_mvt"]) - _sec(dep["AOBT_3_flt"]))
    ok_ctx = np.isfinite(proxy_ctx) & (proxy_ctx > 60) & (proxy_ctx < 14400)

    out = pd.DataFrame(index=target.index, columns=COLS, dtype=float)
    tgt_proxy_all = (_sec(target["MVT_TIME_UTC_mvt"]) - _sec(target["AOBT_3_flt"]))

    for ap, tgt in target.groupby("AIRPORT", observed=True):
        c = (dep["AIRPORT"] == ap).to_numpy() & ok_ctx
        if c.sum() < 500:
            continue
        tc = _sec(dep["MVT_TIME_UTC_mvt"])[c]
        vc = proxy_ctx[c]
        o = np.argsort(tc)
        tc, vc = tc[o], vc[o]
        rwy_c = dep["RUNWAY_mvt"].astype(str).to_numpy()[c][o]

        pos = target.index.get_indexer(tgt.index)
        t = _sec(tgt["MVT_TIME_UTC_mvt"])
        self_v = tgt_proxy_all[pos]
        self_v = np.where(np.isfinite(self_v) & (self_v > 60) & (self_v < 14400), self_v, np.nan)

        for w in WINDOWS:
            mean, n = _roll_mean_excl_self(tc, vc, t, w, self_v)
            out.loc[tgt.index, f"dep_proxy_mean_{w}s"] = mean
            if w == 3600:
                out.loc[tgt.index, "dep_proxy_n_3600s"] = n

        # robust median over the last hour (small windows -> loop over unique hours)
        hi = np.searchsorted(tc, t, "right")
        lo = np.searchsorted(tc, t - 3600, "left")
        med = np.full(len(t), np.nan)
        for j in range(len(t)):
            if hi[j] > lo[j]:
                med[j] = np.median(vc[lo[j]:hi[j]])
        out.loc[tgt.index, "dep_proxy_med_3600s"] = med

        # anomaly versus this airport's usual level at that hour of day
        hod = tgt["MVT_TIME_UTC_mvt"].dt.hour.to_numpy()
        hod_ctx = pd.to_datetime(tc, unit="s", utc=True).hour.to_numpy()
        norm = pd.Series(vc).groupby(hod_ctx).median()
        base = pd.Series(hod).map(norm).to_numpy(dtype=float)
        out.loc[tgt.index, "dep_proxy_anom_3600s"] = out.loc[tgt.index, "dep_proxy_mean_3600s"].to_numpy() - base

        # share of recent departures running at more than twice the norm
        long_flag = (vc > 2 * pd.Series(hod_ctx).map(norm).to_numpy(dtype=float)).astype(float)
        share, _ = _roll_mean_excl_self(tc, long_flag, t, 3600)
        out.loc[tgt.index, "dep_long_share_3600s"] = share

        # same-runway proxy over 30 min
        tgt_rwy = tgt["RUNWAY_mvt"].astype(str).to_numpy()
        for rwy in pd.unique(tgt_rwy):
            sr = tgt_rwy == rwy
            cr = rwy_c == rwy
            if cr.sum() < 50:
                continue
            m, _ = _roll_mean_excl_self(tc[cr], vc[cr], t[sr], 1800)
            out.loc[tgt.index[sr], "dep_proxy_rwy_1800s"] = m

        # runway-configuration change: is our runway among those used an hour ago?
        changed = np.zeros(len(t))
        for rwy in pd.unique(tgt_rwy):
            sr = tgt_rwy == rwy
            cr = rwy_c == rwy
            if cr.sum() == 0:
                continue
            tcr = tc[cr]
            used_before = (np.searchsorted(tcr, t[sr] - 1800, "left")
                           < np.searchsorted(tcr, t[sr] - 300, "right"))
            changed[sr] = (~used_before).astype(float)
        out.loc[tgt.index, "rwy_config_changed_1h"] = changed
    return out.astype("float32")
