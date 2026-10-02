"""R02: look FORWARD at the arrivals, not only backward.

Every arrival-side feature we build uses a strictly backward window, from t-w to t.
That is the right reflex for forecasting - but this competition is transductive: the
ranking file contains every arrival of January and July 2026 WITH its taxi-in time,
including the ones that land after the departure we are predicting. Restricting
ourselves to the past throws away half of the observable regime.

It matters precisely where we are losing. At LIRF in July, three days carry 38.7% of
the month's error, and a day's share of that error correlates +0.663 with that same
day's arrival taxi-in p90. The regime is visible in the arrivals; we just refuse to
look at it.

Features added, all computed identically on train and ranking:
  arr_med_fwd_1h / 3h     median arrival taxi-in AFTER our takeoff
  arr_med_ctr_3h          centred window, the local regime
  arr_p90_ctr_6h          the tail of that regime, which is what hurts
  arr_med_day             the whole calendar day at that airport
  arr_p90_day             its tail
  arr_ratio_day           day median over the airport's own yearly median
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
PROC = ROOT / "data" / "processed"

_EPOCH = pd.Timestamp(0, tz="UTC")
COLS = ["arr_med_fwd_1h", "arr_med_fwd_3h", "arr_med_ctr_3h", "arr_p90_ctr_6h",
        "arr_med_day", "arr_p90_day", "arr_ratio_day"]


def _sec(s):
    return ((s - _EPOCH) / pd.Timedelta(seconds=1)).to_numpy(dtype=float)


def build(target, movements):
    arr = movements[movements["PHASE_mvt"] == "ARR"]
    out = pd.DataFrame(index=target.index, columns=COLS, dtype=float)
    for ap, tgt in target.groupby("AIRPORT", observed=True):
        a = arr[arr["AIRPORT"] == ap]
        ti = a["TAXITIME_SEC_mvt"].astype(float).to_numpy()
        at = _sec(a["MVT_TIME_UTC_mvt"])
        m = np.isfinite(ti) & (ti > 0) & (ti < 20000) & np.isfinite(at)
        at, ti = at[m], ti[m]
        if len(at) < 1000:
            continue
        o = np.argsort(at)
        at, ti = at[o], ti[o]
        t = _sec(tgt["MVT_TIME_UTC_mvt"])
        ss = np.searchsorted
        ref = np.median(ti)

        def stat(lo_t, hi_t, q):
            lo, hi = ss(at, lo_t, "left"), ss(at, hi_t, "right")
            v = np.full(len(t), np.nan)
            for i in range(len(t)):
                if hi[i] - lo[i] >= 5:
                    v[i] = np.quantile(ti[lo[i]:hi[i]], q)
            return v

        out.loc[tgt.index, "arr_med_fwd_1h"] = stat(t, t + 3600, 0.5)
        out.loc[tgt.index, "arr_med_fwd_3h"] = stat(t, t + 10800, 0.5)
        out.loc[tgt.index, "arr_med_ctr_3h"] = stat(t - 5400, t + 5400, 0.5)
        out.loc[tgt.index, "arr_p90_ctr_6h"] = stat(t - 10800, t + 10800, 0.9)
        d0 = np.floor(t / 86400) * 86400
        out.loc[tgt.index, "arr_med_day"] = stat(d0, d0 + 86400, 0.5)
        p90d = stat(d0, d0 + 86400, 0.9)
        out.loc[tgt.index, "arr_p90_day"] = p90d
        out.loc[tgt.index, "arr_ratio_day"] = out.loc[tgt.index, "arr_med_day"] / ref
    return out


if __name__ == "__main__":
    from dc26.data import load_movements_2025, load_ranking_2026
    for tag, loader, dst in (("2025", load_movements_2025, "arrfwd_train2025.parquet"),
                             ("2026", load_ranking_2026, "arrfwd_ranking2026.parquet")):
        mv = loader()
        dep = mv[mv["PHASE_mvt"] == "DEP"].copy()
        t0 = time.time()
        f = build(dep, mv).reset_index(drop=True)
        f.to_parquet(PROC / dst, index=False)
        print("%s: %d lignes, %.0f s | manquant %s" % (
            tag, len(f), time.time() - t0,
            " ".join("%s=%.1f%%" % (c, 100 * f[c].isna().mean()) for c in COLS)), flush=True)
