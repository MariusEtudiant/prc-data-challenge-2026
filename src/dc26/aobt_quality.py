"""Detect a reconstructed (non-measured) AOBT_3.

EUROCONTROL's ADRR documentation states the NM off-block time "may be calculated
from the known take-off time minus a standard taxi time value for the airport"
when no flight-data update was received. Such values carry no information about
the real off-block, and they show up as sharp spikes in the distribution of
MVT_TIME - AOBT_3 (one bucket per standard constant, smeared over a minute by the
minute-rounding of AOBT_3).

Spikes are learned from the training year and applied unchanged to the ranking set.
"""
import numpy as np
import pandas as pd

COLS = ["aobt_spike_flag", "aobt_dev_from_spike", "aobt_bucket_share"]


def _dmin(df):
    d = (df["MVT_TIME_UTC_mvt"] - df["AOBT_3_flt"]).dt.total_seconds()
    return d, np.floor(d / 60)


def fit_aobt_spikes(df: pd.DataFrame, thresh: float = 2.5, floor: float = 0.03) -> dict:
    d, dmin = _dmin(df)
    ok = d.between(0, 7200)
    out = {}
    for a, idx in df[ok].groupby("AIRPORT", observed=True).groups.items():
        h = dmin.loc[idx].value_counts(normalize=True).sort_index()
        sp = {}
        for m, v in h.items():
            nb = np.mean([h.get(m + k, 0.0) for k in (-3, -2, 2, 3)])
            if v > thresh * max(nb, 1e-9) and v > floor:
                sp[float(m)] = float(v)
        out[str(a)] = {"spikes": sp, "hist": {float(k): float(v) for k, v in h.items()}}
    return out


def build_aobt_quality(df: pd.DataFrame, model: dict) -> pd.DataFrame:
    d, dmin = _dmin(df)
    X = pd.DataFrame(index=df.index, columns=COLS, dtype=float)
    for a, idx in df.groupby("AIRPORT", observed=True).groups.items():
        m = model.get(str(a))
        if not m:
            continue
        sp = np.array(sorted(m["spikes"])) if m["spikes"] else np.array([])
        dm = dmin.loc[idx].to_numpy(dtype=float)
        X.loc[idx, "aobt_bucket_share"] = pd.Series(dm).map(m["hist"]).to_numpy()
        if len(sp) == 0:
            X.loc[idx, "aobt_spike_flag"] = 0.0
            continue
        X.loc[idx, "aobt_spike_flag"] = np.isin(dm, sp).astype(float)
        near = sp[np.argmin(np.abs(dm[:, None] - sp[None, :]), axis=1)]
        X.loc[idx, "aobt_dev_from_spike"] = d.loc[idx].to_numpy() - near * 60.0
    return X.astype("float32")
