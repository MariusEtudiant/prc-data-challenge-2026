"""Coarse entity groupings (Wu et al. 2026 recipe).

The model sees 1,883 raw stand categories, many with only a handful of flights, so
it can learn almost nothing for the rare ones. Wu et al. group stands and airlines
into 4-6 clusters by mean taxi-out instead. A coarse cluster label is far less
over-fittable than the continuous per-stand median that leaked in our earlier
target-encoding attempt (M5b, 405.6).

Two families:
  - structural, no target involved: pier/terminal prefix of the stand code
  - target-derived, FITTED ON TRAINING ROWS ONLY: quantile clusters of the entity's
    mean taxi-out, plus a training-frequency count (rarity)
"""
import numpy as np
import pandas as pd

ENTITIES = {
    "stand": ("AIRPORT", "STAND_mvt"),
    "op": ("AIRPORT", "AIRCRAFT_OPERATOR_flt"),
    "actype": ("AIRPORT", "AIRCRAFT_TYPE_mvt"),
}
NBINS = {"stand": 6, "op": 4, "actype": 4}
MIN_N = 30


def _key(df, cols):
    k = df[cols[0]].astype(str)
    for c in cols[1:]:
        k = k + "_" + df[c].astype(str)
    return k


def fit_entity_groups(df: pd.DataFrame, y: pd.Series, mask=None) -> dict:
    m = y.between(60, 7200).to_numpy()
    if mask is not None:
        m = m & np.asarray(mask)
    out = {}
    for name, cols in ENTITIES.items():
        k = _key(df, cols)
        g = y[m].groupby(k[m])
        mean, n = g.mean(), g.size()
        mean = mean[n >= MIN_N]
        if len(mean) < NBINS[name] * 2:
            out[name] = {"bins": {}, "count": n.to_dict()}
            continue
        q = pd.qcut(mean.rank(method="first"), NBINS[name], labels=False)
        out[name] = {"bins": dict(zip(mean.index, q.astype(int))), "count": n.to_dict()}
    return out


def build_entity_features(df: pd.DataFrame, groups: dict) -> pd.DataFrame:
    X = pd.DataFrame(index=df.index)
    # structural: pier / terminal prefix of the stand code (no target involved)
    st = df["STAND_mvt"].astype(str)
    X["stand_pier"] = (df["AIRPORT"].astype(str) + "_" + st.str[:1]).astype("category")
    X["stand_zone"] = (df["AIRPORT"].astype(str) + "_" + st.str[:2]).astype("category")
    X["stand_len"] = st.str.len().astype("float32")
    X["stand_is_numeric"] = st.str.isnumeric().astype("float32")

    for name, cols in ENTITIES.items():
        k = _key(df, cols)
        g = groups.get(name, {})
        X[f"{name}_cluster"] = k.map(g.get("bins", {})).astype("float32")
        X[f"{name}_train_n"] = k.map(g.get("count", {})).astype("float32")
    return X
