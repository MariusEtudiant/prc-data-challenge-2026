"""Submission v7: per-airport ensemble (global + specialised LGBM/CatBoost),
per-airport blend weights refitted on the ES window, all models retrained on 2025.

Stages: fit | blend  (fit is the long one; blend assembles and writes the file)
"""
import gc
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb
from catboost import CatBoostRegressor, Pool

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from dc26.mixture import REG_PARAMS
from dc26.validation import rmse

PROC = ROOT / "data" / "processed"
NPZ = ROOT / "experiments" / "v7_rank_preds.npz"
CAP = 86400
MIN_TRAIN = 20000
BOOST = 1.12


def _load():
    X = pd.read_parquet(PROC / "X_v2_full_t.parquet")
    Xr = pd.read_parquet(PROC / "X_v2_ranking_t.parquet")
    meta = pd.read_parquet(PROC / "meta_train2025.parquet")
    y = meta["y"].astype("float64")
    keep = (y > 0).to_numpy() & (y < 200000).to_numpy()
    return X, Xr, meta, y, keep


def _save(**kw):
    d = dict(np.load(NPZ, allow_pickle=True)) if NPZ.exists() else {}
    d.update(kw)
    np.savez(NPZ, **d)
    assert set(kw) <= set(np.load(NPZ, allow_pickle=True).keys()), "persist check failed"


def stage_fit():
    X, Xr, meta, y, keep = _load()
    cat = [c for c in X.columns if X[c].dtype.name == "category"]
    apt = X["AIRPORT"].astype(str).to_numpy()
    apt_r = Xr["AIRPORT"].astype(str).to_numpy()
    mir = np.load(ROOT / "experiments" / "m19_preds.npz", allow_pickle=True)
    g13 = np.load(ROOT / "experiments" / "m13_preds_t.npz")
    airports = [str(a) for a in mir["airports"]]

    # --- global models (rounds from the mirror run)
    n = int(g13["lgbm_iters"] * BOOST)
    ml = lgb.train(REG_PARAMS, lgb.Dataset(X[keep], y[keep].clip(upper=CAP), categorical_feature=cat),
                   num_boost_round=n)
    _save(glob_lgbm=ml.predict(Xr))
    ml.save_model(str(ROOT / "models" / "v7_glob_lgbm.txt"))
    print(f"[fit] global lgbm ({n})", flush=True)

    # --- per-airport LGBM
    it_air = dict(zip(airports, mir["air_lgbm_iters"]))
    p = np.full(len(Xr), np.nan)
    for a in airports:
        f = keep & (apt == a)
        r = apt_r == a
        if f.sum() < MIN_TRAIN or not r.any() or it_air.get(a, 0) == 0:
            continue
        n = int(it_air[a] * BOOST)
        m = lgb.train(REG_PARAMS, lgb.Dataset(X[f], y[f].clip(upper=CAP), categorical_feature=cat),
                      num_boost_round=n)
        p[r] = m.predict(Xr[r])
        m.save_model(str(ROOT / "models" / f"v7_air_lgbm_{a}.txt"))
        print(f"[fit] air lgbm {a} ({n})", flush=True)
    _save(air_lgbm=p)

    # --- CatBoost (string categories)
    for c in cat:
        X[c] = X[c].cat.add_categories("__NA__").fillna("__NA__")
        Xr[c] = Xr[c].cat.add_categories("__NA__").fillna("__NA__")
    gc.collect()

    n = int(g13["cb_iters"] * BOOST)
    mc = CatBoostRegressor(loss_function="RMSE", iterations=n, learning_rate=0.1, depth=7,
                           l2_leaf_reg=3.0, random_seed=42, verbose=False, thread_count=-1,
                           max_ctr_complexity=1, cat_features=cat)
    mc.fit(Pool(X[keep], y[keep].clip(upper=CAP), cat_features=cat))
    _save(glob_cb=mc.predict(Xr))
    mc.save_model(str(ROOT / "models" / "v7_glob_cb.cbm"))
    print(f"[fit] global catboost ({n})", flush=True)

    it_cb = dict(zip(airports, mir["air_cb_iters"]))
    p = np.full(len(Xr), np.nan)
    for a in airports:
        f = keep & (apt == a)
        r = apt_r == a
        if f.sum() < MIN_TRAIN or not r.any() or it_cb.get(a, 0) == 0:
            continue
        n = max(int(it_cb[a] * BOOST), 50)
        m = CatBoostRegressor(loss_function="RMSE", iterations=n, learning_rate=0.1, depth=7,
                              l2_leaf_reg=3.0, random_seed=42, verbose=False, thread_count=-1,
                              max_ctr_complexity=1, cat_features=cat)
        m.fit(Pool(X[f], y[f].clip(upper=CAP), cat_features=cat))
        p[r] = m.predict(Xr[r])
        m.save_model(str(ROOT / "models" / f"v7_air_cb_{a}.cbm"))
        print(f"[fit] air catboost {a} ({n})", flush=True)
    _save(air_cb=p)


def stage_blend():
    X, Xr, meta, y, keep = _load()
    apt_r = Xr["AIRPORT"].astype(str).to_numpy()
    d = np.load(NPZ, allow_pickle=True)
    # per-airport weights refitted on the mirror ES window (same recipe as M19)
    Xm = pd.read_parquet(PROC / "X_v2_mirror_t.parquet")
    tr, es = meta["tr"].to_numpy(), meta["es"].to_numpy()
    apt_m = Xm["AIRPORT"].astype(str).to_numpy()
    mir = np.load(ROOT / "experiments" / "m19_preds.npz", allow_pickle=True)
    g13 = np.load(ROOT / "experiments" / "m13_preds_t.npz")
    y_es = y[tr & es].to_numpy()
    P_es = np.vstack([g13["lgbm_es"], g13["cb_es"], mir["air_lgbm_es"], mir["air_cb_es"]])
    P_rk = np.vstack([d["glob_lgbm"], d["glob_cb"], d["air_lgbm"], d["air_cb"]])
    apt_es = apt_m[tr & es]

    grid = np.arange(0, 1.0001, 0.1)
    cands = [np.array([a, b, c, max(0.0, 1 - a - b - c)])
             for a in grid for b in grid for c in grid if a + b + c <= 1.0001]
    pred = np.full(len(Xr), np.nan)
    for a in np.unique(apt_r):
        me, mr = apt_es == a, apt_r == a
        if me.sum() < 100:
            continue
        Pe = P_es[:, me]
        avail = np.all(np.isfinite(Pe), axis=1) & np.all(np.isfinite(P_rk[:, mr]), axis=1)
        best = (None, 1e9)
        for w in cands:
            if w[~avail].sum() > 0:
                continue
            s = rmse(y_es[me], w @ Pe)
            if s < best[1]:
                best = (w, s)
        pred[mr] = best[0] @ P_rk[:, mr]
        print(f"  [{a}] w={np.round(best[0], 2)} ES={best[1]:.1f}", flush=True)

    fallback = np.isnan(pred)
    pred[fallback] = d["glob_lgbm"][fallback]
    print(f"fallback rows: {fallback.sum()}", flush=True)
    pred = np.clip(pred, 60, CAP)

    ids = pd.read_parquet(PROC / "rank_ids_t.parquet")["MVT_ID_mvt"]
    tpl = pd.read_parquet(ROOT / "data" / "submitting.parquet").drop(columns=["TAXITIME_SEC_mvt"])
    out = tpl.merge(pd.DataFrame({"MVT_ID_mvt": ids.values, "TAXITIME_SEC_mvt": pred}),
                    on="MVT_ID_mvt", how="left")
    assert out["TAXITIME_SEC_mvt"].notna().all() and len(out) == 215876
    out.to_parquet(ROOT / "submissions" / "reliable-hamburger_v7.parquet", index=False)
    print(f"v7 written: mean={out['TAXITIME_SEC_mvt'].mean():.0f}s "
          f"median={out['TAXITIME_SEC_mvt'].median():.0f}s "
          f"q99={out['TAXITIME_SEC_mvt'].quantile(.99):.0f}s", flush=True)


if __name__ == "__main__":
    {"fit": stage_fit, "blend": stage_blend}[sys.argv[1]]()
