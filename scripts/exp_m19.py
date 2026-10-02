"""M19 (staged): per-airport ensemble.

Four predictors per row — global LGBM, global CatBoost (from the M13_t run),
per-airport LGBM, per-airport CatBoost — blended with weights chosen INDEPENDENTLY
FOR EACH AIRPORT on the early-stopping window. Airports where the specialised
models do not help (e.g. LSZH) naturally fall back to the global ones.

stages: air_lgbm | air_cb | blend
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
PROC = ROOT / "data" / "processed"
NPZ = ROOT / "experiments" / "m19_preds.npz"
CAP = 86400
MIN_TRAIN = 20000


def _load():
    X = pd.read_parquet(PROC / "X_v2_mirror_t.parquet")
    meta = pd.read_parquet(PROC / "meta_train2025.parquet")
    y = meta["y"].astype("float64")
    tr, va, es = meta["tr"].to_numpy(), meta["va"].to_numpy(), meta["es"].to_numpy()
    keep = tr & (y > 0).to_numpy() & (y < 200000).to_numpy()
    return X, meta, y, tr, va, es, keep


def _save(**kw):
    d = dict(np.load(NPZ, allow_pickle=True)) if NPZ.exists() else {}
    d.update(kw)
    np.savez(NPZ, **d)


def stage_air_lgbm():
    import lightgbm as lgb
    from dc26.mixture import REG_PARAMS
    X, meta, y, tr, va, es, keep = _load()
    cat = [c for c in X.columns if X[c].dtype.name == "category"]
    apt = X["AIRPORT"].astype(str).to_numpy()
    es_rows = tr & es
    p_va = np.full(va.sum(), np.nan)
    p_es = np.full(es_rows.sum(), np.nan)
    iters = {}
    for a in np.unique(apt):
        fit = keep & (apt == a) & ~es
        if fit.sum() < MIN_TRAIN:
            continue
        m = lgb.train(REG_PARAMS, lgb.Dataset(X[fit], y[fit].clip(upper=CAP), categorical_feature=cat),
                      num_boost_round=6000,
                      valid_sets=[lgb.Dataset(X[keep & (apt == a) & es],
                                              y[keep & (apt == a) & es].clip(upper=CAP))],
                      callbacks=[lgb.early_stopping(150, verbose=False)])
        iters[a] = m.best_iteration
        sv = apt[va] == a
        if sv.any():
            p_va[sv] = m.predict(X[va][sv], num_iteration=m.best_iteration)
        se = apt[es_rows] == a
        if se.any():
            p_es[se] = m.predict(X[es_rows][se], num_iteration=m.best_iteration)
        print(f"[air_lgbm] {a}: iters={m.best_iteration}", flush=True)
    _save(air_lgbm_va=p_va, air_lgbm_es=p_es,
          air_lgbm_iters=np.array([iters.get(a, 0) for a in np.unique(apt)]),
          airports=np.array(sorted(set(apt)), dtype="U8"))


def stage_air_cb():
    from catboost import CatBoostRegressor, Pool
    X, meta, y, tr, va, es, keep = _load()
    cat = [c for c in X.columns if X[c].dtype.name == "category"]
    for c in cat:
        X[c] = X[c].cat.add_categories("__NA__").fillna("__NA__")
    apt = X["AIRPORT"].astype(str).to_numpy()
    es_rows = tr & es
    p_va = np.full(va.sum(), np.nan)
    p_es = np.full(es_rows.sum(), np.nan)
    iters = {}
    for a in np.unique(apt):
        fit = keep & (apt == a) & ~es
        ev = keep & (apt == a) & es
        if fit.sum() < MIN_TRAIN:
            continue
        m = CatBoostRegressor(loss_function="RMSE", iterations=1500, learning_rate=0.1,
                              depth=7, l2_leaf_reg=3.0, random_seed=42, od_type="Iter",
                              od_wait=80, verbose=False, thread_count=-1,
                              max_ctr_complexity=1, cat_features=cat)
        m.fit(Pool(X[fit], y[fit].clip(upper=CAP), cat_features=cat),
              eval_set=Pool(X[ev], y[ev].clip(upper=CAP), cat_features=cat))
        iters[a] = m.get_best_iteration()
        sv = apt[va] == a
        if sv.any():
            p_va[sv] = m.predict(X[va][sv])
        se = apt[es_rows] == a
        if se.any():
            p_es[se] = m.predict(X[es_rows][se])
        print(f"[air_cb] {a}: iters={m.get_best_iteration()}", flush=True)
    _save(air_cb_va=p_va, air_cb_es=p_es,
          air_cb_iters=np.array([iters.get(a, 0) for a in np.unique(apt)]))


def stage_blend():
    from dc26.validation import rmse_report, rmse
    from dc26 import log_experiment
    X, meta, y, tr, va, es, keep = _load()
    apt = X["AIRPORT"].astype(str).to_numpy()
    g = np.load(ROOT / "experiments" / "m13_preds_t.npz")
    d = np.load(NPZ, allow_pickle=True)
    y_va, vp = y[va].to_numpy(), meta.loc[va, "val_part"].to_numpy()
    y_es = y[tr & es].to_numpy()

    P_va = np.vstack([g["lgbm_va"], g["cb_va"], d["air_lgbm_va"], d["air_cb_va"]])
    P_es = np.vstack([g["lgbm_es"], g["cb_es"], d["air_lgbm_es"], d["air_cb_es"]])
    names = ["glob_lgbm", "glob_cb", "air_lgbm", "air_cb"]
    for i, n in enumerate(names):
        ok = np.isfinite(P_va[i])
        print(f"  {n}: mirror={rmse(y_va[ok], np.clip(P_va[i][ok], 60, CAP)):.1f} "
              f"(n={ok.sum():,})", flush=True)

    grid = np.arange(0, 1.0001, 0.1)
    cands = [np.array([a, b, c, max(0.0, 1 - a - b - c)])
             for a in grid for b in grid for c in grid if a + b + c <= 1.0001]
    pred = np.zeros(len(y_va))
    apt_es, apt_va = apt[tr & es], apt[va]
    for a in np.unique(apt_va):
        me, mv = apt_es == a, apt_va == a
        Pe, Pv = P_es[:, me], P_va[:, mv]
        ok = np.all(np.isfinite(Pe), axis=1)
        avail = np.where(ok)[0]
        best = (None, 1e9)
        for w in cands:
            if w[~np.isin(np.arange(4), avail)].sum() > 0:
                continue
            s = rmse(y_es[me], w @ Pe)
            if s < best[1]:
                best = (w, s)
        w = best[0]
        pred[mv] = w @ Pv
        print(f"  [{a}] w={np.round(w, 2)} ES={best[1]:.1f} "
              f"mirror={rmse(y_va[mv], np.clip(pred[mv], 60, CAP)):.1f}", flush=True)
    np.save(ROOT / "experiments" / "m19_weights.npy",
            np.array([1.0]))  # placeholder, per-airport weights printed above
    rep = rmse_report(y_va, np.clip(pred, 60, CAP), vp)
    print(f"[blend] M19 per-airport ensemble: overall={rep['overall']:.1f} "
          f"jan={rep['jan']:.1f} jul={rep['jul']:.1f}", flush=True)
    log_experiment("M19 per-airport ensemble", model="global+per-airport LGBM/CB",
                   features="v2+turnaround", val_rmse=rep["overall"],
                   val_rmse_jan=rep["jan"], val_rmse_jul=rep["jul"],
                   notes="per-airport blend weights on ES", keep="KEEP")


if __name__ == "__main__":
    {"air_lgbm": stage_air_lgbm, "air_cb": stage_air_cb, "blend": stage_blend}[sys.argv[1]]()
