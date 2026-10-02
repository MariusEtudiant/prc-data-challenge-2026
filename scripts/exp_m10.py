"""M10 (staged): 3-GBDT ensemble on the m3bw config (extremes kept, cap 86400).
Stages: lgbm -> cb -> xgb -> blend. Weights optimized on the ES window (Dec 10-31).
Uses cached X_train2025.parquet / meta_train2025.parquet (M5 feature set)."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
PROC = ROOT / "data" / "processed"
NPZ = ROOT / "experiments" / "m10_preds.npz"
CAP = 86400


def _load():
    X = pd.read_parquet(PROC / "X_train2025.parquet")
    meta = pd.read_parquet(PROC / "meta_train2025.parquet")
    y = meta["y"].astype("float64")
    tr, va, es = meta["tr"].to_numpy(), meta["va"].to_numpy(), meta["es"].to_numpy()
    keep = tr & (y > 0).to_numpy() & (y < 200000).to_numpy()
    return X, meta, y, tr, va, es, keep


def _save(**kw):
    d = dict(np.load(NPZ)) if NPZ.exists() else {}
    d.update(kw)
    np.savez(NPZ, **d)


def stage_lgbm():
    import lightgbm as lgb
    from dc26.mixture import REG_PARAMS
    X, meta, y, tr, va, es, keep = _load()
    cat_cols = [c for c in X.columns if X[c].dtype.name == "category"]
    yk = y[keep].clip(upper=CAP)
    es_k = es[keep]
    m = lgb.train(REG_PARAMS, lgb.Dataset(X[keep][~es_k], yk[~es_k], categorical_feature=cat_cols),
                  num_boost_round=6000,
                  valid_sets=[lgb.Dataset(X[keep][es_k], yk[es_k])],
                  callbacks=[lgb.early_stopping(150, verbose=False)])
    print(f"[lgbm] iters={m.best_iteration}", flush=True)
    _save(lgbm_va=m.predict(X[va], num_iteration=m.best_iteration),
          lgbm_es=m.predict(X[tr & es], num_iteration=m.best_iteration))
    m.save_model(str(ROOT / "models" / "m10_lgbm.txt"))


def stage_cb():
    from catboost import CatBoostRegressor, Pool
    X, meta, y, tr, va, es, keep = _load()
    cat_cols = [c for c in X.columns if X[c].dtype.name == "category"]
    for c in cat_cols:
        X[c] = X[c].cat.add_categories("__NA__").fillna("__NA__")
    yk = y[keep].clip(upper=CAP)
    es_k = es[keep]
    m = CatBoostRegressor(loss_function="RMSE", iterations=1500, learning_rate=0.1,
                          depth=7, l2_leaf_reg=3.0, random_seed=42, od_type="Iter",
                          od_wait=80, verbose=200, thread_count=-1,
                          max_ctr_complexity=1, cat_features=cat_cols)
    m.fit(Pool(X[keep][~es_k], yk[~es_k], cat_features=cat_cols),
          eval_set=Pool(X[keep][es_k], yk[es_k], cat_features=cat_cols))
    print(f"[cb] best_iteration={m.get_best_iteration()}", flush=True)
    _save(cb_va=m.predict(X[va]), cb_es=m.predict(X[tr & es]))
    m.save_model(str(ROOT / "models" / "m10_catboost.cbm"))


def stage_xgb():
    import xgboost as xgb
    X, meta, y, tr, va, es, keep = _load()
    yk = y[keep].clip(upper=CAP)
    es_k = es[keep]
    dtr = xgb.DMatrix(X[keep][~es_k], label=yk[~es_k], enable_categorical=True)
    des = xgb.DMatrix(X[keep][es_k], label=yk[es_k], enable_categorical=True)
    params = dict(objective="reg:squarederror", eval_metric="rmse", tree_method="hist",
                  max_depth=0, grow_policy="lossguide", max_leaves=255, eta=0.06,
                  subsample=0.8, colsample_bytree=0.9, reg_lambda=1.0, min_child_weight=100,
                  nthread=0, seed=42)
    m = xgb.train(params, dtr, num_boost_round=4000, evals=[(des, "es")],
                  early_stopping_rounds=150, verbose_eval=200)
    print(f"[xgb] best_iteration={m.best_iteration}", flush=True)
    _save(xgb_va=m.predict(xgb.DMatrix(X[va], enable_categorical=True), iteration_range=(0, m.best_iteration + 1)),
          xgb_es=m.predict(xgb.DMatrix(X[tr & es], enable_categorical=True), iteration_range=(0, m.best_iteration + 1)))
    m.save_model(str(ROOT / "models" / "m10_xgb.json"))


def stage_blend():
    from dc26.validation import rmse_report, rmse
    from dc26 import log_experiment
    X, meta, y, tr, va, es, keep = _load()
    d = np.load(NPZ)
    y_es = y[tr & es].to_numpy()
    y_va = y[va].to_numpy()
    vp = meta.loc[va, "val_part"].to_numpy()

    P_es = np.vstack([d["lgbm_es"], d["cb_es"], d["xgb_es"]])
    P_va = np.vstack([d["lgbm_va"], d["cb_va"], d["xgb_va"]])
    names = ["lgbm", "cb", "xgb"]
    for i, n in enumerate(names):
        print(f"  {n}: ES={rmse(y_es, P_es[i]):.2f} | val={rmse_report(y_va, P_va[i].clip(60, CAP), vp)}", flush=True)

    best = (None, 1e9)
    grid = np.arange(0, 1.0001, 0.05)
    for w1 in grid:
        for w2 in grid:
            if w1 + w2 > 1:
                continue
            w = np.array([w1, w2, 1 - w1 - w2])
            s = rmse(y_es, w @ P_es)
            if s < best[1]:
                best = (w, s)
    w = best[0]
    print(f"[blend] weights lgbm/cb/xgb = {np.round(w, 2)} | ES rmse={best[1]:.2f}", flush=True)
    pred = (w @ P_va).clip(60, CAP)
    rep = rmse_report(y_va, pred, vp)
    print(f"[blend] M10 ensemble: overall={rep['overall']:.1f} jan={rep['jan']:.1f} jul={rep['jul']:.1f}", flush=True)
    np.savez(ROOT / "experiments" / "m10_blend.npz", w=w)
    log_experiment("M10 3-GBDT ensemble (m3bw cfg)", model="LGBM+CB+XGB blend",
                   features="M5 features; extremes kept cap 86400",
                   val_rmse=rep["overall"], val_rmse_jan=rep["jan"], val_rmse_jul=rep["jul"],
                   notes=f"w={np.round(w, 2).tolist()}", keep="KEEP")


if __name__ == "__main__":
    {"lgbm": stage_lgbm, "cb": stage_cb, "xgb": stage_xgb, "blend": stage_blend}[sys.argv[1]]()
