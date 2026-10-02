"""M13 (staged): 3-GBDT ensemble on v2 features, then full-2025 refit -> submission v5.

Stages:
  build   : X_v2 for mirror (refs fitted on mirror-train), for full 2025, and for ranking
  lgbm/cb/xgb : mirror models -> val + ES predictions
  blend   : ES-optimized weights + mirror RMSE
  final   : refit the 3 models on all 2025 (fixed rounds), predict ranking, write v5
"""
import gc
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
PROC = ROOT / "data" / "processed"
NPZ = ROOT / "experiments" / "m13_preds.npz"
CAP = 86400

TURN = "--turnaround" in sys.argv
SUF = "_t" if TURN else ""
NPZ = ROOT / "experiments" / f"m13_preds{SUF}.npz"
X_MIRROR = PROC / f"X_v2_mirror{SUF}.parquet"
X_FULL = PROC / f"X_v2_full{SUF}.parquet"
X_RANK = PROC / f"X_v2_ranking{SUF}.parquet"
META = PROC / "meta_train2025.parquet"


def stage_build():
    from dc26 import load_movements_2025, load_ranking_2026, mirror_split
    from dc26.weather import load_metar
    from dc26.pipeline import fit_reference_times, build_features_v2, align_categories

    mv = load_movements_2025()
    dep = mv[mv["PHASE_mvt"] == "DEP"].copy()
    y = dep["TAXITIME_SEC_mvt"]
    met = load_metar()

    train, val, val_part = mirror_split(dep)
    tr_mask = dep.index.isin(train.index)

    # meta (mirror split bookkeeping)
    meta = pd.DataFrame({
        "y": y.astype("float32").to_numpy(),
        "tr": tr_mask,
        "va": dep.index.isin(val.index),
        "es": (dep["MVT_TIME_UTC_mvt"] >= "2025-12-10").to_numpy(),
    })
    vp = np.full(len(dep), "", dtype=object)
    vp[meta["va"].to_numpy()] = val_part
    meta["val_part"] = vp
    meta.to_parquet(META, index=False)

    refs_mirror = fit_reference_times(dep, y, sample_mask=tr_mask)
    Xm = build_features_v2(dep, mv, refs_mirror, met=met, state_cache="train2025",
                           with_turnaround=TURN)
    Xm.reset_index(drop=True).to_parquet(X_MIRROR, index=False)
    print(f"[build] mirror X: {Xm.shape}", flush=True)

    refs_full = fit_reference_times(dep, y)
    Xf = build_features_v2(dep, mv, refs_full, met=met, state_cache="train2025",
                           with_turnaround=TURN)
    Xf.reset_index(drop=True).to_parquet(X_FULL, index=False)
    print(f"[build] full X: {Xf.shape}", flush=True)
    del Xm
    gc.collect()

    rk = load_ranking_2026()
    rk_dep = rk[rk["PHASE_mvt"] == "DEP"].copy()
    Xr = build_features_v2(rk_dep, rk, refs_full, met=met, state_cache="ranking2026",
                           with_turnaround=TURN)
    Xr = align_categories(Xr, Xf)
    Xr.reset_index(drop=True).to_parquet(X_RANK, index=False)
    rk_dep[["MVT_ID_mvt"]].reset_index(drop=True).to_parquet(PROC / f"rank_ids{SUF}.parquet", index=False)
    print(f"[build] ranking X: {Xr.shape} | ref coverage "
          f"{Xr['ref_txot_p10'].notna().mean():.3f}", flush=True)


def _mirror():
    X = pd.read_parquet(X_MIRROR)
    meta = pd.read_parquet(META)
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
    X, meta, y, tr, va, es, keep = _mirror()
    cat = [c for c in X.columns if X[c].dtype.name == "category"]
    yk, esk = y[keep].clip(upper=CAP), es[keep]
    m = lgb.train(REG_PARAMS, lgb.Dataset(X[keep][~esk], yk[~esk], categorical_feature=cat),
                  num_boost_round=6000, valid_sets=[lgb.Dataset(X[keep][esk], yk[esk])],
                  callbacks=[lgb.early_stopping(150, verbose=False)])
    print(f"[lgbm] iters={m.best_iteration}", flush=True)
    _save(lgbm_va=m.predict(X[va], num_iteration=m.best_iteration),
          lgbm_es=m.predict(X[tr & es], num_iteration=m.best_iteration),
          lgbm_iters=m.best_iteration)


def stage_cb():
    from catboost import CatBoostRegressor, Pool
    X, meta, y, tr, va, es, keep = _mirror()
    cat = [c for c in X.columns if X[c].dtype.name == "category"]
    for c in cat:
        X[c] = X[c].cat.add_categories("__NA__").fillna("__NA__")
    yk, esk = y[keep].clip(upper=CAP), es[keep]
    m = CatBoostRegressor(loss_function="RMSE", iterations=1500, learning_rate=0.1,
                          depth=7, l2_leaf_reg=3.0, random_seed=42, od_type="Iter",
                          od_wait=80, verbose=300, thread_count=-1,
                          max_ctr_complexity=1, cat_features=cat)
    m.fit(Pool(X[keep][~esk], yk[~esk], cat_features=cat),
          eval_set=Pool(X[keep][esk], yk[esk], cat_features=cat))
    print(f"[cb] iters={m.get_best_iteration()}", flush=True)
    _save(cb_va=m.predict(X[va]), cb_es=m.predict(X[tr & es]), cb_iters=m.get_best_iteration())


def stage_xgb():
    import xgboost as xgb
    X, meta, y, tr, va, es, keep = _mirror()
    yk, esk = y[keep].clip(upper=CAP), es[keep]
    params = dict(objective="reg:squarederror", eval_metric="rmse", tree_method="hist",
                  max_depth=0, grow_policy="lossguide", max_leaves=255, eta=0.06,
                  subsample=0.8, colsample_bytree=0.9, reg_lambda=1.0,
                  min_child_weight=100, nthread=0, seed=42)
    m = xgb.train(params, xgb.DMatrix(X[keep][~esk], label=yk[~esk], enable_categorical=True),
                  num_boost_round=4000,
                  evals=[(xgb.DMatrix(X[keep][esk], label=yk[esk], enable_categorical=True), "es")],
                  early_stopping_rounds=150, verbose_eval=300)
    print(f"[xgb] iters={m.best_iteration}", flush=True)
    rng = (0, m.best_iteration + 1)
    _save(xgb_va=m.predict(xgb.DMatrix(X[va], enable_categorical=True), iteration_range=rng),
          xgb_es=m.predict(xgb.DMatrix(X[tr & es], enable_categorical=True), iteration_range=rng),
          xgb_iters=m.best_iteration)


def stage_blend():
    from dc26.validation import rmse_report, rmse
    from dc26 import log_experiment
    X, meta, y, tr, va, es, keep = _mirror()
    d = np.load(NPZ)
    y_es, y_va = y[tr & es].to_numpy(), y[va].to_numpy()
    vp = meta.loc[va, "val_part"].to_numpy()

    names = [n for n in ["lgbm", "cb", "xgb"] if f"{n}_es" in d]
    P_es = np.vstack([d[f"{n}_es"] for n in names])
    P_va = np.vstack([d[f"{n}_va"] for n in names])
    for i, n in enumerate(names):
        r = rmse_report(y_va, P_va[i].clip(60, CAP), vp)
        print(f"  {n}: ES={rmse(y_es, P_es[i]):.2f} | mirror={r['overall']:.1f} "
              f"(jan {r['jan']:.1f}, jul {r['jul']:.1f})", flush=True)

    grid = np.arange(0, 1.0001, 0.05)
    best = (None, 1e9)
    if len(names) == 2:
        cands = [np.array([g, 1 - g]) for g in grid]
    else:
        cands = [np.array([w1, w2, max(0.0, 1 - w1 - w2)])
                 for w1 in grid for w2 in grid if w1 + w2 <= 1.0001]
    for w in cands:
        sc = rmse(y_es, w @ P_es)
        if sc < best[1]:
            best = (w, sc)
    w = best[0]
    rep = rmse_report(y_va, (w @ P_va).clip(60, CAP), vp)
    print(f"[blend] w={np.round(w, 2)} ES={best[1]:.2f} | M13 mirror: overall={rep['overall']:.1f} "
          f"jan={rep['jan']:.1f} jul={rep['jul']:.1f}", flush=True)
    np.save(ROOT / "experiments" / f"m13_weights{SUF}.npy", w)
    log_experiment(f"M13{SUF} ensemble + v2 features", model="LGBM+CB+XGB",
                   features="v2: +PRU refP10, implied_add, ATFM daily, ATXOT",
                   val_rmse=rep["overall"], val_rmse_jan=rep["jan"], val_rmse_jul=rep["jul"],
                   notes=f"w={np.round(w, 2).tolist()}", keep="KEEP")


def stage_final():
    import lightgbm as lgb
    import xgboost as xgb
    from catboost import CatBoostRegressor, Pool
    from dc26.mixture import REG_PARAMS

    X = pd.read_parquet(X_FULL)
    Xr = pd.read_parquet(X_RANK)
    meta = pd.read_parquet(META)
    y = meta["y"].astype("float64")
    d = np.load(NPZ)
    w = np.load(ROOT / "experiments" / "m13_weights.npy")
    keep = (y > 0).to_numpy() & (y < 200000).to_numpy()
    yk = y[keep].clip(upper=CAP)
    cat = [c for c in X.columns if X[c].dtype.name == "category"]
    boost = 1.12  # more data than the mirror fit -> slightly more rounds

    n = int(d["lgbm_iters"] * boost)
    ml = lgb.train(REG_PARAMS, lgb.Dataset(X[keep], yk, categorical_feature=cat), num_boost_round=n)
    pl = ml.predict(Xr)
    ml.save_model(str(ROOT / "models" / "m13_lgbm_full.txt"))
    print(f"[final] lgbm done ({n} rounds)", flush=True)

    Xc, Xrc = X.copy(), Xr.copy()
    for c in cat:
        Xc[c] = Xc[c].cat.add_categories("__NA__").fillna("__NA__")
        Xrc[c] = Xrc[c].cat.add_categories("__NA__").fillna("__NA__")
    n = int(d["cb_iters"] * boost)
    mc = CatBoostRegressor(loss_function="RMSE", iterations=n, learning_rate=0.1, depth=7,
                           l2_leaf_reg=3.0, random_seed=42, verbose=300, thread_count=-1,
                           max_ctr_complexity=1, cat_features=cat)
    mc.fit(Pool(Xc[keep], yk, cat_features=cat))
    pc = mc.predict(Xrc)
    mc.save_model(str(ROOT / "models" / "m13_catboost_full.cbm"))
    print(f"[final] catboost done ({n} rounds)", flush=True)
    del Xc, Xrc
    gc.collect()

    params = dict(objective="reg:squarederror", eval_metric="rmse", tree_method="hist",
                  max_depth=0, grow_policy="lossguide", max_leaves=255, eta=0.06,
                  subsample=0.8, colsample_bytree=0.9, reg_lambda=1.0,
                  min_child_weight=100, nthread=0, seed=42)
    n = int(d["xgb_iters"] * boost)
    mx = xgb.train(params, xgb.DMatrix(X[keep], label=yk, enable_categorical=True),
                   num_boost_round=n)
    px = mx.predict(xgb.DMatrix(Xr, enable_categorical=True))
    mx.save_model(str(ROOT / "models" / "m13_xgb_full.json"))
    print(f"[final] xgb done ({n} rounds)", flush=True)

    pred = (w[0] * pl + w[1] * pc + w[2] * px).clip(60, CAP)
    ids = pd.read_parquet(PROC / "rank_ids.parquet")["MVT_ID_mvt"]
    tpl = pd.read_parquet(ROOT / "data" / "submitting.parquet").drop(columns=["TAXITIME_SEC_mvt"])
    out = tpl.merge(pd.DataFrame({"MVT_ID_mvt": ids.values, "TAXITIME_SEC_mvt": pred}),
                    on="MVT_ID_mvt", how="left")
    assert out["TAXITIME_SEC_mvt"].notna().all() and len(out) == 215876
    path = ROOT / "submissions" / "reliable-hamburger_v5.parquet"
    out.to_parquet(path, index=False)
    print(f"[final] v5 written: mean={out['TAXITIME_SEC_mvt'].mean():.0f}s "
          f"median={out['TAXITIME_SEC_mvt'].median():.0f}s "
          f"q99={out['TAXITIME_SEC_mvt'].quantile(.99):.0f}s", flush=True)


if __name__ == "__main__":
    {"build": stage_build, "lgbm": stage_lgbm, "cb": stage_cb, "xgb": stage_xgb,
     "blend": stage_blend, "final": stage_final}[sys.argv[1]]()
