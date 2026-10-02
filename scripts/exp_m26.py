"""M26 (staged): final architecture on the mirror split.

per-airport tuned LGBM (3 seeds averaged) + per-airport CatBoost + global LGBM,
blended with per-airport weights fitted on the early-stopping window.
Feature set: v2 + turnaround + flight context + departure regime.

stages: lgbm | cb | glob | blend
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
PROC = ROOT / "data" / "processed"
NPZ = ROOT / "experiments" / "m26_preds.npz"
CAP = 86400
SEEDS = [42, 1337, 2024]
MIN_TRAIN = 20000


def _load():
    X = pd.concat([pd.read_parquet(PROC / "X_v2_mirror_t.parquet"),
                   pd.read_parquet(PROC / "flightctx_train2025.parquet"),
                   pd.read_parquet(PROC / "depregime_train2025.parquet")], axis=1)
    meta = pd.read_parquet(PROC / "meta_train2025.parquet")
    y = meta["y"].astype("float64")
    tr, va, es = meta["tr"].to_numpy(), meta["va"].to_numpy(), meta["es"].to_numpy()
    keep = tr & (y > 0).to_numpy() & (y < 200000).to_numpy()
    return X, meta, y, tr, va, es, keep


def _save(**kw):
    d = dict(np.load(NPZ, allow_pickle=True)) if NPZ.exists() else {}
    d.update(kw)
    np.savez(NPZ, **d)
    assert set(kw) <= set(np.load(NPZ, allow_pickle=True).keys())


def tuned_params(a):
    p = ROOT / "experiments" / f"optuna_{a}.json"
    base = dict(objective="regression", metric="rmse", verbosity=-1,
                num_threads=0, bagging_freq=1, seed=42)
    if not p.exists():
        from dc26.mixture import REG_PARAMS
        return dict(REG_PARAMS)
    d = json.loads(p.read_text())
    base.update({k: v for k, v in d.items() if not k.startswith("_")})
    base["num_leaves"] = int(base["num_leaves"])
    base["min_data_in_leaf"] = int(base["min_data_in_leaf"])
    base["max_bin"] = int(base["max_bin"])
    return base


def stage_lgbm():
    import lightgbm as lgb
    X, meta, y, tr, va, es, keep = _load()
    cat = [c for c in X.columns if X[c].dtype.name == "category"]
    apt = X["AIRPORT"].astype(str).to_numpy()
    es_rows = tr & es
    P = np.full((len(SEEDS), va.sum()), np.nan)
    Pe = np.full((len(SEEDS), es_rows.sum()), np.nan)
    iters = {}
    for a in np.unique(apt):
        fit, ev = keep & (apt == a) & ~es, keep & (apt == a) & es
        sv, se = apt[va] == a, apt[es_rows] == a
        if fit.sum() < MIN_TRAIN or not sv.any():
            continue
        for si, seed in enumerate(SEEDS):
            p = dict(tuned_params(a), seed=seed, bagging_seed=seed + 1,
                     feature_fraction_seed=seed + 2)
            m = lgb.train(p, lgb.Dataset(X[fit], y[fit].clip(upper=CAP), categorical_feature=cat),
                          num_boost_round=6000,
                          valid_sets=[lgb.Dataset(X[ev], y[ev].clip(upper=CAP))],
                          callbacks=[lgb.early_stopping(150, verbose=False)])
            P[si, sv] = m.predict(X[va][sv], num_iteration=m.best_iteration)
            if se.any():
                Pe[si, se] = m.predict(X[es_rows][se], num_iteration=m.best_iteration)
            iters.setdefault(a, []).append(m.best_iteration)
        print(f"[lgbm] {a}: iters={iters[a]}", flush=True)
    _save(air_lgbm_va=np.nanmean(P, axis=0), air_lgbm_es=np.nanmean(Pe, axis=0),
          air_lgbm_iters=json.dumps({k: v for k, v in iters.items()}))


def stage_cb():
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
        fit, ev = keep & (apt == a) & ~es, keep & (apt == a) & es
        sv, se = apt[va] == a, apt[es_rows] == a
        if fit.sum() < MIN_TRAIN or not sv.any():
            continue
        m = CatBoostRegressor(loss_function="RMSE", iterations=1500, learning_rate=0.1,
                              depth=7, l2_leaf_reg=3.0, random_seed=42, od_type="Iter",
                              od_wait=80, verbose=False, thread_count=-1,
                              max_ctr_complexity=1, cat_features=cat)
        m.fit(Pool(X[fit], y[fit].clip(upper=CAP), cat_features=cat),
              eval_set=Pool(X[ev], y[ev].clip(upper=CAP), cat_features=cat))
        p_va[sv] = m.predict(X[va][sv])
        if se.any():
            p_es[se] = m.predict(X[es_rows][se])
        iters[a] = int(m.get_best_iteration())
        print(f"[cb] {a}: iters={iters[a]}", flush=True)
    _save(air_cb_va=p_va, air_cb_es=p_es, air_cb_iters=json.dumps(iters))


def stage_glob():
    import lightgbm as lgb
    from dc26.mixture import REG_PARAMS
    X, meta, y, tr, va, es, keep = _load()
    cat = [c for c in X.columns if X[c].dtype.name == "category"]
    es_rows = tr & es
    m = lgb.train(REG_PARAMS, lgb.Dataset(X[keep & ~es], y[keep & ~es].clip(upper=CAP),
                                          categorical_feature=cat),
                  num_boost_round=6000,
                  valid_sets=[lgb.Dataset(X[keep & es], y[keep & es].clip(upper=CAP))],
                  callbacks=[lgb.early_stopping(150, verbose=False)])
    print(f"[glob] iters={m.best_iteration}", flush=True)
    _save(glob_lgbm_va=m.predict(X[va], num_iteration=m.best_iteration),
          glob_lgbm_es=m.predict(X[es_rows], num_iteration=m.best_iteration),
          glob_lgbm_iters=m.best_iteration)


def stage_blend():
    from dc26.validation import rmse, rmse_report
    from dc26 import log_experiment
    X, meta, y, tr, va, es, keep = _load()
    apt = X["AIRPORT"].astype(str).to_numpy()
    d = np.load(NPZ, allow_pickle=True)
    y_va, y_es = y[va].to_numpy(), y[tr & es].to_numpy()
    vp = meta.loc[va, "val_part"].to_numpy()
    P_va = np.vstack([d["air_lgbm_va"], d["air_cb_va"], d["glob_lgbm_va"]])
    P_es = np.vstack([d["air_lgbm_es"], d["air_cb_es"], d["glob_lgbm_es"]])
    for i, n in enumerate(["air_lgbm(tuned,3seed)", "air_cb", "glob_lgbm"]):
        print(f"  {n}: mirror={rmse_report(y_va, np.clip(P_va[i], 60, CAP), vp)['overall']:.1f}", flush=True)

    grid = np.arange(0, 1.0001, 0.05)
    cands = [np.array([a, b, max(0.0, 1 - a - b)]) for a in grid for b in grid if a + b <= 1.0001]
    pred = np.zeros(len(y_va))
    weights = {}
    apt_es, apt_va = apt[tr & es], apt[va]
    for a in np.unique(apt_va):
        me, mv = apt_es == a, apt_va == a
        Pe = P_es[:, me]
        ok = np.all(np.isfinite(Pe), axis=1)
        best = (None, 1e18)
        for w in cands:
            if w[~ok].sum() > 0:
                continue
            s = rmse(y_es[me], w @ Pe)
            if s < best[1]:
                best = (w, s)
        pred[mv] = best[0] @ P_va[:, mv]
        weights[a] = best[0].tolist()
        print(f"  [{a}] w={np.round(best[0], 2)} ES={best[1]:.1f} "
              f"mirror={rmse(y_va[mv], np.clip(pred[mv], 60, CAP)):.1f}", flush=True)
    (ROOT / "experiments" / "m26_weights.json").write_text(json.dumps(weights))
    rep = rmse_report(y_va, np.clip(pred, 60, CAP), vp)
    print(f"[blend] M26 FINAL: overall={rep['overall']:.1f} jan={rep['jan']:.1f} "
          f"jul={rep['jul']:.1f} | predicted leaderboard ~{rep['overall'] * 0.867:.0f}", flush=True)
    log_experiment("M26 final: tuned per-airport ensemble", model="tuned LGBM x3 + CB + global",
                   features="v2+turnaround+flightctx+depregime",
                   val_rmse=rep["overall"], val_rmse_jan=rep["jan"], val_rmse_jul=rep["jul"],
                   notes="Optuna params per airport", keep="KEEP")


if __name__ == "__main__":
    {"lgbm": stage_lgbm, "cb": stage_cb, "glob": stage_glob, "blend": stage_blend}[sys.argv[1]]()
