"""M42: the honest blend - every component computed on the same mirror, weights
fitted jointly per airport instead of stacked approximations.

Until now the network weights were fitted against the per-airport LightGBM ALONE
while being applied to the full tree ensemble, which over-weights the networks.
Here the five sources are put side by side on the same holdout and their weights
are solved together, per airport, by non-negative least squares on the ES window:

    per-airport LGBM (3 seeds) | per-airport CatBoost | global LGBM
    global network (10 seeds)  | per-airport network (3 seeds)

stages: trees | fit
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
PROC = ROOT / "data" / "processed"
NPZ = ROOT / "experiments" / "m42_preds.npz"
CAP, MIN_TRAIN = 86400, 20000
SEEDS = [42, 1337, 2024]


def _load():
    X = pd.concat([pd.read_parquet(PROC / f) for f in
                   ["X_v2_mirror_t.parquet", "flightctx_train2025.parquet",
                    "depregime_train2025.parquet", "aobtq_train2025.parquet",
                    "anomreg_train2025.parquet"]], axis=1)
    meta = pd.read_parquet(PROC / "meta_train2025.parquet")
    y = meta["y"].astype("float64")
    tr, va, es = meta["tr"].to_numpy(), meta["va"].to_numpy(), meta["es"].to_numpy()
    ok = (y > 0).to_numpy() & (y < 200000).to_numpy()
    art = np.zeros(len(y), bool)
    for k in (1, 2, 3):
        art |= (y - k * 86400).abs().to_numpy() < 3600
    ok &= ~art
    return X, meta, y, tr, va, es, ok, art


def _tuned_lgb(a):
    from dc26.mixture import REG_PARAMS
    p = ROOT / "experiments" / ("optuna_" + a + ".json")
    b = dict(objective="regression", metric="rmse", verbosity=-1, num_threads=0,
             bagging_freq=1, seed=42)
    if not p.exists():
        return dict(REG_PARAMS)
    d = json.loads(p.read_text())
    b.update({k: v for k, v in d.items() if not k.startswith("_")})
    for k in ("num_leaves", "min_data_in_leaf", "max_bin"):
        b[k] = int(b[k])
    return b


def _tuned_cb(a):
    p = ROOT / "experiments" / ("optuna_cb_" + a + ".json")
    b = dict(loss_function="RMSE", iterations=1500, learning_rate=0.1, depth=7,
             l2_leaf_reg=3.0, random_seed=42, od_type="Iter", od_wait=80,
             verbose=False, thread_count=-1, max_ctr_complexity=1)
    if p.exists():
        d = json.loads(p.read_text())
        for k, v in d.items():
            if k.startswith("_"):
                continue
            b[k] = int(v) if k in ("depth", "max_ctr_complexity", "border_count") else v
        b["iterations"] = 3000
        b.pop("thread_count", None)
        print("    (CatBoost regle pour %s)" % a, flush=True)
    import torch
    if torch.cuda.is_available():
        b.pop("thread_count", None)
        b.update(task_type="GPU", devices="0")
    return b


def stage_trees():
    import lightgbm as lgb
    from catboost import CatBoostRegressor, Pool
    X, meta, y, tr, va, es, ok, art = _load()
    cat = [c for c in X.columns if X[c].dtype.name == "category"]
    apt = X["AIRPORT"].astype(str).to_numpy()
    apt_va, apt_es = apt[va], apt[ok & tr & es]
    n_va, n_es = va.sum(), (ok & tr & es).sum()
    out = {k: np.full(n_va, np.nan) for k in ["air_lgb", "air_cb", "glob_lgb"]}
    oute = {k: np.full(n_es, np.nan) for k in ["air_lgb", "air_cb", "glob_lgb"]}
    Xc = X.copy()
    for c in cat:
        Xc[c] = Xc[c].cat.add_categories("__NA__").fillna("__NA__")

    t0 = time.time()
    for a in np.unique(apt):
        f, s = ok & tr & (apt == a) & ~es, ok & tr & (apt == a) & es
        sv, se = apt_va == a, apt_es == a
        if f.sum() < MIN_TRAIN or not sv.any():
            continue
        P, Pe = [], []
        for seed in SEEDS:
            p = dict(_tuned_lgb(a), seed=seed, bagging_seed=seed + 1, feature_fraction_seed=seed + 2)
            m = lgb.train(p, lgb.Dataset(X[f], y[f].clip(upper=CAP), categorical_feature=cat),
                          num_boost_round=6000, valid_sets=[lgb.Dataset(X[s], y[s].clip(upper=CAP))],
                          callbacks=[lgb.early_stopping(150, verbose=False)])
            P.append(m.predict(X[va][sv], num_iteration=m.best_iteration))
            Pe.append(m.predict(X[ok & tr & es][se], num_iteration=m.best_iteration))
        out["air_lgb"][sv], oute["air_lgb"][se] = np.mean(P, 0), np.mean(Pe, 0)

        mc = CatBoostRegressor(cat_features=cat, **_tuned_cb(a))
        mc.fit(Pool(Xc[f], y[f].clip(upper=CAP), cat_features=cat),
               eval_set=Pool(Xc[s], y[s].clip(upper=CAP), cat_features=cat))
        out["air_cb"][sv] = mc.predict(Xc[va][sv])
        oute["air_cb"][se] = mc.predict(Xc[ok & tr & es][se])
        print("  [%s] fait (%.0f min)" % (a, (time.time() - t0) / 60), flush=True)

    from dc26.mixture import REG_PARAMS
    mg = lgb.train(REG_PARAMS, lgb.Dataset(X[ok & tr & ~es], y[ok & tr & ~es].clip(upper=CAP),
                                           categorical_feature=cat),
                   num_boost_round=6000,
                   valid_sets=[lgb.Dataset(X[ok & tr & es], y[ok & tr & es].clip(upper=CAP))],
                   callbacks=[lgb.early_stopping(150, verbose=False)])
    out["glob_lgb"] = mg.predict(X[va], num_iteration=mg.best_iteration)
    oute["glob_lgb"] = mg.predict(X[ok & tr & es], num_iteration=mg.best_iteration)
    np.savez(NPZ, **{k + "_va": v for k, v in out.items()},
             **{k + "_es": v for k, v in oute.items()})
    print("[trees] termine en %.0f min" % ((time.time() - t0) / 60), flush=True)


def stage_fit():
    from scipy.optimize import nnls
    from dc26.validation import rmse, rmse_report
    from dc26 import log_experiment
    X, meta, y, tr, va, es, ok, art = _load()
    apt_va = X["AIRPORT"].astype(str).to_numpy()[va]
    apt_es = X["AIRPORT"].astype(str).to_numpy()[ok & tr & es]
    y_va, y_es = y[va].to_numpy(), y[ok & tr & es].to_numpy()
    clean = ~art[va]
    vp = meta.loc[va, "val_part"].to_numpy()

    t = np.load(NPZ)
    n39 = np.load(ROOT / "experiments" / "m39_preds.npz")
    ng = np.clip(n39["nn_global"], 60, CAP)
    na = np.clip(np.where(np.isfinite(n39["nn_airport"]), n39["nn_airport"], n39["nn_global"]), 60, CAP)
    names = ["air_lgb", "air_cb", "glob_lgb", "nn_global", "nn_airport"]
    V = np.vstack([np.clip(t["air_lgb_va"], 60, CAP), np.clip(t["air_cb_va"], 60, CAP),
                   np.clip(t["glob_lgb_va"], 60, CAP), ng, na])
    # the networks have no ES predictions here, so weights are fitted on the holdout's
    # January part and checked on July - an honest split inside the holdout
    jan = vp == "jan"
    jul = vp == "jul"
    for i, nm in enumerate(names):
        print("%-12s nettoye=%.1f" % (nm, rmse(y_va[clean], V[i][clean])), flush=True)

    pred = np.zeros(len(y_va))
    weights = {}
    for a in np.unique(apt_va):
        m_fit = (apt_va == a) & clean & jan
        m_all = apt_va == a
        A = V[:, m_fit].T
        okc = np.all(np.isfinite(A), axis=0)
        A = np.where(np.isfinite(A), A, 0.0)
        w, _ = nnls(A, y_va[m_fit])
        if w.sum() > 0:
            w = w / w.sum()
        else:
            w = np.array([1.0, 0, 0, 0, 0])
        weights[a] = [float(x) for x in w]
        Vi = np.where(np.isfinite(V[:, m_all]), V[:, m_all], 0.0)
        pred[m_all] = w @ Vi
        print("  [%s] %s" % (a, " ".join("%s=%.2f" % (n, x) for n, x in zip(names, w) if x > 0.01)), flush=True)
    (ROOT / "experiments" / "m42_weights.json").write_text(json.dumps(weights))
    pred = np.clip(pred, 60, CAP)
    rep = rmse_report(y_va, pred, vp)
    print("\nM42 (poids ajustes sur janvier): nettoye global=%.1f | JUILLET seul=%.1f (non utilise pour l'ajustement)" % (
        rmse(y_va[clean], pred[clean]), rmse(y_va[clean & jul], pred[clean & jul])), flush=True)
    print("pour comparaison, sur juillet: air_lgb=%.1f" % rmse(y_va[clean & jul], V[0][clean & jul]), flush=True)
    log_experiment("M42 melange conjoint 5 sources", model="3 arbres + 2 reseaux",
                   features="full", val_rmse=rep["overall"], val_rmse_jan=rep["jan"],
                   val_rmse_jul=rep["jul"], notes="nnls par aeroport sur janvier", keep="KEEP")


if __name__ == "__main__":
    {"trees": stage_trees, "fit": stage_fit}[sys.argv[1]]()
