"""M4: regime mixture. final = p_ext * E[y|ext] + (1-p_ext) * E[y|normal].
Recovers bulk accuracy (M3a level) while keeping the tail gain (M3b level).
Variants: extreme threshold 7200s (M4a) and 3600s (M4b).
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from dc26 import load_movements_2025, mirror_split, log_experiment
from dc26.validation import rmse_report
from dc26.state_features import get_state_features
from dc26.features import build_features_v1

mv = load_movements_2025()
dep = mv[mv["PHASE_mvt"] == "DEP"].copy()
X_all = pd.concat([build_features_v1(dep), get_state_features("train2025", dep, mv)], axis=1)
cat_cols = [c for c in X_all.columns if X_all[c].dtype.name == "category"]

train, val, val_part = mirror_split(dep)
tr_mask = dep.index.isin(train.index)
va_mask = dep.index.isin(val.index)
X_va = X_all[va_mask]
y_va = dep.loc[va_mask, "TAXITIME_SEC_mvt"]
y_all = dep["TAXITIME_SEC_mvt"]

reg_params = dict(objective="regression", metric="rmse", learning_rate=0.06,
                  num_leaves=255, min_data_in_leaf=100, feature_fraction=0.9,
                  bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0,
                  num_threads=0, verbosity=-1, seed=42)


def fit_reg(mask, cap=None, params=None, es_date="2025-12-10"):
    p = dict(params or reg_params)
    y = y_all[mask]
    if cap:
        y = y.clip(upper=cap)
    es = (dep.loc[mask, "MVT_TIME_UTC_mvt"] >= es_date).to_numpy()
    m = lgb.train(p, lgb.Dataset(X_all[mask][~es], y[~es], categorical_feature=cat_cols),
                  num_boost_round=4000,
                  valid_sets=[lgb.Dataset(X_all[mask][es], y[es])],
                  callbacks=[lgb.early_stopping(150, verbose=False)])
    return m


def report(tag, pred):
    rep = rmse_report(y_va.to_numpy(), pred, val_part)
    buckets = pd.cut(y_va, [-np.inf, 1800, 7200, np.inf], labels=["<30m", "30m-2h", ">2h"])
    det = {b: float(np.sqrt(np.mean((pred[(buckets == b).to_numpy()] - y_va[(buckets == b).to_numpy()]) ** 2)))
           for b in ["<30m", "30m-2h", ">2h"]}
    print(f"{tag}: overall={rep['overall']:.1f} jan={rep['jan']:.1f} jul={rep['jul']:.1f} "
          f"| <30m={det['<30m']:.1f} 30m-2h={det['30m-2h']:.1f} >2h={det['>2h']:.1f}")
    return rep


results = {}
for tag, thr in [("M4a thr7200", 7200), ("M4b thr3600", 3600)]:
    ext = (y_all > thr) & tr_mask & (y_all < 200000)
    normal = tr_mask & y_all.between(30, thr)

    # classifier P(extreme)
    clf_params = dict(objective="binary", metric="auc", learning_rate=0.05, num_leaves=63,
                      min_data_in_leaf=50, feature_fraction=0.9, scale_pos_weight=50,
                      num_threads=0, verbosity=-1, seed=42)
    lab = ext[tr_mask].astype(int).to_numpy()
    es = (dep.loc[tr_mask, "MVT_TIME_UTC_mvt"] >= "2025-12-10").to_numpy()
    clf = lgb.train(clf_params, lgb.Dataset(X_all[tr_mask][~es], lab[~es], categorical_feature=cat_cols),
                    num_boost_round=2000,
                    valid_sets=[lgb.Dataset(X_all[tr_mask][es], lab[es])],
                    callbacks=[lgb.early_stopping(100, verbose=False)])
    p_raw = clf.predict(X_va, num_iteration=clf.best_iteration)
    # calibrate: scale_pos_weight distorts probabilities -> undo via odds correction
    p_ext = p_raw / (p_raw + (1 - p_raw) * 50)

    mA = fit_reg(normal)
    predA = mA.predict(X_va, num_iteration=mA.best_iteration)

    # E[y|ext]: small robust model on few hundred rows
    extB_params = dict(objective="regression", metric="rmse", learning_rate=0.05, num_leaves=15,
                       min_data_in_leaf=20, feature_fraction=0.8, num_threads=0, verbosity=-1, seed=42)
    yB = y_all[ext].clip(upper=86400)
    mB = lgb.train(extB_params, lgb.Dataset(X_all[ext], yB, categorical_feature=cat_cols),
                   num_boost_round=300)
    predB = np.maximum(mB.predict(X_va), float(thr))

    mix = (1 - p_ext) * predA + p_ext * predB
    print(f"[{tag}] ext train n={ext.sum()} | clf AUC(es)={clf.best_score['valid_0']['auc']:.4f} "
          f"| p_ext val: mean={p_ext.mean():.5f} max={p_ext.max():.3f}")
    results[tag] = report(tag, mix)

best_tag, best = min(results.items(), key=lambda kv: kv[1]["overall"])
log_experiment(f"M4 regime mixture ({best_tag})", model="LGBM clf + 2 LGBM reg",
               features="M3 features; mixture p*E[y|ext]+(1-p)*E[y|normal]",
               val_rmse=best["overall"], val_rmse_jan=best["jan"], val_rmse_jul=best["jul"],
               notes="calibrated p from weighted classifier", keep="")
print("logged:", best_tag)
