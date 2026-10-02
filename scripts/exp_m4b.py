"""M4': properly calibrated regime mixture.
- classifier P(y>7200) without scale_pos_weight, isotonic-calibrated on the ES window
- final = (1-p)*E[y|normal] + p*E[y|ext], no artificial floor on E[y|ext]
Compared against M3b (single model with extremes, val 360.7).
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.isotonic import IsotonicRegression

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
THR = 7200

reg_params = dict(objective="regression", metric="rmse", learning_rate=0.06,
                  num_leaves=255, min_data_in_leaf=100, feature_fraction=0.9,
                  bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0,
                  num_threads=0, verbosity=-1, seed=42)

es_all = (dep["MVT_TIME_UTC_mvt"] >= "2025-12-10") & tr_mask

# --- classifier, unweighted, calibrated on ES window
lab = ((y_all > THR) & (y_all < 200000)).astype(int)
fit_m = tr_mask & ~es_all
clf_params = dict(objective="binary", metric="binary_logloss", learning_rate=0.05,
                  num_leaves=63, min_data_in_leaf=50, feature_fraction=0.9,
                  num_threads=0, verbosity=-1, seed=42)
clf = lgb.train(clf_params, lgb.Dataset(X_all[fit_m], lab[fit_m], categorical_feature=cat_cols),
                num_boost_round=2000,
                valid_sets=[lgb.Dataset(X_all[es_all], lab[es_all])],
                callbacks=[lgb.early_stopping(100, verbose=False)])
p_es = clf.predict(X_all[es_all], num_iteration=clf.best_iteration)
iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
iso.fit(p_es, lab[es_all])
p_va = iso.predict(clf.predict(X_va, num_iteration=clf.best_iteration))
print(f"clf iters={clf.best_iteration} | base rate train={lab[tr_mask].mean():.5f} "
      f"| p_va mean={p_va.mean():.5f} (val true rate={(y_va > THR).mean():.5f}) max={p_va.max():.3f}")

# --- normal regressor (same as M3a config)
normal = tr_mask & y_all.between(30, THR)
es_n = (dep.loc[normal, "MVT_TIME_UTC_mvt"] >= "2025-12-10").to_numpy()
mA = lgb.train(reg_params, lgb.Dataset(X_all[normal][~es_n], y_all[normal][~es_n], categorical_feature=cat_cols),
               num_boost_round=4000,
               valid_sets=[lgb.Dataset(X_all[normal][es_n], y_all[normal][es_n])],
               callbacks=[lgb.early_stopping(150, verbose=False)])
predA = mA.predict(X_va, num_iteration=mA.best_iteration)

# --- extreme regressor (small, on ~540 rows, capped)
ext = tr_mask & (y_all > THR) & (y_all < 200000)
extB_params = dict(objective="regression", learning_rate=0.05, num_leaves=15,
                   min_data_in_leaf=20, feature_fraction=0.8, num_threads=0, verbosity=-1, seed=42)
mB = lgb.train(extB_params, lgb.Dataset(X_all[ext], y_all[ext].clip(upper=86400),
                                        categorical_feature=cat_cols), num_boost_round=300)
predB = mB.predict(X_va)
print(f"ext n={ext.sum()} | predB stats: median={np.median(predB):.0f} q95={np.quantile(predB, .95):.0f}")

mix = (1 - p_va) * predA + p_va * predB
rep = rmse_report(y_va.to_numpy(), mix, val_part)
buckets = pd.cut(y_va, [-np.inf, 1800, THR, np.inf], labels=["<30m", "30m-2h", ">2h"])
det = {b: float(np.sqrt(np.mean((mix[(buckets == b).to_numpy()] - y_va[(buckets == b).to_numpy()]) ** 2)))
       for b in ["<30m", "30m-2h", ">2h"]}
print(f"M4' calibrated mixture: overall={rep['overall']:.1f} jan={rep['jan']:.1f} jul={rep['jul']:.1f} "
      f"| <30m={det['<30m']:.1f} 30m-2h={det['30m-2h']:.1f} >2h={det['>2h']:.1f}")

log_experiment("M4' calibrated regime mixture", model="LGBM clf(iso) + 2 LGBM reg",
               features="M3 features; thr 7200; isotonic-calibrated p",
               val_rmse=rep["overall"], val_rmse_jan=rep["jan"], val_rmse_jul=rep["jul"],
               notes=f"p mean {p_va.mean():.5f} vs true {(y_va > THR).mean():.5f}",
               keep="KEEP" if rep["overall"] < 360.7 else "DROP")
print("done")
