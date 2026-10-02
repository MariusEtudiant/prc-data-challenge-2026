"""M3: keep extreme targets in training (capped), add flight-plan-match flag.
Hypothesis: the >2h regime (70% of holdout squared error) is identifiable
(LIRF, missing AOBT_3/FPL, large MVT-SCHED) but was amputated from training.
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

mv = load_movements_2025()
dep = mv[mv["PHASE_mvt"] == "DEP"].copy()


def build_features_v1(df):
    X = pd.DataFrame(index=df.index)
    t = df["MVT_TIME_UTC_mvt"]
    X["hour"] = t.dt.hour + t.dt.minute / 60
    X["dow"] = t.dt.dayofweek.astype("int8")
    X["month"] = t.dt.month.astype("int8")
    X["doy"] = t.dt.dayofyear.astype("int16")
    for c, ts in [("aobt3", "AOBT_3_flt"), ("eobt1", "EOBT_1_flt"),
                  ("lobt", "LOBT_flt"), ("iobt", "IOBT_flt"), ("sched", "SCHED_TIME_UTC_mvt")]:
        X[f"d_{c}"] = (t - df[ts]).dt.total_seconds()
    X["d_aobt3_vs_eobt1"] = (df["AOBT_3_flt"] - df["EOBT_1_flt"]).dt.total_seconds()
    X["has_fpl"] = df["FLIGHT_ID_mvt"].notna().astype("int8")   # new in M3
    for c in ["AIRPORT", "AIRCRAFT_TYPE_mvt", "WK_TBL_CAT_flt", "MARKET_SEGMENT_flt",
              "AIRCRAFT_OPERATOR_flt", "FLIGHT_RULE_mvt"]:
        X[c] = df[c].astype("category")
    X["RUNWAY"] = (df["AIRPORT"].astype(str) + "_" + df["RUNWAY_mvt"].astype(str)).astype("category")
    X["STAND"] = (df["AIRPORT"].astype(str) + "_" + df["STAND_mvt"].astype(str)).astype("category")
    X["ADES_REGION"] = df["ADES_mvt"].astype(str).str[:2].astype("category")
    return X


X_all = pd.concat([build_features_v1(dep), get_state_features("train2025", dep, mv)], axis=1)
cat_cols = [c for c in X_all.columns if X_all[c].dtype.name == "category"]

train, val, val_part = mirror_split(dep)
tr_mask = dep.index.isin(train.index)
va_mask = dep.index.isin(val.index)
X_va, y_va = X_all[va_mask], dep.loc[va_mask, "TAXITIME_SEC_mvt"]

params = dict(objective="regression", metric="rmse", learning_rate=0.06,
              num_leaves=255, min_data_in_leaf=100, feature_fraction=0.9,
              bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0,
              num_threads=0, verbosity=-1, seed=42)


def run(tag, clean_mask, y_transform=None):
    y = dep.loc[clean_mask, "TAXITIME_SEC_mvt"]
    if y_transform:
        y = y.clip(upper=y_transform)
    es = (dep.loc[clean_mask, "MVT_TIME_UTC_mvt"] >= "2025-12-10").to_numpy()
    m = lgb.train(params, lgb.Dataset(X_all[clean_mask][~es], y[~es], categorical_feature=cat_cols),
                  num_boost_round=4000,
                  valid_sets=[lgb.Dataset(X_all[clean_mask][es], y[es])],
                  callbacks=[lgb.early_stopping(150, verbose=False)])
    p = m.predict(X_va, num_iteration=m.best_iteration)
    rep = rmse_report(y_va.to_numpy(), p, val_part)
    # bucket detail
    buckets = pd.cut(y_va, [-np.inf, 1800, 7200, np.inf], labels=["<30m", "30m-2h", ">2h"])
    det = {}
    for b in ["<30m", "30m-2h", ">2h"]:
        mm = (buckets == b).to_numpy()
        det[b] = float(np.sqrt(np.mean((p[mm] - y_va.to_numpy()[mm]) ** 2)))
    print(f"{tag}: overall={rep['overall']:.1f} jan={rep['jan']:.1f} jul={rep['jul']:.1f} "
          f"| <30m={det['<30m']:.1f} 30m-2h={det['30m-2h']:.1f} >2h={det['>2h']:.1f} | iter={m.best_iteration}")
    return rep, m, p


# A: reference = M2 cleaning (30..7200) but with has_fpl
repA, mA, pA = run("M3a (clean 30..7200 + has_fpl)", tr_mask & dep["TAXITIME_SEC_mvt"].between(30, 7200))
# B: keep extremes, cap at 24h
repB, mB, pB = run("M3b (keep extremes, cap 86400)", tr_mask & dep["TAXITIME_SEC_mvt"].between(1, 200000), 86400)
# C: keep extremes, cap at 4h (bounded influence)
repC, mC, pC = run("M3c (keep extremes, cap 14400)", tr_mask & dep["TAXITIME_SEC_mvt"].between(1, 200000), 14400)

best_tag, best_rep = min([("M3a", repA), ("M3b", repB), ("M3c", repC)], key=lambda x: x[1]["overall"])
log_experiment(f"M3 extremes-in-train ({best_tag})", model="LightGBM",
               features="M2 + has_fpl; train includes >2h targets",
               val_rmse=best_rep["overall"], val_rmse_jan=best_rep["jan"], val_rmse_jul=best_rep["jul"],
               notes="error-budget driven: 43 rows >2h were 70% of sq error", keep="KEEP")
print("logged:", best_tag)
