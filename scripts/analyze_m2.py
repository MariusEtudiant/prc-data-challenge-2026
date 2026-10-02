"""Error analysis of M2 on the mirror holdout: where do we lose RMSE points?"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from dc26 import load_movements_2025, mirror_split
from dc26.state_features import get_state_features

sys.path.insert(0, str(ROOT / "scripts"))

mv = load_movements_2025()
dep = mv[mv["PHASE_mvt"] == "DEP"].copy()


def build_features_v0(df):
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
    for c in ["AIRPORT", "AIRCRAFT_TYPE_mvt", "WK_TBL_CAT_flt", "MARKET_SEGMENT_flt",
              "AIRCRAFT_OPERATOR_flt", "FLIGHT_RULE_mvt"]:
        X[c] = df[c].astype("category")
    X["RUNWAY"] = (df["AIRPORT"].astype(str) + "_" + df["RUNWAY_mvt"].astype(str)).astype("category")
    X["STAND"] = (df["AIRPORT"].astype(str) + "_" + df["STAND_mvt"].astype(str)).astype("category")
    X["ADES_REGION"] = df["ADES_mvt"].astype(str).str[:2].astype("category")
    return X


X_all = pd.concat([build_features_v0(dep), get_state_features("train2025", dep, mv)], axis=1)
cat_cols = [c for c in X_all.columns if X_all[c].dtype.name == "category"]

train, val, val_part = mirror_split(dep)
tr_mask = dep.index.isin(train.index)
clean = tr_mask & dep["TAXITIME_SEC_mvt"].between(30, 7200)
X_tr, y_tr = X_all[clean], dep.loc[clean, "TAXITIME_SEC_mvt"]
va_mask = dep.index.isin(val.index)
X_va, y_va = X_all[va_mask], dep.loc[va_mask, "TAXITIME_SEC_mvt"]

es_mask = (dep.loc[clean, "MVT_TIME_UTC_mvt"] >= "2025-12-10").to_numpy()
params = dict(objective="regression", metric="rmse", learning_rate=0.06,
              num_leaves=255, min_data_in_leaf=100, feature_fraction=0.9,
              bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0,
              num_threads=0, verbosity=-1, seed=42)
model = lgb.train(params, lgb.Dataset(X_tr[~es_mask], y_tr[~es_mask], categorical_feature=cat_cols),
                  num_boost_round=4000,
                  valid_sets=[lgb.Dataset(X_tr[es_mask], y_tr[es_mask])],
                  callbacks=[lgb.early_stopping(150, verbose=False)])
pred = model.predict(X_va, num_iteration=model.best_iteration)

res = val[["AIRPORT", "MVT_TIME_UTC_mvt", "TAXITIME_SEC_mvt", "AOBT_3_flt", "RUNWAY_mvt"]].copy()
res["part"] = val_part
res["pred"] = pred
res["err"] = res["pred"] - res["TAXITIME_SEC_mvt"]
res["sq"] = res["err"] ** 2
res["naive"] = (res["MVT_TIME_UTC_mvt"] - res["AOBT_3_flt"]).dt.total_seconds()
res.to_parquet(ROOT / "experiments" / "preds_M2_holdout.parquet", index=False)

total_sq = res["sq"].sum()
overall = np.sqrt(res["sq"].mean())
print(f"M2 holdout RMSE = {overall:.1f} s  (n={len(res):,})\n")

print("== 1) Error budget: share of total squared error by |err| percentile ==")
r = res.sort_values("sq", ascending=False).reset_index(drop=True)
for frac in [0.001, 0.01, 0.05, 0.10, 0.25]:
    k = int(len(r) * frac)
    print(f"  top {frac:>5.1%} of rows ({k:>6,}) -> {r['sq'][:k].sum()/total_sq:5.1%} of squared error"
          f" | |err| >= {np.sqrt(r['sq'][k]):.0f}s")

print("\n== 2) Squared-error share by airport x part ==")
g = res.groupby(["part", "AIRPORT"])["sq"].agg(["sum", "count", "mean"])
g["share"] = g["sum"] / total_sq
g["rmse"] = np.sqrt(g["mean"])
print(g.sort_values("share", ascending=False)[["count", "rmse", "share"]].head(12).round(3).to_string())

print("\n== 3) By true-target bucket ==")
buckets = pd.cut(res["TAXITIME_SEC_mvt"], [-np.inf, 600, 1200, 1800, 3600, 7200, np.inf],
                 labels=["<10m", "10-20m", "20-30m", "30-60m", "1-2h", ">2h"])
g3 = res.groupby(buckets)["sq"].agg(["count", "sum", "mean"])
g3["share"] = g3["sum"] / total_sq
g3["rmse"] = np.sqrt(g3["mean"])
print(g3[["count", "rmse", "share"]].round(3).to_string())

print("\n== 4) By AOBT_3 reliability (|naive - y|) ==")
rel = pd.cut((res["naive"] - res["TAXITIME_SEC_mvt"]).abs(),
             [-np.inf, 60, 300, 900, np.inf], labels=["<1m", "1-5m", "5-15m", ">15m/missing"])
g4 = res.groupby(rel, dropna=False)["sq"].agg(["count", "sum", "mean"])
g4["share"] = g4["sum"] / total_sq
g4["rmse"] = np.sqrt(g4["mean"])
print(g4[["count", "rmse", "share"]].round(3).to_string())

print("\n== 5) Worst 12 rows ==")
print(r.head(12)[["AIRPORT", "part", "TAXITIME_SEC_mvt", "pred", "naive", "err"]].round(0).to_string())
