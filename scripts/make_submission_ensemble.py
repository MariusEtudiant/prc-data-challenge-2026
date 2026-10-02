"""Final submission from the LGBM+CatBoost ensemble (XGB gets weight 0).

Usage: python scripts/make_submission_ensemble.py <version> [--turnaround]
Trains both models on ALL of 2025 with the mirror-derived round counts, blends with
the mirror-derived weights, writes submissions/reliable-hamburger_v<version>.parquet.
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

PROC = ROOT / "data" / "processed"
CAP = 86400
BOOST = 1.12

version = sys.argv[1]
turn = "--turnaround" in sys.argv
suffix = "_t" if turn else ""

X = pd.read_parquet(PROC / f"X_v2_full{suffix}.parquet")
Xr = pd.read_parquet(PROC / f"X_v2_ranking{suffix}.parquet")
meta = pd.read_parquet(PROC / "meta_train2025.parquet")
d = np.load(ROOT / "experiments" / f"m13_preds{suffix}.npz")
w = np.load(ROOT / "experiments" / f"m13_weights{suffix}.npy")
y = meta["y"].astype("float64")
keep = (y > 0).to_numpy() & (y < 200000).to_numpy()
yk = y[keep].clip(upper=CAP)
cat = [c for c in X.columns if X[c].dtype.name == "category"]
print(f"features={X.shape[1]} weights lgbm/cb={w[0]:.2f}/{w[1]:.2f}", flush=True)

n = int(d["lgbm_iters"] * BOOST)
ml = lgb.train(REG_PARAMS, lgb.Dataset(X[keep], yk, categorical_feature=cat), num_boost_round=n)
pl = ml.predict(Xr)
ml.save_model(str(ROOT / "models" / f"v{version}_lgbm.txt"))
print(f"lgbm done ({n} rounds)", flush=True)

for c in cat:
    X[c] = X[c].cat.add_categories("__NA__").fillna("__NA__")
    Xr[c] = Xr[c].cat.add_categories("__NA__").fillna("__NA__")
gc.collect()
n = int(d["cb_iters"] * BOOST)
mc = CatBoostRegressor(loss_function="RMSE", iterations=n, learning_rate=0.1, depth=7,
                       l2_leaf_reg=3.0, random_seed=42, verbose=300, thread_count=-1,
                       max_ctr_complexity=1, cat_features=cat)
mc.fit(Pool(X[keep], yk, cat_features=cat))
pc = mc.predict(Xr)
mc.save_model(str(ROOT / "models" / f"v{version}_catboost.cbm"))
print(f"catboost done ({n} rounds)", flush=True)

tot = w[0] + w[1]
pred = ((w[0] * pl + w[1] * pc) / tot).clip(60, CAP)
ids = pd.read_parquet(PROC / f"rank_ids{suffix}.parquet")["MVT_ID_mvt"]
tpl = pd.read_parquet(ROOT / "data" / "submitting.parquet").drop(columns=["TAXITIME_SEC_mvt"])
out = tpl.merge(pd.DataFrame({"MVT_ID_mvt": ids.values, "TAXITIME_SEC_mvt": pred}),
                on="MVT_ID_mvt", how="left")
assert out["TAXITIME_SEC_mvt"].notna().all() and len(out) == 215876
path = ROOT / "submissions" / f"reliable-hamburger_v{version}.parquet"
out.to_parquet(path, index=False)
print(f"v{version} written: mean={out['TAXITIME_SEC_mvt'].mean():.0f}s "
      f"median={out['TAXITIME_SEC_mvt'].median():.0f}s "
      f"q99={out['TAXITIME_SEC_mvt'].quantile(.99):.0f}s", flush=True)
