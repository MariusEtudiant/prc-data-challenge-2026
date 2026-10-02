"""M45: let the search pick the models instead of us.

Every model in the pipeline so far was chosen by hand - LightGBM by reflex, CatBoost
because the 2025 winners used it, an MLP because I reasoned about inductive bias.
We never asked which families actually suit these data. AutoGluon placed 2nd in the
2024 edition of this very challenge doing exactly that.

It explores families we never tried (random forests, ExtraTrees, k-NN, regularised
linear models) and builds a multi-layer stack on out-of-fold predictions, where we
only ever did a weighted average.

Run per airport - that is our architecture anyway, and 130-270k rows per airport is
comfortable. Runs in the isolated venv (.venv_automl) so the main pipeline is safe.

Usage: python scripts/exp_m45_automl.py <AIRPORT> [minutes]
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from autogluon.tabular import TabularPredictor

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"
OUT = ROOT / "experiments" / "automl"
OUT.mkdir(exist_ok=True)
CAP = 86400

GPU = "--gpu" in sys.argv
AIRPORT = sys.argv[1]
MINUTES = int(sys.argv[2]) if len(sys.argv) > 2 else 25

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
apt = X["AIRPORT"].astype(str).to_numpy()

fit = ok & tr & (apt == AIRPORT) & ~es
stop = ok & tr & (apt == AIRPORT) & es
hold = va & (apt == AIRPORT)
print("%s: entrainement=%d validation=%d holdout=%d" % (
    AIRPORT, fit.sum(), stop.sum(), hold.sum()), flush=True)

# AutoGluon wants plain dtypes; keep categoricals as strings
def prep(df):
    d = df.copy()
    for c in d.columns:
        if d[c].dtype.name == "category":
            d[c] = d[c].astype(str)
    return d

train_df = prep(X[fit])
train_df["taxi"] = y[fit].clip(upper=CAP).to_numpy()
tune_df = prep(X[stop])
tune_df["taxi"] = y[stop].clip(upper=CAP).to_numpy()
hold_df = prep(X[hold])
y_hold = y[hold].to_numpy()
clean_hold = ~art[hold]

t0 = time.time()
pred = TabularPredictor(label="taxi", eval_metric="root_mean_squared_error",
                        path=str(OUT / ("ag_" + AIRPORT)), verbosity=2)
print("GPU active:", GPU, flush=True)
pred.fit(train_data=train_df, tuning_data=tune_df, use_bag_holdout=True,
         time_limit=MINUTES * 60, presets="best_quality", num_stack_levels=1,
         ag_args_fit={"num_gpus": 1} if GPU else None)

lb = pred.leaderboard(silent=True)
print("\n=== modeles retenus par la recherche (%s) ===" % AIRPORT, flush=True)
print(lb[["model", "score_val", "fit_time", "pred_time_val"]].head(15).to_string(index=False), flush=True)

p = np.clip(pred.predict(hold_df).to_numpy(), 60, CAP)
rmse_clean = float(np.sqrt(np.mean((p[clean_hold] - y_hold[clean_hold]) ** 2)))
print("\n%s AutoML sur le holdout NETTOYE: %.1f  (%.0f min)" % (
    AIRPORT, rmse_clean, (time.time() - t0) / 60), flush=True)
np.save(OUT / ("pred_" + AIRPORT + ".npy"), p)
lb.to_csv(OUT / ("leaderboard_" + AIRPORT + ".csv"), index=False)
print("sauvegarde dans %s" % OUT, flush=True)
