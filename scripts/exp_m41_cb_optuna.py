"""M41 (GPU): Optuna for CatBoost, never done - only LightGBM was ever tuned, yet
CatBoost carries 40-60% of the tree blend depending on the airport.

Same clean protocol as M23: fit before 1 Nov, Optuna objective on 1 Nov - 9 Dec,
early stopping on 10-31 Dec, mirror holdout untouched.

Usage: python scripts/exp_m41_cb_optuna.py <AIRPORT> [n_trials]
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import optuna
from catboost import CatBoostRegressor, Pool

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from dc26.validation import rmse

PROC = ROOT / "data" / "processed"
CAP = 86400
optuna.logging.set_verbosity(optuna.logging.WARNING)

AIRPORT = sys.argv[1]
N_TRIALS = int(sys.argv[2]) if len(sys.argv) > 2 else 15

X = pd.concat([pd.read_parquet(PROC / f) for f in
               ["X_v2_mirror_t.parquet", "flightctx_train2025.parquet",
                "depregime_train2025.parquet", "aobtq_train2025.parquet",
                "anomreg_train2025.parquet"]], axis=1)
meta = pd.read_parquet(PROC / "meta_train2025.parquet")
cat = [c for c in X.columns if X[c].dtype.name == "category"]
for c in cat:
    X[c] = X[c].cat.add_categories("__NA__").fillna("__NA__")
y = meta["y"].astype("float64")
tr, es = meta["tr"].to_numpy(), meta["es"].to_numpy()
ok = (y > 0).to_numpy() & (y < 200000).to_numpy()
art = np.zeros(len(y), bool)
for k in (1, 2, 3):
    art |= (y - k * 86400).abs().to_numpy() < 3600
ok &= ~art
apt = X["AIRPORT"].astype(str).to_numpy()
doy = X["doy"].to_numpy()

sel = ok & tr & (apt == AIRPORT)
fit = sel & (doy < 305) & ~es
tune = sel & (doy >= 305) & ~es
stop = sel & es
print("%s: fit=%d tune=%d stop=%d" % (AIRPORT, fit.sum(), tune.sum(), stop.sum()), flush=True)

yk = y.clip(upper=CAP)
pool_fit = Pool(X[fit], yk[fit], cat_features=cat)
pool_stop = Pool(X[stop], yk[stop], cat_features=cat)
X_tune, y_tune = X[tune], y[tune].to_numpy()


def score(model):
    return rmse(y_tune, np.clip(model.predict(X_tune), 60, CAP))


import torch
GPU = torch.cuda.is_available()
DEV = dict(task_type="GPU", devices="0") if GPU else dict(thread_count=-1)
print("CatBoost sur %s" % ("GPU" if GPU else "CPU"), flush=True)

base = CatBoostRegressor(loss_function="RMSE", iterations=1500, learning_rate=0.1, depth=7,
                         l2_leaf_reg=3.0, random_seed=42, od_type="Iter", od_wait=80,
                         verbose=False, max_ctr_complexity=1,
                         cat_features=cat, **DEV)
base.fit(pool_fit, eval_set=pool_stop)
base_score = score(base)
print("baseline tune-RMSE: %.2f (iters=%d)" % (base_score, base.get_best_iteration()), flush=True)


def objective(trial):
    p = dict(
        loss_function="RMSE", iterations=3000, random_seed=42, od_type="Iter", od_wait=80,
        verbose=False, cat_features=cat, **DEV,
        learning_rate=trial.suggest_float("learning_rate", 0.03, 0.2, log=True),
        depth=trial.suggest_int("depth", 5, 9),
        l2_leaf_reg=trial.suggest_float("l2_leaf_reg", 0.5, 30.0, log=True),
        bagging_temperature=trial.suggest_float("bagging_temperature", 0.0, 1.0),
        max_ctr_complexity=trial.suggest_int("max_ctr_complexity", 1, 2),
        border_count=trial.suggest_categorical("border_count", [64, 128, 254]),
    )
    m = CatBoostRegressor(**p)
    m.fit(pool_fit, eval_set=pool_stop)
    return score(m)


study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=42))
study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=False)
print("best tune-RMSE: %.2f (baseline %.2f, delta %+.2f)" %
      (study.best_value, base_score, study.best_value - base_score), flush=True)
print("best params:", study.best_params, flush=True)
pd.Series({**study.best_params, "_tune_rmse": study.best_value, "_baseline": base_score}).to_json(
    ROOT / "experiments" / ("optuna_cb_" + AIRPORT + ".json"))
print("sauvegarde", flush=True)
