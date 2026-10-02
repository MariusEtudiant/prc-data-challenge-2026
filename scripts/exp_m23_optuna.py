"""M23: Optuna hyperparameter search, per airport, for the airports that carry the
error (LFPG ~49%, LIRF ~16%, EGLL, LTFM), plus a shared config for the rest.

Clean protocol — three disjoint time slices inside the mirror-train months:
  fit    : everything before 2025-11-01
  tune   : 2025-11-01 .. 2025-12-09   (Optuna objective)
  stop   : 2025-12-10 .. 2025-12-31   (early stopping)
The mirror holdout (Jan + Jul hubs) is never touched during the search.

Usage: python scripts/exp_m23_optuna.py <AIRPORT> [n_trials]
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb
import optuna

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from dc26.validation import rmse
from dc26.mixture import REG_PARAMS

PROC = ROOT / "data" / "processed"
CAP = 86400
optuna.logging.set_verbosity(optuna.logging.WARNING)

AIRPORT = sys.argv[1]
N_TRIALS = int(sys.argv[2]) if len(sys.argv) > 2 else 25

X = pd.concat([pd.read_parquet(PROC / "X_v2_mirror_t.parquet"),
               pd.read_parquet(PROC / "flightctx_train2025.parquet"),
               pd.read_parquet(PROC / "depregime_train2025.parquet")], axis=1)
meta = pd.read_parquet(PROC / "meta_train2025.parquet")
cat = [c for c in X.columns if X[c].dtype.name == "category"]
y = meta["y"].astype("float64")
tr, va, es = meta["tr"].to_numpy(), meta["va"].to_numpy(), meta["es"].to_numpy()
keep = tr & (y > 0).to_numpy() & (y < 200000).to_numpy()
apt = X["AIRPORT"].astype(str).to_numpy()
doy = X["doy"].to_numpy()
month = X["month"].to_numpy()

sel = keep & (apt == AIRPORT)
# Tune window kept as the ORIGINAL 1 Nov - 9 Dec. On 2026-09-05 it was changed to
# August + November to match the test's seasonal mix, and that made things WORSE
# (LIRF holdout 688.81 -> 703.40, below even the defaults at 701.29): carving August
# out of the tune window removes it from `fit`, and August is the only summer month
# we have, i.e. the most valuable one for predicting July. Reverted.
fit = sel & (doy < 305) & ~es          # before 1 Nov
tune = sel & (doy >= 305) & ~es        # 1 Nov .. 9 Dec
stop = sel & es                        # 10-31 Dec
print(f"{AIRPORT}: fit={fit.sum():,} tune={tune.sum():,} stop={stop.sum():,}", flush=True)

yk = y.clip(upper=CAP)
dtr = lgb.Dataset(X[fit], yk[fit], categorical_feature=cat, free_raw_data=False,
                  params={"feature_pre_filter": False})
dstop = lgb.Dataset(X[stop], yk[stop], reference=dtr, free_raw_data=False)
X_tune, y_tune = X[tune], y[tune].to_numpy()


def objective(trial):
    p = dict(
        objective="regression", metric="rmse", verbosity=-1, feature_pre_filter=False, num_threads=0,
        seed=42,
        learning_rate=trial.suggest_float("learning_rate", 0.02, 0.12, log=True),
        num_leaves=trial.suggest_int("num_leaves", 31, 511, log=True),
        min_data_in_leaf=trial.suggest_int("min_data_in_leaf", 20, 400, log=True),
        feature_fraction=trial.suggest_float("feature_fraction", 0.5, 1.0),
        bagging_fraction=trial.suggest_float("bagging_fraction", 0.5, 1.0),
        bagging_freq=1,
        lambda_l1=trial.suggest_float("lambda_l1", 1e-3, 10.0, log=True),
        lambda_l2=trial.suggest_float("lambda_l2", 1e-3, 20.0, log=True),
        min_gain_to_split=trial.suggest_float("min_gain_to_split", 0.0, 1.0),
        max_bin=trial.suggest_categorical("max_bin", [127, 255, 511]),
    )
    m = lgb.train(p, dtr, num_boost_round=4000, valid_sets=[dstop],
                  callbacks=[lgb.early_stopping(120, verbose=False)])
    return rmse(y_tune, np.clip(m.predict(X_tune, num_iteration=m.best_iteration), 60, CAP))


base = lgb.train(REG_PARAMS, dtr, num_boost_round=6000, valid_sets=[dstop],
                 callbacks=[lgb.early_stopping(150, verbose=False)])
base_score = rmse(y_tune, np.clip(base.predict(X_tune, num_iteration=base.best_iteration), 60, CAP))
print(f"baseline tune-RMSE: {base_score:.2f} (iters={base.best_iteration})", flush=True)

study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=42))
study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=False)
print(f"best tune-RMSE: {study.best_value:.2f} (baseline {base_score:.2f}, "
      f"delta {study.best_value - base_score:+.2f})", flush=True)
print("best params:", study.best_params, flush=True)

out = ROOT / "experiments" / f"optuna_{AIRPORT}.json"
pd.Series({**study.best_params, "_tune_rmse": study.best_value,
           "_baseline": base_score}).to_json(out)
print(f"saved {out.name}", flush=True)
