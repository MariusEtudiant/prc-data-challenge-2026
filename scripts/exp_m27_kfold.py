"""M27: proper K-fold OOF instead of a single 21-day early-stopping window.

Our whole model selection has rested on 10-31 Dec 2025 (2% of the data, one season).
Here: 5 folds over month blocks, each fold early-stopped on its own held-out block,
producing out-of-fold predictions over the entire training year. Benefits:
  - iteration counts averaged over 5 seasons instead of one December,
  - OOF predictions on ~1.9M rows to fit ensemble weights (instead of ~110k),
  - an unbiased estimate of training-year performance.

Evaluated per airport (the M18 structure) with the tuned parameters.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from dc26 import log_experiment
from dc26.validation import rmse, rmse_report
from dc26.mixture import REG_PARAMS

PROC = ROOT / "data" / "processed"
CAP = 86400
MIN_TRAIN = 20000
FOLDS = [(1, 2, 3), (4, 5), (6, 7), (8, 9), (10, 11, 12)]

X = pd.concat([pd.read_parquet(PROC / "X_v2_mirror_t.parquet"),
               pd.read_parquet(PROC / "flightctx_train2025.parquet"),
               pd.read_parquet(PROC / "depregime_train2025.parquet")], axis=1)
meta = pd.read_parquet(PROC / "meta_train2025.parquet")
cat = [c for c in X.columns if X[c].dtype.name == "category"]
y = meta["y"].astype("float64")
tr, va = meta["tr"].to_numpy(), meta["va"].to_numpy()
vp = meta.loc[va, "val_part"].to_numpy()
keep = tr & (y > 0).to_numpy() & (y < 200000).to_numpy()
y_va = y[va].to_numpy()
apt = X["AIRPORT"].astype(str).to_numpy()
apt_va = apt[va]
month = X["month"].to_numpy()


def tuned_params(a):
    p = ROOT / "experiments" / ("optuna_" + a + ".json")
    base = dict(objective="regression", metric="rmse", verbosity=-1,
                num_threads=0, bagging_freq=1, seed=42)
    if not p.exists():
        return dict(REG_PARAMS)
    d = json.loads(p.read_text())
    base.update({k: v for k, v in d.items() if not k.startswith("_")})
    for k in ("num_leaves", "min_data_in_leaf", "max_bin"):
        base[k] = int(base[k])
    return base


oof = np.full(len(X), np.nan)
pred_va = np.zeros(va.sum())
iters_all = {}
for a in np.unique(apt):
    sel_tr = keep & (apt == a)
    sel_va = apt_va == a
    if sel_tr.sum() < MIN_TRAIN or not sel_va.any():
        continue
    params = tuned_params(a)
    fold_preds, iters = [], []
    for months in FOLDS:
        hold = sel_tr & np.isin(month, months)
        fit = sel_tr & ~np.isin(month, months)
        if hold.sum() < 500 or fit.sum() < 5000:
            continue
        m = lgb.train(params, lgb.Dataset(X[fit], y[fit].clip(upper=CAP), categorical_feature=cat),
                      num_boost_round=6000,
                      valid_sets=[lgb.Dataset(X[hold], y[hold].clip(upper=CAP))],
                      callbacks=[lgb.early_stopping(150, verbose=False)])
        oof[hold] = m.predict(X[hold], num_iteration=m.best_iteration)
        fold_preds.append(m.predict(X[va][sel_va], num_iteration=m.best_iteration))
        iters.append(m.best_iteration)
    pred_va[sel_va] = np.mean(fold_preds, axis=0)
    iters_all[a] = iters
    oof_rmse = rmse(y[sel_tr & np.isfinite(oof)], oof[sel_tr & np.isfinite(oof)])
    print("[%s] folds=%d iters=%s OOF-RMSE=%.1f mirror=%.1f" % (
        a, len(iters), iters, oof_rmse,
        rmse(y_va[sel_va], np.clip(pred_va[sel_va], 60, CAP))), flush=True)

np.save(ROOT / "experiments" / "m27_oof.npy", oof)
np.save(ROOT / "experiments" / "m27_pred_va.npy", pred_va)
(ROOT / "experiments" / "m27_iters.json").write_text(json.dumps(iters_all))
rep = rmse_report(y_va, np.clip(pred_va, 60, CAP), vp)
print("M27 K-fold bagged per-airport: overall=%.1f jan=%.1f jul=%.1f | predicted LB ~%.0f" % (
    rep["overall"], rep["jan"], rep["jul"], rep["overall"] * 0.867), flush=True)
log_experiment("M27 K-fold OOF per-airport", model="5-fold bagged tuned LGBM",
               features="v2+turnaround+flightctx+depregime",
               val_rmse=rep["overall"], val_rmse_jan=rep["jan"], val_rmse_jul=rep["jul"],
               notes="replaces the single 21-day ES window", keep="KEEP")
print("done", flush=True)
