"""M22: seed averaging as variance reduction.

Four flights drive ~90% of the squared error, and predictions for that segment are
the least stable part of the model. Averaging several seeds should cut exactly the
variance that hurts most — a low-risk gain that hyperparameter tuning cannot give.

Per-airport LGBM, S seeds with different bagging/feature subsampling, averaged.
"""
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
SEEDS = [42, 1337, 2024, 7, 99]
X = pd.concat([pd.read_parquet(PROC / "X_v2_mirror_t.parquet"),
               pd.read_parquet(PROC / "flightctx_train2025.parquet")], axis=1)
meta = pd.read_parquet(PROC / "meta_train2025.parquet")
cat = [c for c in X.columns if X[c].dtype.name == "category"]
y = meta["y"].astype("float64")
tr, va, es = meta["tr"].to_numpy(), meta["va"].to_numpy(), meta["es"].to_numpy()
vp = meta.loc[va, "val_part"].to_numpy()
keep = tr & (y > 0).to_numpy() & (y < 200000).to_numpy()
y_va = y[va].to_numpy()
apt = X["AIRPORT"].astype(str).to_numpy()
apt_va = apt[va]

preds = np.full((len(SEEDS), va.sum()), np.nan)
for si, seed in enumerate(SEEDS):
    params = dict(REG_PARAMS, seed=seed, bagging_seed=seed + 1,
                  feature_fraction_seed=seed + 2, bagging_fraction=0.75,
                  feature_fraction=0.85)
    for a in np.unique(apt):
        fit, ev = keep & (apt == a) & ~es, keep & (apt == a) & es
        sel = apt_va == a
        if fit.sum() < 20000 or not sel.any():
            continue
        m = lgb.train(params, lgb.Dataset(X[fit], y[fit].clip(upper=CAP), categorical_feature=cat),
                      num_boost_round=6000, valid_sets=[lgb.Dataset(X[ev], y[ev].clip(upper=CAP))],
                      callbacks=[lgb.early_stopping(150, verbose=False)])
        preds[si, sel] = m.predict(X[va][sel], num_iteration=m.best_iteration)
    r = rmse_report(y_va, np.clip(preds[si], 60, CAP), vp)
    avg = np.clip(np.nanmean(preds[:si + 1], axis=0), 60, CAP)
    ra = rmse_report(y_va, avg, vp)
    print(f"seed {seed}: single={r['overall']:.1f} | average of {si+1} seeds={ra['overall']:.1f} "
          f"(jan {ra['jan']:.1f}, jul {ra['jul']:.1f})", flush=True)

avg = np.clip(np.nanmean(preds, axis=0), 60, CAP)
rep = rmse_report(y_va, avg, vp)
np.save(ROOT / "experiments" / "m22_seedavg_va.npy", avg)

# where does the averaging help?
single = np.clip(preds[0], 60, CAP)
for a in np.unique(apt_va):
    s = apt_va == a
    print(f"   [{a}] single={rmse(y_va[s], single[s]):8.1f} -> avg={rmse(y_va[s], avg[s]):8.1f}", flush=True)

log_experiment("M22 seed-averaged per-airport", model="5-seed LGBM per airport",
               features="v2+turnaround+flight_context", val_rmse=rep["overall"],
               val_rmse_jan=rep["jan"], val_rmse_jul=rep["jul"],
               notes="variance reduction on the extreme segment", keep="KEEP")
print("done", flush=True)
