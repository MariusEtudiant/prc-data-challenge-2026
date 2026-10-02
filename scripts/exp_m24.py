"""M24: departure-side regime sensor.

Earlier the surface probe used arrival taxi-in — the wrong sensor, since de-icing,
ATFM slot holding and runway queueing hit departures only. Here the sensor is the
recent taxi-out of other departures, reconstructed as MVT_TIME - AOBT_3 (both
columns given for every departure in the ranking file), self excluded.

Evaluated per airport, the structure M18 established.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from dc26 import load_movements_2025, log_experiment
from dc26.validation import rmse, rmse_report
from dc26.mixture import REG_PARAMS
from dc26.dep_regime import build_dep_regime

PROC = ROOT / "data" / "processed"
CAP = 86400
X = pd.concat([pd.read_parquet(PROC / "X_v2_mirror_t.parquet"),
               pd.read_parquet(PROC / "flightctx_train2025.parquet")], axis=1)
meta = pd.read_parquet(PROC / "meta_train2025.parquet")
y = meta["y"].astype("float64")
tr, va, es = meta["tr"].to_numpy(), meta["va"].to_numpy(), meta["es"].to_numpy()
vp = meta.loc[va, "val_part"].to_numpy()
keep = tr & (y > 0).to_numpy() & (y < 200000).to_numpy()
y_va = y[va].to_numpy()

mv = load_movements_2025()
dep = mv[mv["PHASE_mvt"] == "DEP"].reset_index(drop=True)
Xr = build_dep_regime(dep, mv)
Xr.to_parquet(PROC / "depregime_train2025.parquet", index=False)
print("dep-regime corr:", {c: round(float(Xr[c].corr(y)), 3) for c in Xr.columns}, flush=True)

apt = X["AIRPORT"].astype(str).to_numpy()
apt_va = apt[va]


def per_airport(XX, tag):
    cat = [c for c in XX.columns if XX[c].dtype.name == "category"]
    pred = np.full(va.sum(), np.nan)
    for a in np.unique(apt):
        fit, ev = keep & (apt == a) & ~es, keep & (apt == a) & es
        sel = apt_va == a
        if fit.sum() < 20000 or not sel.any():
            continue
        m = lgb.train(REG_PARAMS, lgb.Dataset(XX[fit], y[fit].clip(upper=CAP), categorical_feature=cat),
                      num_boost_round=6000, valid_sets=[lgb.Dataset(XX[ev], y[ev].clip(upper=CAP))],
                      callbacks=[lgb.early_stopping(150, verbose=False)])
        pred[sel] = m.predict(XX[va][sel], num_iteration=m.best_iteration)
    pred = np.clip(pred, 60, CAP)
    rep = rmse_report(y_va, pred, vp)
    print(f"{tag}: overall={rep['overall']:.1f} jan={rep['jan']:.1f} jul={rep['jul']:.1f}", flush=True)
    return rep, pred


repA, pA = per_airport(X, "M24a baseline")
repB, pB = per_airport(pd.concat([X, Xr], axis=1), "M24b + departure regime")
for a in np.unique(apt_va):
    s = apt_va == a
    print(f"   [{a}] {rmse(y_va[s], pA[s]):8.1f} -> {rmse(y_va[s], pB[s]):8.1f}", flush=True)
log_experiment("M24 departure-regime sensor", model="per-airport LGBM",
               features="v2+turnaround+flightctx+dep_regime (proxy taxi of other deps)",
               val_rmse=repB["overall"], val_rmse_jan=repB["jan"], val_rmse_jul=repB["jul"],
               notes=f"vs baseline {repA['overall']:.1f}",
               keep="KEEP" if repB["overall"] < repA["overall"] else "DROP")
print("done", flush=True)
