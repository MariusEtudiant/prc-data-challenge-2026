"""M20: flight-specific context (plan-revision chain + previously unused columns).
Evaluated per-airport, since M18 showed that is the right model structure."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from dc26 import load_movements_2025, log_experiment
from dc26.validation import rmse_report, rmse
from dc26.mixture import REG_PARAMS
from dc26.flight_context import build_flight_context

PROC = ROOT / "data" / "processed"
CAP = 86400
MIN_TRAIN = 20000
X = pd.read_parquet(PROC / "X_v2_mirror_t.parquet")
meta = pd.read_parquet(PROC / "meta_train2025.parquet")
y = meta["y"].astype("float64")
tr, va, es = meta["tr"].to_numpy(), meta["va"].to_numpy(), meta["es"].to_numpy()
vp = meta.loc[va, "val_part"].to_numpy()
keep = tr & (y > 0).to_numpy() & (y < 200000).to_numpy()
y_va = y[va].to_numpy()

mv = load_movements_2025()
dep = mv[mv["PHASE_mvt"] == "DEP"].reset_index(drop=True)
Xc = build_flight_context(dep)
Xc.to_parquet(PROC / "flightctx_train2025.parquet", index=False)
num = [c for c in Xc.columns if Xc[c].dtype.name != "category"]
print("flight-context corr with target:",
      {c: round(float(Xc[c].corr(y)), 3) for c in num}, flush=True)

X2 = pd.concat([X, Xc], axis=1)
apt = X["AIRPORT"].astype(str).to_numpy()
apt_va = apt[va]


def per_airport(XX, tag):
    cat = [c for c in XX.columns if XX[c].dtype.name == "category"]
    pred = np.full(va.sum(), np.nan)
    for a in np.unique(apt):
        fit = keep & (apt == a) & ~es
        ev = keep & (apt == a) & es
        sel = apt_va == a
        if fit.sum() < MIN_TRAIN or not sel.any():
            continue
        m = lgb.train(REG_PARAMS, lgb.Dataset(XX[fit], y[fit].clip(upper=CAP), categorical_feature=cat),
                      num_boost_round=6000,
                      valid_sets=[lgb.Dataset(XX[ev], y[ev].clip(upper=CAP))],
                      callbacks=[lgb.early_stopping(150, verbose=False)])
        pred[sel] = m.predict(XX[va][sel], num_iteration=m.best_iteration)
    rep = rmse_report(y_va, np.clip(pred, 60, CAP), vp)
    print(f"{tag}: overall={rep['overall']:.1f} jan={rep['jan']:.1f} jul={rep['jul']:.1f}", flush=True)
    return rep, pred


repA, pA = per_airport(X, "M20a per-airport baseline")
repB, pB = per_airport(X2, "M20b per-airport + flight context")
for a in np.unique(apt_va):
    s = apt_va == a
    print(f"   [{a}] {rmse(y_va[s], np.clip(pA[s],60,CAP)):8.1f} -> "
          f"{rmse(y_va[s], np.clip(pB[s],60,CAP)):8.1f}", flush=True)
log_experiment("M20 flight context (plan revisions + unused cols)", model="per-airport LGBM",
               features="v2+turnaround+flight_context",
               val_rmse=repB["overall"], val_rmse_jan=repB["jan"], val_rmse_jul=repB["jul"],
               notes=f"vs per-airport baseline {repA['overall']:.1f}",
               keep="KEEP" if repB["overall"] < repA["overall"] else "DROP")
print("done", flush=True)
