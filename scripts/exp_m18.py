"""M18: two untested structural levers.

  B  covariate-shift weighting: the leaderboard is January + July only, yet we fit
     every month with equal weight. Up-weight the months adjacent to the test
     season (mirror-safe: Jan and Jul-hubs are the holdout, so we can only weight
     Feb/Nov/Dec and Jun/Aug).
  C  per-airport models: the irreducible noise ranges from 171s (LSZH) to 411s
     (LIRF); one global tree structure may be a poor compromise.
  D  per-airport for the three hubs (40% of the leaderboard) + global elsewhere.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from dc26 import log_experiment
from dc26.validation import rmse_report, rmse
from dc26.mixture import REG_PARAMS

PROC = ROOT / "data" / "processed"
CAP = 86400
X = pd.read_parquet(PROC / "X_v2_mirror_t.parquet")
meta = pd.read_parquet(PROC / "meta_train2025.parquet")
cat = [c for c in X.columns if X[c].dtype.name == "category"]
y = meta["y"].astype("float64")
tr, va, es = meta["tr"].to_numpy(), meta["va"].to_numpy(), meta["es"].to_numpy()
vp = meta.loc[va, "val_part"].to_numpy()
keep = tr & (y > 0).to_numpy() & (y < 200000).to_numpy()
yk, esk = y[keep].clip(upper=CAP), es[keep]
y_va = y[va].to_numpy()
month = X["month"].to_numpy()
apt = X["AIRPORT"].astype(str).to_numpy()


def fit_predict(mask_fit, X_pred, w=None, params=None):
    m_fit = mask_fit & ~es
    m_es = mask_fit & es
    dtr = lgb.Dataset(X[m_fit], y[m_fit].clip(upper=CAP), categorical_feature=cat,
                      weight=None if w is None else w[m_fit])
    des = lgb.Dataset(X[m_es], y[m_es].clip(upper=CAP), reference=dtr)
    m = lgb.train(params or REG_PARAMS, dtr, num_boost_round=6000, valid_sets=[des],
                  callbacks=[lgb.early_stopping(150, verbose=False)])
    return m.predict(X_pred, num_iteration=m.best_iteration), m.best_iteration


def report(tag, pred):
    rep = rmse_report(y_va, np.clip(pred, 60, CAP), vp)
    print(f"{tag}: overall={rep['overall']:.1f} jan={rep['jan']:.1f} jul={rep['jul']:.1f}", flush=True)
    return rep


# A - baseline
pA, itA = fit_predict(keep, X[va])
repA = report(f"M18a baseline (iters={itA})", pA)

# B - seasonal weighting
for boost in [2.0, 3.0]:
    w = np.ones(len(X))
    w[np.isin(month, [11, 12, 2])] = boost     # winter neighbours of January
    w[np.isin(month, [6, 8])] = boost          # summer neighbours of July
    pB, _ = fit_predict(keep, X[va], w=w)
    report(f"M18b seasonal weight x{boost:.0f}", pB)

# C / D - per-airport models
pred_air = np.full(va.sum(), np.nan)
va_apt = apt[va]
iters = {}
for a in np.unique(apt):
    m_keep = keep & (apt == a)
    sel = va_apt == a
    if sel.sum() == 0 or m_keep.sum() < 20000:
        continue
    p, it = fit_predict(m_keep, X[va][sel])
    pred_air[sel] = p
    iters[a] = it
    print(f"   [{a}] n_train={m_keep.sum():,} iters={it} "
          f"rmse={rmse(y_va[sel], np.clip(p, 60, CAP)):.1f} (global {rmse(y_va[sel], np.clip(pA[sel], 60, CAP)):.1f})",
          flush=True)
miss = np.isnan(pred_air)
pred_air[miss] = pA[miss]
repC = report("M18c per-airport models", pred_air)

hubs = ["EDDF", "EGLL", "EHAM"]
pD = np.where(np.isin(va_apt, hubs), pred_air, pA)
repD = report("M18d hubs specialised + global elsewhere", pD)

pE = 0.5 * pred_air + 0.5 * pA
repE = report("M18e average(global, per-airport)", pE)

best = min([("A", repA), ("C", repC), ("D", repD), ("E", repE)], key=lambda kv: kv[1]["overall"])
log_experiment(f"M18 shift-weighting & per-airport (best {best[0]})", model="LGBM variants",
               features="v2+turnaround", val_rmse=best[1]["overall"],
               val_rmse_jan=best[1]["jan"], val_rmse_jul=best[1]["jul"],
               notes=f"baseline {repA['overall']:.1f}", keep="")
print("done", flush=True)
