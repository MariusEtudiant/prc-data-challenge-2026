"""M21: is the model's differentiation INSIDE the high-risk segment real signal?

The unmatched-flight segment at LIRF carries ~30% probability of a >2h taxi and
dominates the whole metric. The model predicts very spread-out values there
(median 4.9ks, max 50ks). If that spread is noise rather than signal, shrinking
towards the segment's RMSE-optimal constant is worth a large amount of RMSE.

Test on the mirror holdout: model predictions vs shrinkage towards the constant.
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
X = pd.read_parquet(PROC / "X_v2_mirror_t.parquet")
meta = pd.read_parquet(PROC / "meta_train2025.parquet")
cat = [c for c in X.columns if X[c].dtype.name == "category"]
y = meta["y"].astype("float64")
tr, va, es = meta["tr"].to_numpy(), meta["va"].to_numpy(), meta["es"].to_numpy()
vp = meta.loc[va, "val_part"].to_numpy()
keep = tr & (y > 0).to_numpy() & (y < 200000).to_numpy()
y_va = y[va].to_numpy()
apt = X["AIRPORT"].astype(str).to_numpy()
nofpl = (X["has_fpl"].to_numpy() == 0)

# per-airport predictions on the holdout (same recipe as M18c/M19)
pred = np.full(va.sum(), np.nan)
for a in np.unique(apt):
    fit, ev = keep & (apt == a) & ~es, keep & (apt == a) & es
    sel = apt[va] == a
    if fit.sum() < 20000 or not sel.any():
        continue
    m = lgb.train(REG_PARAMS, lgb.Dataset(X[fit], y[fit].clip(upper=CAP), categorical_feature=cat),
                  num_boost_round=6000, valid_sets=[lgb.Dataset(X[ev], y[ev].clip(upper=CAP))],
                  callbacks=[lgb.early_stopping(150, verbose=False)])
    pred[sel] = m.predict(X[va][sel], num_iteration=m.best_iteration)
pred = np.clip(pred, 60, CAP)
print(f"per-airport baseline mirror: {rmse_report(y_va, pred, vp)['overall']:.1f}", flush=True)

apt_va, nofpl_va = apt[va], nofpl[va]
THR = 7200

for a in ["LIRF", "LFPG", "EGLL", "LTFM", "LSZH"]:
    seg_tr = keep & (apt == a) & nofpl
    seg_va = (apt_va == a) & nofpl_va
    if seg_va.sum() < 20 or seg_tr.sum() < 100:
        continue
    ys = y[seg_tr]
    P = float((ys > THR).mean())
    const = float(P * ys[ys > THR].mean() + (1 - P) * ys[ys <= THR].mean()) if P > 0 else float(ys.mean())
    yv, pv = y_va[seg_va], pred[seg_va]
    print(f"\n=== {a} unmatched: train n={seg_tr.sum()} P(ext)={P:.3f} const={const:.0f}s "
          f"| holdout n={seg_va.sum()} (ext {int((yv > THR).sum())})", flush=True)
    print(f"    model pred: median={np.median(pv):.0f} mean={pv.mean():.0f} max={pv.max():.0f}")
    best = (None, 1e18)
    for alpha in [0, 0.25, 0.5, 0.75, 1.0]:
        p = alpha * pv + (1 - alpha) * const
        r = rmse(yv, p)
        tag = "  <-- model" if alpha == 1 else ("  <-- constant" if alpha == 0 else "")
        print(f"    alpha={alpha:4.2f}: segment RMSE={r:9.1f}{tag}")
        if r < best[1]:
            best = (alpha, r)
    print(f"    best alpha={best[0]}")

# global effect of shrinking every high-risk segment with the best alpha found on ES rows
es_va = None
for alpha in [0.0, 0.25, 0.5, 0.75, 1.0]:
    p = pred.copy()
    for a in np.unique(apt_va):
        seg_tr = keep & (apt == a) & nofpl
        seg_va = (apt_va == a) & nofpl_va
        if seg_tr.sum() < 100 or seg_va.sum() == 0:
            continue
        ys = y[seg_tr]
        P = float((ys > THR).mean())
        if P <= 0.01:
            continue
        const = float(P * ys[ys > THR].mean() + (1 - P) * ys[ys <= THR].mean())
        p[seg_va] = alpha * pred[seg_va] + (1 - alpha) * const
    rep = rmse_report(y_va, np.clip(p, 60, CAP), vp)
    print(f"\nglobal alpha={alpha:4.2f}: mirror={rep['overall']:.1f} jan={rep['jan']:.1f} jul={rep['jul']:.1f}",
          flush=True)
print("done", flush=True)
