"""M9: robustness battery on M5 features (weather included).
Candidates (all sharing the same fitted components):
  A  m3bw        : single LGBM, extremes kept (cap 86400)  [M3b + weather, no calibration]
  B  mixture     : 3-seed Platt mixture (reference)
  C  mixture-cap : pA + min(p*(pB-pA), cap) for cap in {1200, 3600}
  D  avg(A, pA)  : hedge bulk contamination
  E  avg(A, C3600)
Decision criterion: overall, but also jan/jul balance and month-stability."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from dc26 import load_movements_2025, mirror_split, log_experiment
from dc26.validation import rmse_report
from dc26.state_features import get_state_features
from dc26.features import build_features_v1
from dc26.weather import build_weather_features
from dc26.mixture import fit_mixture, predict_mixture, REG_PARAMS

mv = load_movements_2025()
dep = mv[mv["PHASE_mvt"] == "DEP"].copy()
y_all = dep["TAXITIME_SEC_mvt"]

X = pd.concat([build_features_v1(dep), get_state_features("train2025", dep, mv),
               build_weather_features(dep)], axis=1)
cat_cols = [c for c in X.columns if X[c].dtype.name == "category"]

train, val, val_part = mirror_split(dep)
tr_mask = dep.index.isin(train.index)
va_mask = dep.index.isin(val.index)
y_va = y_all[va_mask].to_numpy()
es_all = ((dep["MVT_TIME_UTC_mvt"] >= "2025-12-10") & tr_mask).to_numpy()
es_tr = es_all[tr_mask]

# --- mixture components (3-seed calibrated)
mdl = fit_mixture(X[tr_mask], y_all[tr_mask], es_tr, cat_cols)
pred_mix, p = predict_mixture(mdl, X[va_mask])
pA = mdl["mA"].predict(X[va_mask], num_iteration=mdl["mA"].best_iteration)
pB = mdl["mB"].predict(X[va_mask])
print(f"[mixture fitted] p mean={p.mean():.5f} max={p.max():.3f}", flush=True)

# --- m3bw: single model with extremes kept
keep = tr_mask & (y_all > 0) & (y_all < 200000)
yk = y_all[keep].clip(upper=86400)
es_k = es_all[keep]
m3 = lgb.train(REG_PARAMS, lgb.Dataset(X[keep][~es_k], yk[~es_k], categorical_feature=cat_cols),
               num_boost_round=6000,
               valid_sets=[lgb.Dataset(X[keep][es_k], yk[es_k])],
               callbacks=[lgb.early_stopping(150, verbose=False)])
pM3 = m3.predict(X[va_mask], num_iteration=m3.best_iteration)
print(f"[m3bw fitted] iters={m3.best_iteration}", flush=True)

np.savez(ROOT / "experiments" / "m9_preds.npz", p=p, pA=pA, pB=pB, pM3=pM3, y_va=y_va,
         val_part=val_part)

cands = {
    "A m3bw (stable, no cal)": pM3.clip(60, 86400),
    "B mixture raw": pred_mix,
    "C1 mixture cap1200": (pA + np.minimum(p * (pB - pA), 1200)).clip(60, 86400),
    "C2 mixture cap3600": (pA + np.minimum(p * (pB - pA), 3600)).clip(60, 86400),
    "D avg(m3bw, pA)": (0.5 * pM3 + 0.5 * pA).clip(60, 86400),
    "E avg(m3bw, C2)": (0.5 * pM3 + 0.5 * (pA + np.minimum(p * (pB - pA), 3600))).clip(60, 86400),
}
best = None
for tag, pr in cands.items():
    rep = rmse_report(y_va, pr, val_part)
    print(f"{tag:26s}: overall={rep['overall']:.1f} jan={rep['jan']:.1f} jul={rep['jul']:.1f}", flush=True)
    if best is None or rep["overall"] < best[1]["overall"]:
        best = (tag, rep)

log_experiment(f"M9 robustness battery ({best[0]})", model="see tag",
               features="M5 features", val_rmse=best[1]["overall"],
               val_rmse_jan=best[1]["jan"], val_rmse_jul=best[1]["jul"],
               notes="candidates: m3bw, mixture, capped, averages", keep="KEEP")
print("done", flush=True)
