"""M8: stabilized comparison with 3-seed averaged calibration.
  M8a = M5 features (reference re-run under stable calibration)
  M8b = M5 + runway-config features
Reports both mixture RMSE and pA-only RMSE (feature effect, calibration-free)."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from dc26 import load_movements_2025, mirror_split, log_experiment
from dc26.validation import rmse_report
from dc26.state_features import get_state_features, build_runway_config_features
from dc26.features import build_features_v1
from dc26.weather import build_weather_features
from dc26.mixture import fit_mixture, predict_mixture

mv = load_movements_2025()
dep = mv[mv["PHASE_mvt"] == "DEP"].copy()
y_all = dep["TAXITIME_SEC_mvt"]

Xbase = pd.concat([build_features_v1(dep), get_state_features("train2025", dep, mv),
                   build_weather_features(dep)], axis=1)
Xrwy = build_runway_config_features(dep, mv)

train, val, val_part = mirror_split(dep)
tr_mask = dep.index.isin(train.index)
va_mask = dep.index.isin(val.index)
y_va = y_all[va_mask]
es_all = ((dep["MVT_TIME_UTC_mvt"] >= "2025-12-10") & tr_mask).to_numpy()
es_tr = es_all[tr_mask]


def run(tag, X):
    cat_cols = [c for c in X.columns if X[c].dtype.name == "category"]
    mdl = fit_mixture(X[tr_mask], y_all[tr_mask], es_tr, cat_cols)
    pred, p = predict_mixture(mdl, X[va_mask])
    pA = mdl["mA"].predict(X[va_mask], num_iteration=mdl["mA"].best_iteration)
    rep = rmse_report(y_va.to_numpy(), pred, val_part)
    repA = rmse_report(y_va.to_numpy(), pA, val_part)
    print(f"{tag} mixture: overall={rep['overall']:.1f} jan={rep['jan']:.1f} jul={rep['jul']:.1f} "
          f"| pA-only: overall={repA['overall']:.1f} jan={repA['jan']:.1f} jul={repA['jul']:.1f} "
          f"| p mean={p.mean():.5f} max={p.max():.3f}", flush=True)
    log_experiment(tag, model="mixture 3-seed cal", features="see script",
                   val_rmse=rep["overall"], val_rmse_jan=rep["jan"], val_rmse_jul=rep["jul"],
                   notes=f"pA-only={repA['overall']:.1f}", keep="")
    return rep


run("M8a stable-cal M5 features", Xbase)
run("M8b stable-cal + runway config", pd.concat([Xbase, Xrwy], axis=1))
print("done", flush=True)
