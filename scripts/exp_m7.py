"""M7a: M5 features + runway-config features.
M7b: M5 features + higher-capacity LGBM (lr 0.03, leaves 511) for the normal regressor."""
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from dc26 import load_movements_2025, mirror_split, log_experiment
from dc26.validation import rmse_report
from dc26.state_features import get_state_features, build_runway_config_features
from dc26.features import build_features_v1
from dc26.weather import build_weather_features
from dc26.mixture import fit_mixture, predict_mixture, REG_PARAMS

mv = load_movements_2025()
dep = mv[mv["PHASE_mvt"] == "DEP"].copy()
y_all = dep["TAXITIME_SEC_mvt"]

Xbase = pd.concat([build_features_v1(dep), get_state_features("train2025", dep, mv),
                   build_weather_features(dep)], axis=1)
Xrwy = build_runway_config_features(dep, mv)
print("runway-config features built:", list(Xrwy.columns))

train, val, val_part = mirror_split(dep)
tr_mask = dep.index.isin(train.index)
va_mask = dep.index.isin(val.index)
y_va = y_all[va_mask]
es_all = ((dep["MVT_TIME_UTC_mvt"] >= "2025-12-10") & tr_mask).to_numpy()
es_tr = es_all[tr_mask]


def run(tag, X, reg_params=None):
    cat_cols = [c for c in X.columns if X[c].dtype.name == "category"]
    mdl = fit_mixture(X[tr_mask], y_all[tr_mask], es_tr, cat_cols, reg_params=reg_params)
    pred, _ = predict_mixture(mdl, X[va_mask])
    rep = rmse_report(y_va.to_numpy(), pred, val_part)
    print(f"{tag}: overall={rep['overall']:.1f} jan={rep['jan']:.1f} jul={rep['jul']:.1f} "
          f"(iterA={mdl['mA'].best_iteration})")
    log_experiment(tag, model="mixture", features="see script",
                   val_rmse=rep["overall"], val_rmse_jan=rep["jan"], val_rmse_jul=rep["jul"], keep="")
    return rep


run("M7a mixture + runway config", pd.concat([Xbase, Xrwy], axis=1))

big = dict(REG_PARAMS, learning_rate=0.03, num_leaves=511, min_data_in_leaf=50)
run("M7b mixture big-capacity", Xbase, reg_params=big)
