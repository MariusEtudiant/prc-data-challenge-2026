"""M5: champion mixture + METAR weather features.
Also M5b: + stand->runway target encoding (median taxi per stand x runway, train-only)."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from dc26 import load_movements_2025, mirror_split, log_experiment
from dc26.validation import rmse_report
from dc26.state_features import get_state_features
from dc26.features import build_features_v1
from dc26.weather import build_weather_features
from dc26.mixture import fit_mixture, predict_mixture

mv = load_movements_2025()
dep = mv[mv["PHASE_mvt"] == "DEP"].copy()

Xv1 = build_features_v1(dep)
Xst = get_state_features("train2025", dep, mv)
Xwx = build_weather_features(dep)
print(f"weather features: notna tmpc {Xwx['tmpc'].notna().mean():.3f}, "
      f"snow rate {Xwx['wx_snow'].mean():.4f}, deice rate {Xwx['deice_proxy'].mean():.4f}")

train, val, val_part = mirror_split(dep)
tr_mask = dep.index.isin(train.index)
va_mask = dep.index.isin(val.index)
y_all = dep["TAXITIME_SEC_mvt"]
y_va = y_all[va_mask]
es_all = ((dep["MVT_TIME_UTC_mvt"] >= "2025-12-10") & tr_mask).to_numpy()
es_tr = es_all[tr_mask]


def evaluate(tag, X_blocks, keep_flag=""):
    X = pd.concat(X_blocks, axis=1)
    cat_cols = [c for c in X.columns if X[c].dtype.name == "category"]
    mdl = fit_mixture(X[tr_mask], y_all[tr_mask], es_tr, cat_cols)
    pred, p = predict_mixture(mdl, X[va_mask])
    rep = rmse_report(y_va.to_numpy(), pred, val_part)
    print(f"{tag}: overall={rep['overall']:.1f} jan={rep['jan']:.1f} jul={rep['jul']:.1f} "
          f"(iterA={mdl['mA'].best_iteration})")
    log_experiment(tag, model="mixture", features=",".join(b.columns[0] for b in X_blocks),
                   val_rmse=rep["overall"], val_rmse_jan=rep["jan"], val_rmse_jul=rep["jul"],
                   notes="", keep=keep_flag)
    return rep, mdl


# M5: + weather
rep5, mdl5 = evaluate("M5 mixture + METAR", [Xv1, Xst, Xwx])

# M5b: + stand x runway target encoding (computed on train rows only, mirror-safe)
key = dep["AIRPORT"].astype(str) + "_" + dep["STAND_mvt"].astype(str) + "_" + dep["RUNWAY_mvt"].astype(str)
tr_norm = tr_mask & y_all.between(30, 7200)
te = y_all[tr_norm].groupby(key[tr_norm]).agg(["median", "count"])
te = te[te["count"] >= 30]["median"]
Xte = pd.DataFrame({"te_stand_rwy_med": key.map(te)}, index=dep.index)
print(f"target encoding coverage: {Xte['te_stand_rwy_med'].notna().mean():.3f}")
rep5b, _ = evaluate("M5b mixture + METAR + TE stand-rwy", [Xv1, Xst, Xwx, Xte])
