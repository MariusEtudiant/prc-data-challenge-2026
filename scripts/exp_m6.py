"""M6: ensemble on the normal regressor — LightGBM mA + CatBoost mA, blend weight
optimized on the ES window. Classifier and extreme model unchanged (best config so far).
Feature set: best of M5/M5b (weather + TE if it helped) — controlled by USE_TE flag set
after reading the M5 log."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb
from catboost import CatBoostRegressor, Pool

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from dc26 import load_movements_2025, mirror_split, log_experiment, load_log
from dc26.validation import rmse_report, rmse
from dc26.state_features import get_state_features
from dc26.features import build_features_v1
from dc26.weather import build_weather_features
from dc26.mixture import fit_mixture, predict_mixture, THR, CAP, REG_PARAMS

mv = load_movements_2025()
dep = mv[mv["PHASE_mvt"] == "DEP"].copy()
y_all = dep["TAXITIME_SEC_mvt"]

blocks = [build_features_v1(dep), get_state_features("train2025", dep, mv), build_weather_features(dep)]

log = load_log()
m5 = log[log["name"] == "M5 mixture + METAR"]["val_rmse"]
m5b = log[log["name"] == "M5b mixture + METAR + TE stand-rwy"]["val_rmse"]
USE_TE = (not m5.empty and not m5b.empty and float(m5b.iloc[0]) < float(m5.iloc[0]))
if USE_TE:
    train_tmp, _, _ = mirror_split(dep)
    tr_mask_tmp = dep.index.isin(train_tmp.index)
    key = dep["AIRPORT"].astype(str) + "_" + dep["STAND_mvt"].astype(str) + "_" + dep["RUNWAY_mvt"].astype(str)
    tr_norm = tr_mask_tmp & y_all.between(30, 7200)
    te = y_all[tr_norm].groupby(key[tr_norm]).agg(["median", "count"])
    te = te[te["count"] >= 30]["median"]
    blocks.append(pd.DataFrame({"te_stand_rwy_med": key.map(te)}, index=dep.index))
print("USE_TE:", USE_TE)

X = pd.concat(blocks, axis=1)
cat_cols = [c for c in X.columns if X[c].dtype.name == "category"]

train, val, val_part = mirror_split(dep)
tr_mask = dep.index.isin(train.index)
va_mask = dep.index.isin(val.index)
y_va = y_all[va_mask]
es_all = ((dep["MVT_TIME_UTC_mvt"] >= "2025-12-10") & tr_mask).to_numpy()
es_tr = es_all[tr_mask]

# --- reference mixture (LGBM only) reused for clf/mB and LGBM-mA
mdl = fit_mixture(X[tr_mask], y_all[tr_mask], es_tr, cat_cols)
pred_l, p_va = predict_mixture(mdl, X[va_mask])
pA_l_va = mdl["mA"].predict(X[va_mask], num_iteration=mdl["mA"].best_iteration)
pB_va = mdl["mB"].predict(X[va_mask])

# --- CatBoost normal regressor on the same rows
normal = y_all.between(30, THR).to_numpy() & tr_mask
es_n = es_all[normal]
Xn = X[normal].copy()
for c in cat_cols:
    Xn[c] = Xn[c].cat.add_categories("__NA__").fillna("__NA__").astype(str)
Xva_cb = X[va_mask].copy()
for c in cat_cols:
    Xva_cb[c] = Xva_cb[c].cat.add_categories("__NA__").fillna("__NA__").astype(str)

cb = CatBoostRegressor(loss_function="RMSE", iterations=1500, learning_rate=0.1,
                       depth=7, l2_leaf_reg=3.0, random_seed=42, od_type="Iter",
                       od_wait=80, verbose=100, thread_count=-1,
                       max_ctr_complexity=1, cat_features=cat_cols)
cb.fit(Pool(Xn[~es_n], y_all[normal][~es_n], cat_features=cat_cols),
       eval_set=Pool(Xn[es_n], y_all[normal][es_n], cat_features=cat_cols))
pA_c_va = cb.predict(Xva_cb)
print(f"catboost iters: {cb.get_best_iteration()}", flush=True)

# --- blend weight optimized on ES window (normal rows)
esn_rows = normal & es_all
Xes_cb = X[esn_rows].copy()
for c in cat_cols:
    Xes_cb[c] = Xes_cb[c].cat.add_categories("__NA__").fillna("__NA__").astype(str)
pA_l_es = mdl["mA"].predict(X[esn_rows], num_iteration=mdl["mA"].best_iteration)
pA_c_es = cb.predict(Xes_cb)
y_es = y_all[esn_rows].to_numpy()
ws = np.linspace(0, 1, 21)
scores = [rmse(y_es, w * pA_c_es + (1 - w) * pA_l_es) for w in ws]
w_best = float(ws[int(np.argmin(scores))])
print(f"blend: w_catboost={w_best:.2f} | ES rmse lgbm={scores[0]:.2f} cb={scores[-1]:.2f} best={min(scores):.2f}")

# --- final mixture with blended mA
pA_blend = w_best * pA_c_va + (1 - w_best) * pA_l_va
pred = ((1 - p_va) * pA_blend + p_va * pB_va).clip(60, CAP)
rep = rmse_report(y_va.to_numpy(), pred, val_part)
rep_l = rmse_report(y_va.to_numpy(), pred_l, val_part)
print(f"LGBM-only mixture : overall={rep_l['overall']:.1f} jan={rep_l['jan']:.1f} jul={rep_l['jul']:.1f}")
print(f"M6 blended mixture: overall={rep['overall']:.1f} jan={rep['jan']:.1f} jul={rep['jul']:.1f}")
log_experiment("M6 mixture + CatBoost blend", model="mixture(LGBM+CB)",
               features=f"M5{'b' if USE_TE else ''} features; w_cb={w_best:.2f}",
               val_rmse=rep["overall"], val_rmse_jan=rep["jan"], val_rmse_jul=rep["jul"],
               notes=f"cb_iters={cb.get_best_iteration()}", keep="")
print("done")
