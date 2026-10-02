"""M6 staged (anti-OOM): each stage runs in its own process, communicating via disk.

stages: build -> lgbm -> cb -> blend
  build : feature matrix (v1+state+weather, float32) + meta -> data/processed/
  lgbm  : reference mixture (clf+mA+mB), saves val/ES predictions
  cb    : CatBoost normal regressor (category dtype, no string blowup), saves preds
  blend : ES-optimized weight, final report, experiment log
"""
import gc
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
PROC = ROOT / "data" / "processed"
NPZ = ROOT / "experiments" / "m6_preds.npz"

X_PATH = PROC / "X_train2025.parquet"
META_PATH = PROC / "meta_train2025.parquet"


def stage_build():
    from dc26 import load_movements_2025, mirror_split
    from dc26.state_features import get_state_features
    from dc26.features import build_features_v1
    from dc26.weather import build_weather_features

    mv = load_movements_2025()
    dep = mv[mv["PHASE_mvt"] == "DEP"].copy()

    X = build_features_v1(dep)
    X = pd.concat([X, get_state_features("train2025", dep, mv)], axis=1)
    X = pd.concat([X, build_weather_features(dep)], axis=1)
    for c in X.columns:
        if X[c].dtype == "float64":
            X[c] = X[c].astype("float32")
    gc.collect()

    train, val, val_part = mirror_split(dep)
    meta = pd.DataFrame({
        "y": dep["TAXITIME_SEC_mvt"].astype("float32"),
        "tr": dep.index.isin(train.index),
        "va": dep.index.isin(val.index),
        "es": (dep["MVT_TIME_UTC_mvt"] >= "2025-12-10").to_numpy(),
    })
    vp = np.full(len(dep), "", dtype=object)
    vp[meta["va"].to_numpy()] = val_part
    meta["val_part"] = vp
    del mv, dep, train, val
    gc.collect()

    X.reset_index(drop=True).to_parquet(X_PATH, index=False)
    meta.reset_index(drop=True).to_parquet(META_PATH, index=False)
    print(f"[build] X: {X.shape}, saved.", flush=True)


def _load():
    X = pd.read_parquet(X_PATH)
    meta = pd.read_parquet(META_PATH)
    return X, meta


def stage_lgbm():
    from dc26.mixture import fit_mixture, predict_mixture
    from dc26.validation import rmse_report

    X, meta = _load()
    cat_cols = [c for c in X.columns if X[c].dtype.name == "category"]
    tr, va = meta["tr"].to_numpy(), meta["va"].to_numpy()
    y = meta["y"].astype("float64")
    es_tr = meta["es"].to_numpy()[tr]

    mdl = fit_mixture(X[tr], y[tr], es_tr, cat_cols)
    pred_l, p_va = predict_mixture(mdl, X[va])
    pA_l_va = mdl["mA"].predict(X[va], num_iteration=mdl["mA"].best_iteration)
    pB_va = mdl["mB"].predict(X[va])

    normal = y.between(30, 7200).to_numpy()
    esn = normal & meta["es"].to_numpy() & tr
    pA_l_es = mdl["mA"].predict(X[esn], num_iteration=mdl["mA"].best_iteration)

    vp = meta.loc[va, "val_part"].to_numpy()
    rep = rmse_report(y[va].to_numpy(), pred_l, vp)
    print(f"[lgbm] mixture LGBM-only: overall={rep['overall']:.1f} jan={rep['jan']:.1f} "
          f"jul={rep['jul']:.1f} (iterA={mdl['mA'].best_iteration})", flush=True)
    mdl["mA"].save_model(str(ROOT / "models" / "m6_mA_lgbm.txt"))
    np.savez(NPZ, pA_l_va=pA_l_va, pB_va=pB_va, p_va=p_va, pA_l_es=pA_l_es)


def stage_cb():
    from catboost import CatBoostRegressor, Pool

    X, meta = _load()
    cat_cols = [c for c in X.columns if X[c].dtype.name == "category"]
    for c in cat_cols:  # CatBoost: no NaN in categorical features
        X[c] = X[c].cat.add_categories("__NA__").fillna("__NA__")
    tr, va = meta["tr"].to_numpy(), meta["va"].to_numpy()
    y = meta["y"].astype("float64")
    normal = y.between(30, 7200).to_numpy() & tr
    es_n = meta["es"].to_numpy()[normal]
    esn_rows = normal & meta["es"].to_numpy()

    cb = CatBoostRegressor(loss_function="RMSE", iterations=1500, learning_rate=0.1,
                           depth=7, l2_leaf_reg=3.0, random_seed=42, od_type="Iter",
                           od_wait=80, verbose=100, thread_count=-1,
                           max_ctr_complexity=1, cat_features=cat_cols)
    cb.fit(Pool(X[normal][~es_n], y[normal][~es_n], cat_features=cat_cols),
           eval_set=Pool(X[normal][es_n], y[normal][es_n], cat_features=cat_cols))
    print(f"[cb] best_iteration={cb.get_best_iteration()}", flush=True)

    pA_c_va = cb.predict(X[va])
    pA_c_es = cb.predict(X[esn_rows])
    d = dict(np.load(NPZ))
    d.update(pA_c_va=pA_c_va, pA_c_es=pA_c_es)
    np.savez(NPZ, **d)
    cb.save_model(str(ROOT / "models" / "m6_mA_catboost.cbm"))


def stage_blend():
    from dc26.validation import rmse_report, rmse
    from dc26 import log_experiment

    _, meta = _load()
    d = np.load(NPZ, allow_pickle=True)
    y = meta["y"].astype("float64")
    tr, va = meta["tr"].to_numpy(), meta["va"].to_numpy()
    normal = y.between(30, 7200).to_numpy() & tr
    esn_rows = normal & meta["es"].to_numpy()
    y_es = y[esn_rows].to_numpy()
    y_va = y[va].to_numpy()
    vp = meta.loc[va, "val_part"].to_numpy()

    ws = np.linspace(0, 1, 21)
    scores = [rmse(y_es, w * d["pA_c_es"] + (1 - w) * d["pA_l_es"]) for w in ws]
    w = float(ws[int(np.argmin(scores))])
    print(f"[blend] w_catboost={w:.2f} | ES rmse: lgbm={scores[0]:.2f} cb={scores[-1]:.2f} "
          f"blend={min(scores):.2f}", flush=True)

    pA = w * d["pA_c_va"] + (1 - w) * d["pA_l_va"]
    pred = ((1 - d["p_va"]) * pA + d["p_va"] * d["pB_va"]).clip(60, 86400)
    rep = rmse_report(y_va, pred, vp)
    print(f"[blend] M6: overall={rep['overall']:.1f} jan={rep['jan']:.1f} jul={rep['jul']:.1f}", flush=True)
    log_experiment("M6 mixture + CatBoost blend", model="mixture(LGBM+CB)",
                   features=f"M5 features; w_cb={w:.2f}",
                   val_rmse=rep["overall"], val_rmse_jan=rep["jan"], val_rmse_jul=rep["jul"],
                   notes="staged run", keep="")


if __name__ == "__main__":
    stage = sys.argv[1]
    {"build": stage_build, "lgbm": stage_lgbm, "cb": stage_cb, "blend": stage_blend}[stage]()
