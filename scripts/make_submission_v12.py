"""Submission v12 = v11 with the tuned CatBoost and jointly-fitted per-airport weights.

Only the CatBoost part is recomputed (tuned parameters, GPU); the per-airport LightGBM
and the global LightGBM come from the v10 run, the two networks from the v11 run.
The five sources are then combined with the weights fitted jointly per airport on the
mirror (experiments/m42_weights.json), instead of the stacked approximations used up
to v11.

stages: cb | blend
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
PROC = ROOT / "data" / "processed"
EXP = ROOT / "experiments"
NPZ = EXP / "v12_cb.npz"
CAP, MIN_TRAIN = 86400, 20000

TRAIN_FILES = ["X_v2_full_t.parquet", "flightctx_train2025.parquet",
               "depregime_train2025.parquet", "aobtq_full2025.parquet",
               "anomreg_train2025.parquet"]
RANK_FILES = ["X_v2_ranking_t.parquet", "flightctx_ranking2026.parquet",
              "depregime_ranking2026.parquet", "aobtq_ranking2026.parquet",
              "anomreg_ranking2026.parquet"]


def _data():
    from dc26.pipeline import align_categories
    X = pd.concat([pd.read_parquet(PROC / f) for f in TRAIN_FILES], axis=1)
    Xr = align_categories(pd.concat([pd.read_parquet(PROC / f) for f in RANK_FILES], axis=1), X)
    meta = pd.read_parquet(PROC / "meta_train2025.parquet")
    y = meta["y"].astype("float64")
    keep = (y > 0).to_numpy() & (y < 200000).to_numpy()
    art = np.zeros(len(y), bool)
    for k in (1, 2, 3):
        art |= (y - k * 86400).abs().to_numpy() < 3600
    return X, Xr, y, keep & ~art


def _tuned_cb(a, cat):
    p = EXP / ("optuna_cb_" + a + ".json")
    b = dict(loss_function="RMSE", iterations=1500, learning_rate=0.1, depth=7,
             l2_leaf_reg=3.0, random_seed=42, verbose=False, max_ctr_complexity=1,
             cat_features=cat)
    if p.exists():
        d = json.loads(p.read_text())
        for k, v in d.items():
            if not k.startswith("_"):
                b[k] = int(v) if k in ("depth", "max_ctr_complexity", "border_count") else v
        b["iterations"] = 2000
    import torch
    if torch.cuda.is_available():
        b.update(task_type="GPU", devices="0")
    else:
        b["thread_count"] = -1
    return b


def stage_cb():
    from catboost import CatBoostRegressor, Pool
    X, Xr, y, keep = _data()
    cat = [c for c in X.columns if X[c].dtype.name == "category"]
    for c in cat:
        X[c] = X[c].cat.add_categories("__NA__").fillna("__NA__")
        Xr[c] = Xr[c].cat.add_categories("__NA__").fillna("__NA__")
    apt = X["AIRPORT"].astype(str).to_numpy()
    apt_r = Xr["AIRPORT"].astype(str).to_numpy()
    pred = np.full(len(Xr), np.nan)
    t0 = time.time()
    for a in np.unique(apt):
        f, r = keep & (apt == a), apt_r == a
        if f.sum() < MIN_TRAIN or not r.any():
            continue
        m = CatBoostRegressor(**_tuned_cb(a, cat))
        m.fit(Pool(X[f], y[f].clip(upper=CAP), cat_features=cat))
        pred[r] = m.predict(Xr[r])
        m.save_model(str(ROOT / "models" / ("v12_cb_" + a + ".cbm")))
        print("  [%s] fait (%.0f min)" % (a, (time.time() - t0) / 60), flush=True)
    np.savez(NPZ, air_cb=pred)
    print("[cb] termine", flush=True)


def stage_blend():
    X, Xr, y, keep = _data()
    apt_r = Xr["AIRPORT"].astype(str).to_numpy()
    d10 = np.load(EXP / "v10_rank_preds.npz", allow_pickle=True)
    d11 = np.load(EXP / "v11_nn_preds.npz")
    cb = np.load(NPZ)["air_cb"]
    ng = d11["nn_global"]
    na = np.where(np.isfinite(d11["nn_airport"]), d11["nn_airport"], ng)
    V = {"air_lgb": d10["air_lgbm"], "air_cb": cb, "glob_lgb": d10["glob_lgbm"],
         "nn_global": ng, "nn_airport": na}
    names = ["air_lgb", "air_cb", "glob_lgb", "nn_global", "nn_airport"]
    W = json.loads((EXP / "m42_weights.json").read_text())

    pred = np.full(len(Xr), np.nan)
    for a, w in W.items():
        r = apt_r == a
        if not r.any():
            continue
        w = np.array(w, dtype=float)
        M = np.vstack([np.clip(V[n], 60, CAP) for n in names])[:, r]
        ok = np.all(np.isfinite(M), axis=1)
        if w[~ok].sum() > 0:
            w = np.where(ok, w, 0.0)
            w = w / w.sum() if w.sum() > 0 else None
        if w is None:
            continue
        pred[r] = w @ np.nan_to_num(M)
        print("  [%s] %s | n=%d" % (a, " ".join("%s=%.2f" % (n, x) for n, x in zip(names, w) if x > 0.01),
                                    int(r.sum())), flush=True)
    miss = ~np.isfinite(pred)
    if miss.any():
        pred[miss] = np.clip(V["air_lgb"], 60, CAP)[miss]
        print("  repli sur %d lignes" % miss.sum(), flush=True)
    pred = np.clip(pred, 60, CAP)
    v11 = pd.read_parquet(ROOT / "submissions" / "reliable-hamburger_v11.parquet")
    ids = pd.read_parquet(PROC / "rank_ids_t.parquet")["MVT_ID_mvt"]
    tpl = pd.read_parquet(ROOT / "data" / "submitting.parquet").drop(columns=["TAXITIME_SEC_mvt"])
    out = tpl.merge(pd.DataFrame({"MVT_ID_mvt": ids.values, "TAXITIME_SEC_mvt": pred}),
                    on="MVT_ID_mvt", how="left")
    assert out["TAXITIME_SEC_mvt"].notna().all() and len(out) == 215876
    out.to_parquet(ROOT / "submissions" / "reliable-hamburger_v12.parquet", index=False)
    j = out.merge(v11, on="MVT_ID_mvt", suffixes=("_12", "_11"))
    t = out["TAXITIME_SEC_mvt"]
    print("v12 ecrite: mean=%.0f median=%.0f q99=%.0f | ecart moyen vs v11=%.0fs" % (
        t.mean(), t.median(), t.quantile(.99),
        (j["TAXITIME_SEC_mvt_12"] - j["TAXITIME_SEC_mvt_11"]).abs().mean()), flush=True)


if __name__ == "__main__":
    {"cb": stage_cb, "blend": stage_blend}[sys.argv[1]]()
