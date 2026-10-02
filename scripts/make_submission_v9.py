"""Submission v8 - the M26 architecture (mirror 319.1) retrained on all of 2025.

stages:
  feats : build ranking-side flight-context and departure-regime features
  lgbm  : per-airport Optuna-tuned LGBM, 3 seeds averaged
  cb    : per-airport CatBoost
  glob  : global LGBM
  blend : apply the per-airport weights from M26, write the parquet
"""
import gc
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
PROC = ROOT / "data" / "processed"
NPZ = ROOT / "experiments" / "v9_rank_preds.npz"
M26 = ROOT / "experiments" / "m26_preds.npz"
CAP = 86400
SEEDS = [42, 1337, 2024]
MIN_TRAIN = 20000
BOOST = 1.12

F_RANK_CTX = PROC / "flightctx_ranking2026.parquet"
F_RANK_REG = PROC / "depregime_ranking2026.parquet"


def _train():
    X = pd.concat([pd.read_parquet(PROC / "X_v2_full_t.parquet"),
                   pd.read_parquet(PROC / "flightctx_train2025.parquet"),
                   pd.read_parquet(PROC / "depregime_train2025.parquet"),
                   pd.read_parquet(PROC / "aobtq_full2025.parquet")], axis=1)
    meta = pd.read_parquet(PROC / "meta_train2025.parquet")
    y = meta["y"].astype("float64")
    keep = (y > 0).to_numpy() & (y < 200000).to_numpy()
    # M30: day-offset artifacts are data errors absent from the 2026 extract
    art = np.zeros(len(y), bool)
    for k in (1, 2, 3):
        art |= (y - k * 86400).abs().to_numpy() < 3600
    keep &= ~art
    return X, y, keep


def _rank(X_ref):
    from dc26.pipeline import align_categories
    Xr = pd.concat([pd.read_parquet(PROC / "X_v2_ranking_t.parquet"),
                    pd.read_parquet(F_RANK_CTX),
                    pd.read_parquet(F_RANK_REG),
                    pd.read_parquet(PROC / "aobtq_ranking2026.parquet")], axis=1)
    return align_categories(Xr, X_ref)


def _save(**kw):
    d = dict(np.load(NPZ, allow_pickle=True)) if NPZ.exists() else {}
    d.update(kw)
    np.savez(NPZ, **d)
    assert set(kw) <= set(np.load(NPZ, allow_pickle=True).keys()), "persist check failed"


def tuned_params(a):
    p = ROOT / "experiments" / ("optuna_" + a + ".json")
    base = dict(objective="regression", metric="rmse", verbosity=-1,
                num_threads=0, bagging_freq=1, seed=42)
    if not p.exists():
        from dc26.mixture import REG_PARAMS
        return dict(REG_PARAMS)
    d = json.loads(p.read_text())
    base.update({k: v for k, v in d.items() if not k.startswith("_")})
    for k in ("num_leaves", "min_data_in_leaf", "max_bin"):
        base[k] = int(base[k])
    return base


def stage_feats():
    from dc26 import load_ranking_2026, load_movements_2025
    from dc26.flight_context import build_flight_context
    from dc26.dep_regime import build_dep_regime
    from dc26.aobt_quality import fit_aobt_spikes, build_aobt_quality
    rk = load_ranking_2026()
    rk_dep = rk[rk["PHASE_mvt"] == "DEP"].copy()
    mv = load_movements_2025()
    dep = mv[mv["PHASE_mvt"] == "DEP"].reset_index(drop=True)
    spikes = fit_aobt_spikes(dep)                     # fitted on the full training year
    build_aobt_quality(dep, spikes).reset_index(drop=True).to_parquet(
        PROC / "aobtq_full2025.parquet", index=False)
    build_aobt_quality(rk_dep, spikes).reset_index(drop=True).to_parquet(
        PROC / "aobtq_ranking2026.parquet", index=False)
    print("[feats] AOBT quality saved", flush=True)
    build_flight_context(rk_dep).reset_index(drop=True).to_parquet(F_RANK_CTX, index=False)
    print("[feats] flight context saved", flush=True)
    build_dep_regime(rk_dep, rk).reset_index(drop=True).to_parquet(F_RANK_REG, index=False)
    print("[feats] departure regime saved", flush=True)


def _shift_weights(X, keep):
    """M31: importance weights for the measured unmatched-flight shift."""
    from dc26 import load_ranking_2026
    rk = load_ranking_2026()
    rkd = rk[rk["PHASE_mvt"] == "DEP"]
    test_rate = rkd.groupby("AIRPORT")["FLIGHT_ID_mvt"].apply(lambda s: s.isna().mean())
    apt = X["AIRPORT"].astype(str).to_numpy()
    unm = (X["has_fpl"].to_numpy() == 0)
    train_rate = pd.Series(unm[keep]).groupby(apt[keep]).mean()
    w = np.ones(len(X))
    for a in np.unique(apt):
        te, tr_ = float(test_rate.get(a, np.nan)), float(train_rate.get(a, np.nan))
        if not np.isfinite(te) or not np.isfinite(tr_) or tr_ <= 0:
            continue
        w[(apt == a) & unm] = min(te / tr_, 5.0)
        w[(apt == a) & ~unm] = (1 - te) / (1 - tr_)
    return w


def stage_lgbm():
    import lightgbm as lgb
    X, y, keep = _train()
    W = _shift_weights(X, keep)
    Xr = _rank(X)
    cat = [c for c in X.columns if X[c].dtype.name == "category"]
    apt = X["AIRPORT"].astype(str).to_numpy()
    apt_r = Xr["AIRPORT"].astype(str).to_numpy()
    iters = json.loads(str(np.load(M26, allow_pickle=True)["air_lgbm_iters"]))
    P = np.full((len(SEEDS), len(Xr)), np.nan)
    for a in np.unique(apt):
        f = keep & (apt == a)
        r = apt_r == a
        if f.sum() < MIN_TRAIN or not r.any() or a not in iters:
            continue
        for si, seed in enumerate(SEEDS):
            n = max(int(iters[a][si] * BOOST), 50)
            p = dict(tuned_params(a), seed=seed, bagging_seed=seed + 1,
                     feature_fraction_seed=seed + 2)
            m = lgb.train(p, lgb.Dataset(X[f], y[f].clip(upper=CAP), categorical_feature=cat,
                                         weight=W[f]), num_boost_round=n)
            P[si, r] = m.predict(Xr[r])
            m.save_model(str(ROOT / "models" / ("v9_lgbm_" + a + "_s" + str(seed) + ".txt")))
        print("[lgbm] " + a + " done", flush=True)
    _save(air_lgbm=np.nanmean(P, axis=0))


def stage_cb():
    from catboost import CatBoostRegressor, Pool
    X, y, keep = _train()
    Xr = _rank(X)
    cat = [c for c in X.columns if X[c].dtype.name == "category"]
    for c in cat:
        X[c] = X[c].cat.add_categories("__NA__").fillna("__NA__")
        Xr[c] = Xr[c].cat.add_categories("__NA__").fillna("__NA__")
    gc.collect()
    apt = X["AIRPORT"].astype(str).to_numpy()
    apt_r = Xr["AIRPORT"].astype(str).to_numpy()
    iters = json.loads(str(np.load(M26, allow_pickle=True)["air_cb_iters"]))
    p_rk = np.full(len(Xr), np.nan)
    for a in np.unique(apt):
        f = keep & (apt == a)
        r = apt_r == a
        if f.sum() < MIN_TRAIN or not r.any() or a not in iters:
            continue
        n = max(int(iters[a] * BOOST), 50)
        m = CatBoostRegressor(loss_function="RMSE", iterations=n, learning_rate=0.1, depth=7,
                              l2_leaf_reg=3.0, random_seed=42, verbose=False, thread_count=-1,
                              max_ctr_complexity=1, cat_features=cat)
        m.fit(Pool(X[f], y[f].clip(upper=CAP), cat_features=cat))
        p_rk[r] = m.predict(Xr[r])
        m.save_model(str(ROOT / "models" / ("v9_cb_" + a + ".cbm")))
        print("[cb] " + a + " done", flush=True)
    _save(air_cb=p_rk)


def stage_glob():
    import lightgbm as lgb
    from dc26.mixture import REG_PARAMS
    X, y, keep = _train()
    Xr = _rank(X)
    cat = [c for c in X.columns if X[c].dtype.name == "category"]
    n = int(int(np.load(M26, allow_pickle=True)["glob_lgbm_iters"]) * BOOST)
    m = lgb.train(REG_PARAMS, lgb.Dataset(X[keep], y[keep].clip(upper=CAP), categorical_feature=cat),
                  num_boost_round=n)
    m.save_model(str(ROOT / "models" / "v9_glob_lgbm.txt"))
    print("[glob] done", flush=True)
    _save(glob_lgbm=m.predict(Xr))


def stage_blend():
    X, y, keep = _train()
    Xr = _rank(X)
    apt_r = Xr["AIRPORT"].astype(str).to_numpy()
    d = np.load(NPZ, allow_pickle=True)
    w_all = json.loads((ROOT / "experiments" / "m26_weights.json").read_text())
    P = np.vstack([d["air_lgbm"], d["air_cb"], d["glob_lgbm"]])
    pred = np.full(len(Xr), np.nan)
    for a, w in w_all.items():
        r = apt_r == a
        if not r.any():
            continue
        w = np.array(w, dtype=float)
        ok = np.all(np.isfinite(P[:, r]), axis=1)
        if w[~ok].sum() > 0:
            w = np.where(ok, w, 0.0)
            w = w / w.sum() if w.sum() > 0 else np.array([0.0, 0.0, 1.0])
        pred[r] = w @ P[:, r]
        print("  [" + a + "] w=" + str(np.round(w, 2)) + " n=" + str(int(r.sum())), flush=True)
    miss = ~np.isfinite(pred)
    if miss.any():
        pred[miss] = d["glob_lgbm"][miss]
        print("  fallback on " + str(int(miss.sum())) + " rows", flush=True)
    pred = np.clip(pred, 60, CAP)

    ids = pd.read_parquet(PROC / "rank_ids_t.parquet")["MVT_ID_mvt"]
    tpl = pd.read_parquet(ROOT / "data" / "submitting.parquet").drop(columns=["TAXITIME_SEC_mvt"])
    out = tpl.merge(pd.DataFrame({"MVT_ID_mvt": ids.values, "TAXITIME_SEC_mvt": pred}),
                    on="MVT_ID_mvt", how="left")
    assert out["TAXITIME_SEC_mvt"].notna().all() and len(out) == 215876
    out.to_parquet(ROOT / "submissions" / "reliable-hamburger_v9.parquet", index=False)
    print("v8 written: mean=%.0fs median=%.0fs q99=%.0fs over2h=%d" % (
        out["TAXITIME_SEC_mvt"].mean(), out["TAXITIME_SEC_mvt"].median(),
        out["TAXITIME_SEC_mvt"].quantile(.99), (out["TAXITIME_SEC_mvt"] > 7200).sum()), flush=True)


if __name__ == "__main__":
    {"feats": stage_feats, "lgbm": stage_lgbm, "cb": stage_cb,
     "glob": stage_glob, "blend": stage_blend}[sys.argv[1]]()
