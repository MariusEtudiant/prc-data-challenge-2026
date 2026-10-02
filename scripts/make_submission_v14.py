"""Submission v14 - first submission against the regenerated July test set.

Everything before this was built for a 215,876-row test that was 70% January. The new
one is 344,841 rows and 55.7% July, so every stored blend weight is calibrated against
a composition that no longer exists, and our existing submission files are the wrong
shape entirely.

This rebuilds the tree half of the ensemble end to end on the new data:
  cbmirror  per-airport CatBoost on the mirror, so weights can be fitted
  final     per-airport LGBM (3 seeds) + CatBoost trained on 2025 minus the ES window
            and early-stopped on it - no BOOST heuristic, only 5.6% of rows withheld
  submit    per-airport weights fitted on ODD days of the mirror, checked on EVEN days,
            then applied to the ranking predictions

Networks come in a later version; this one establishes the new scale and puts a
valid-format file on the leaderboard.

stages: cbmirror | final | submit
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
PROC, EXP = ROOT / "data" / "processed", ROOT / "experiments"
CAP, MIN_TRAIN, N_ROWS = 86400, 20000, 344841
SEEDS = [42, 1337, 2024]
TRAIN_F = ["X_v2_mirror_t.parquet", "flightctx_train2025.parquet", "depregime_train2025.parquet",
           "aobtq_train2025.parquet", "anomreg_train2025.parquet"]
FULL_F = ["X_v2_full_t.parquet", "flightctx_train2025.parquet", "depregime_train2025.parquet",
          "aobtq_full2025.parquet", "anomreg_train2025.parquet"]
RANK_F = ["X_v2_ranking_t.parquet", "flightctx_ranking2026.parquet",
          "depregime_ranking2026.parquet", "aobtq_ranking2026.parquet",
          "anomreg_ranking2026.parquet"]


def _meta():
    meta = pd.read_parquet(PROC / "meta_train2025.parquet")
    y = meta["y"].astype("float64")
    ok = (y > 0).to_numpy() & (y < 200000).to_numpy()
    art = np.zeros(len(y), bool)
    for k in (1, 2, 3):
        art |= (y - k * 86400).abs().to_numpy() < 3600
    return meta, y, ok & ~art, art


def _lgb_params(a, seed):
    from dc26.mixture import REG_PARAMS
    p = EXP / ("optuna_" + a + ".json")
    b = dict(objective="regression", metric="rmse", verbosity=-1, num_threads=0,
             bagging_freq=1, seed=seed, bagging_seed=seed + 1, feature_fraction_seed=seed + 2)
    if not p.exists():
        return dict(REG_PARAMS, seed=seed)
    d = json.loads(p.read_text())
    b.update({k: v for k, v in d.items() if not k.startswith("_")})
    for k in ("num_leaves", "min_data_in_leaf", "max_bin"):
        b[k] = int(b[k])
    b["seed"] = seed
    return b


def _cb_params(a, cat):
    p = EXP / ("optuna_cb_" + a + ".json")
    b = dict(loss_function="RMSE", iterations=3000, learning_rate=0.1, depth=7,
             l2_leaf_reg=3.0, random_seed=42, od_type="Iter", od_wait=100,
             verbose=False, max_ctr_complexity=1, cat_features=cat)
    if p.exists():
        d = json.loads(p.read_text())
        for k, v in d.items():
            if not k.startswith("_"):
                b[k] = int(v) if k in ("depth", "max_ctr_complexity", "border_count") else v
        b["iterations"] = 3000
    import torch
    if torch.cuda.is_available():
        b.update(task_type="GPU", devices="0")
    else:
        b["thread_count"] = -1
    return b


def stage_cbmirror():
    from catboost import CatBoostRegressor, Pool
    meta, y, ok, art = _meta()
    X = pd.concat([pd.read_parquet(PROC / f) for f in TRAIN_F], axis=1)
    tr, va, es = meta["tr"].to_numpy(), meta["va"].to_numpy(), meta["es"].to_numpy()
    cat = [c for c in X.columns if X[c].dtype.name == "category"]
    for c in cat:
        X[c] = X[c].cat.add_categories("__NA__").fillna("__NA__")
    apt = X["AIRPORT"].astype(str).to_numpy()
    pred = np.full(va.sum(), np.nan)
    t0 = time.time()
    for a in np.unique(apt):
        f, s = ok & tr & (apt == a) & ~es, ok & tr & (apt == a) & es
        sv = apt[va] == a
        if f.sum() < MIN_TRAIN or not sv.any():
            continue
        m = CatBoostRegressor(**_cb_params(a, cat))
        m.fit(Pool(X[f], y[f].clip(upper=CAP), cat_features=cat),
              eval_set=Pool(X[s], y[s].clip(upper=CAP), cat_features=cat))
        pred[sv] = m.predict(X[va][sv])
        print("  [%s] %d iterations (%.0f min)" % (a, m.get_best_iteration(),
                                                   (time.time() - t0) / 60), flush=True)
    np.save(EXP / "v14_cb_mirror.npy", np.clip(pred, 60, CAP))
    print("[cbmirror] termine", flush=True)


def stage_final():
    import lightgbm as lgb
    from catboost import CatBoostRegressor, Pool
    from dc26.pipeline import align_categories
    meta, y, ok, art = _meta()
    X = pd.concat([pd.read_parquet(PROC / f) for f in FULL_F], axis=1)
    Xr = align_categories(pd.concat([pd.read_parquet(PROC / f) for f in RANK_F], axis=1), X)
    assert len(Xr) == N_ROWS, "ranking: %d != %d" % (len(Xr), N_ROWS)
    es = meta["es"].to_numpy()
    cat = [c for c in X.columns if X[c].dtype.name == "category"]
    apt, apt_r = X["AIRPORT"].astype(str).to_numpy(), Xr["AIRPORT"].astype(str).to_numpy()
    lgbm = np.full(len(Xr), np.nan)
    t0 = time.time()
    for a in np.unique(apt):
        f, s = ok & (apt == a) & ~es, ok & (apt == a) & es
        r = apt_r == a
        if f.sum() < MIN_TRAIN or not r.any():
            continue
        P = []
        for seed in SEEDS:
            m = lgb.train(_lgb_params(a, seed),
                          lgb.Dataset(X[f], y[f].clip(upper=CAP), categorical_feature=cat),
                          num_boost_round=6000,
                          valid_sets=[lgb.Dataset(X[s], y[s].clip(upper=CAP))],
                          callbacks=[lgb.early_stopping(150, verbose=False)])
            P.append(m.predict(Xr[r], num_iteration=m.best_iteration))
        lgbm[r] = np.mean(P, 0)
        print("  [lgbm %s] fait (%.0f min)" % (a, (time.time() - t0) / 60), flush=True)
    Xc, Xrc = X.copy(), Xr.copy()
    for c in cat:
        Xc[c] = Xc[c].cat.add_categories("__NA__").fillna("__NA__")
        Xrc[c] = Xrc[c].cat.add_categories("__NA__").fillna("__NA__")
    cb = np.full(len(Xr), np.nan)
    for a in np.unique(apt):
        f, s = ok & (apt == a) & ~es, ok & (apt == a) & es
        r = apt_r == a
        if f.sum() < MIN_TRAIN or not r.any():
            continue
        m = CatBoostRegressor(**_cb_params(a, cat))
        m.fit(Pool(Xc[f], y[f].clip(upper=CAP), cat_features=cat),
              eval_set=Pool(Xc[s], y[s].clip(upper=CAP), cat_features=cat))
        cb[r] = m.predict(Xrc[r])
        print("  [cb %s] %d iterations (%.0f min)" % (a, m.get_best_iteration(),
                                                      (time.time() - t0) / 60), flush=True)
    np.savez(EXP / "v14_rank.npz", lgbm=np.clip(lgbm, 60, CAP), cb=np.clip(cb, 60, CAP))
    print("[final] termine", flush=True)


def stage_submit():
    from scipy.optimize import nnls
    from dc26.validation import rmse
    meta, y, ok, art = _meta()
    X = pd.read_parquet(PROC / "X_v2_mirror_t.parquet", columns=["AIRPORT"])
    va = meta["va"].to_numpy()
    yv = y[va].to_numpy()
    apt_va = X["AIRPORT"].astype(str).to_numpy()[va]
    clean = ~art[va]
    mvt = pd.read_parquet(PROC / "movements_2025.parquet",
                          columns=["PHASE_mvt", "MVT_TIME_UTC_mvt"])
    day = mvt[mvt["PHASE_mvt"] == "DEP"]["MVT_TIME_UTC_mvt"].reset_index(drop=True).dt.day.to_numpy()[va]
    V = np.vstack([np.load(EXP / "m96_reference.npy"), np.load(EXP / "v14_cb_mirror.npy")])
    names = ["lgbm", "cb"]

    W, tot_o, tot_e, n = {}, 0.0, 0.0, 0
    print("=== poids par aeroport, ajustes sur les jours IMPAIRS, mesures sur les PAIRS ===")
    fin = np.all(np.isfinite(V), axis=0)
    for a in sorted(set(apt_va)):
        mo = clean & (apt_va == a) & (day % 2 == 1) & fin
        me = clean & (apt_va == a) & (day % 2 == 0) & fin
        if mo.sum() < 500 or me.sum() < 500:
            continue
        w, _ = nnls(V[:, mo].T, yv[mo])
        w = w / w.sum() if w.sum() > 0 else np.array([1.0, 0.0])
        W[a] = [float(x) for x in w]
        p = np.clip(w @ V[:, me], 60, CAP)
        b = np.clip(V[0, me], 60, CAP)
        tot_o += ((yv[me] - b) ** 2).sum()
        tot_e += ((yv[me] - p) ** 2).sum()
        n += me.sum()
        print("  %-8s %s | LGBM seul=%.1f -> melange=%.1f" % (
            a, " ".join("%s=%.2f" % (nm, x) for nm, x in zip(names, w)),
            rmse(yv[me], b), rmse(yv[me], p)))
    print("\n  jours pairs : LGBM seul=%.2f -> melange=%.2f (%+.2f)" % (
        np.sqrt(tot_o / n), np.sqrt(tot_e / n), np.sqrt(tot_e / n) - np.sqrt(tot_o / n)))
    (EXP / "v14_weights.json").write_text(json.dumps(W))

    d = np.load(EXP / "v14_rank.npz")
    R = np.vstack([d["lgbm"], d["cb"]])
    Xr = pd.read_parquet(PROC / "X_v2_ranking_t.parquet", columns=["AIRPORT"])
    apt_r = Xr["AIRPORT"].astype(str).to_numpy()
    pred = np.full(len(Xr), np.nan)
    for a, w in W.items():
        r = apt_r == a
        if not r.any():
            continue
        M = R[:, r]
        wa = np.array(w, float)
        okc = np.all(np.isfinite(M), axis=1)
        if wa[~okc].sum() > 0:
            wa = np.where(okc, wa, 0.0)
            wa = wa / wa.sum() if wa.sum() > 0 else np.array([1.0, 0.0])
        pred[r] = wa @ np.nan_to_num(M)
    miss = ~np.isfinite(pred)
    if miss.any():
        pred[miss] = np.nan_to_num(d["lgbm"], nan=900.0)[miss]
        print("  repli sur %d lignes" % miss.sum())
    pred = np.clip(pred, 60, CAP)
    ids = pd.read_parquet(PROC / "rank_ids_t.parquet")["MVT_ID_mvt"]
    tpl = pd.read_parquet(ROOT / "data" / "submitting.parquet").drop(columns=["TAXITIME_SEC_mvt"])
    out = tpl.merge(pd.DataFrame({"MVT_ID_mvt": ids.values, "TAXITIME_SEC_mvt": pred}),
                    on="MVT_ID_mvt", how="left")
    assert out["TAXITIME_SEC_mvt"].notna().all(), "valeurs manquantes"
    assert len(out) == N_ROWS, "%d != %d" % (len(out), N_ROWS)
    out.to_parquet(ROOT / "submissions" / "reliable-hamburger_v14.parquet", index=False)
    t = out["TAXITIME_SEC_mvt"]
    print("v14 ecrite: n=%d mean=%.0f median=%.0f q99=%.0f" % (
        len(out), t.mean(), t.median(), t.quantile(.99)))


if __name__ == "__main__":
    {"cbmirror": stage_cbmirror, "final": stage_final, "submit": stage_submit}[sys.argv[1]]()
