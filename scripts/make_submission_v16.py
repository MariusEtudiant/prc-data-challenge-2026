"""Submission v16 - everything measured but never shipped, in one model.

We sit 140th of 217 with 343.31 and TWO submissions, on a model (v14) that was
explicitly a one-hour stopgap to get a valid-format file onto the board: trees only,
no networks, artefacts excluded from training, and none of the gains measured since.
Teams in the 51-200 submission bracket have a median best of 247.5.

Three measured, independent gains go in:
  1. artefacts KEPT in training. On the metric the competition actually scores - the
     RAW one, artefacts included - this was -3.58 (398.35 against 401.93), while the
     "cleaned" mirror we had been optimising said +5.29. We spent five days judging on
     a metric that removes the flights which decide the score.
  2. the neighbour feature: mean predicted taxi of the five previous departures on the
     same runway, built from out-of-fold predictions. -1.46 cleaned overall and -11.7
     at LTFM, a third of what the oracle showed was there.
  3. max_bin 255 instead of 511: same accuracy, roughly twice as fast, which matters
     with nine days left.

stages: nbr | mirror | prod | blend
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
CAP, MIN_TRAIN, N_ROWS, K = 86400, 20000, 344841, 5
SEEDS = [42, 1337, 2024]
FOLDS = [(2, 3), (4, 5), (6, 8), (9, 10), (11, 12)]
_EPOCH = pd.Timestamp(0, tz="UTC")
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
    valid = (y > 0).to_numpy() & (y < 200000).to_numpy()
    art = np.zeros(len(y), bool)
    for k in (1, 2, 3):
        art |= (y - k * 86400).abs().to_numpy() < 3600
    return meta, y, valid, art


def _lgb(a, seed):
    from dc26.mixture import REG_PARAMS
    p = EXP / ("optuna_" + a + ".json")
    b = dict(objective="regression", metric="rmse", verbosity=-1, num_threads=0, bagging_freq=1,
             seed=seed, bagging_seed=seed + 1, feature_fraction_seed=seed + 2,
             feature_pre_filter=False, max_bin=255)
    if p.exists():
        d = json.loads(p.read_text())
        b.update({k: v for k, v in d.items() if not k.startswith("_")})
        for k in ("num_leaves", "min_data_in_leaf"):
            b[k] = int(b[k])
        b["max_bin"] = 255
    else:
        b.update(REG_PARAMS)
    b["seed"] = seed
    return b


def _cb(a, cat):
    p = EXP / ("optuna_cb_" + a + ".json")
    b = dict(loss_function="RMSE", iterations=3000, learning_rate=0.1, depth=7, l2_leaf_reg=3.0,
             random_seed=42, od_type="Iter", od_wait=100, verbose=False, max_ctr_complexity=1,
             cat_features=cat)
    if p.exists():
        d = json.loads(p.read_text())
        for k, v in d.items():
            if not k.startswith("_"):
                b[k] = int(v) if k in ("depth", "max_ctr_complexity", "border_count") else v
        b["iterations"] = 3000
    import torch
    b.update(dict(task_type="GPU", devices="0") if torch.cuda.is_available() else dict(thread_count=-1))
    return b


def _seq_key(side):
    f = "X_v2_mirror_t.parquet" if side == "train" else "X_v2_ranking_t.parquet"
    apt = pd.read_parquet(PROC / f, columns=["AIRPORT"])["AIRPORT"].astype(str)
    src = "movements_2025.parquet" if side == "train" else "ranking_2026.parquet"
    mv = pd.read_parquet(PROC / src, columns=["PHASE_mvt", "RUNWAY_mvt", "MVT_TIME_UTC_mvt"])
    d = mv[mv["PHASE_mvt"] == "DEP"].reset_index(drop=True)
    t = ((d["MVT_TIME_UTC_mvt"] - _EPOCH) / pd.Timedelta(seconds=1)).to_numpy(dtype=float)
    return (apt + "|" + d["RUNWAY_mvt"].astype(str)).to_numpy(), t


def _neighbours(src, key, t):
    out = np.full(len(src), np.nan)
    for _, idx in pd.Series(np.arange(len(src))).groupby(key):
        i = idx.to_numpy()
        o = i[np.argsort(t[i])]
        v, g = src[o], np.isfinite(src[o])
        run, buf = np.full(len(o), np.nan), []
        for j in range(len(o)):
            run[j] = np.mean(buf) if len(buf) >= 3 else np.nan
            if g[j]:
                buf.append(v[j])
                if len(buf) > K:
                    buf.pop(0)
        out[o] = run
    return out


def stage_nbr():
    """Out-of-fold predictions on training rows, v14's model on the ranking rows,
    then the neighbour feature on both sides from those predictions."""
    import lightgbm as lgb
    meta, y, valid, art = _meta()
    X = pd.concat([pd.read_parquet(PROC / f) for f in TRAIN_F], axis=1)
    tr, es = meta["tr"].to_numpy(), meta["es"].to_numpy()
    keep = valid & ~art            # the source model is the plain one, as in S02
    cat = [c for c in X.columns if X[c].dtype.name == "category"]
    apt = X["AIRPORT"].astype(str).to_numpy()
    mon = pd.read_parquet(PROC / "movements_2025.parquet", columns=["PHASE_mvt", "MVT_TIME_UTC_mvt"])
    mon = mon[mon["PHASE_mvt"] == "DEP"]["MVT_TIME_UTC_mvt"].reset_index(drop=True).dt.month.to_numpy()
    src = np.full(len(y), np.nan)
    t0 = time.time()
    for a in np.unique(apt):
        sel = keep & (apt == a)
        if sel.sum() < MIN_TRAIN:
            continue
        for months in FOLDS:
            hold = sel & np.isin(mon, months)
            fit = sel & ~np.isin(mon, months) & ~es
            stop = sel & ~np.isin(mon, months) & es
            if hold.sum() < 500 or fit.sum() < MIN_TRAIN // 2 or stop.sum() < 200:
                continue
            P = [lgb.train(_lgb(a, sd),
                           lgb.Dataset(X[fit], y[fit].clip(upper=CAP), categorical_feature=cat,
                                       params={"feature_pre_filter": False, "max_bin": 255}),
                           num_boost_round=6000,
                           valid_sets=[lgb.Dataset(X[stop], y[stop].clip(upper=CAP))],
                           callbacks=[lgb.early_stopping(150, verbose=False)]) for sd in SEEDS[:2]]
            src[hold] = np.mean([m.predict(X[hold], num_iteration=m.best_iteration) for m in P], 0)
        print("  [%s] hors-pli %.0f%% (%.0f min)" % (
            a, 100 * np.isfinite(src[sel]).mean(), (time.time() - t0) / 60), flush=True)
    kt, tt = _seq_key("train")
    np.save(EXP / "v16_nbr_train.npy", _neighbours(src, kt, tt))
    # ranking side: v14's per-airport LGBM already predicts every ranking row
    rk = np.load(EXP / "v14_rank.npz")["lgbm"]
    kr, tr_ = _seq_key("rank")
    np.save(EXP / "v16_nbr_rank.npy", _neighbours(np.clip(rk, 60, CAP), kr, tr_))
    print("[nbr] termine", flush=True)


def _train_block(X, Xout, y, keep, es, apt, apt_out, tag):
    """Per-airport LGBM (3 seeds) + CatBoost, artefacts KEPT in training."""
    import lightgbm as lgb
    from catboost import CatBoostRegressor, Pool
    cat = [c for c in X.columns if X[c].dtype.name == "category"]
    lg = np.full(len(Xout), np.nan)
    cb = np.full(len(Xout), np.nan)
    Xc, Xoc = X.copy(), Xout.copy()
    for c in cat:
        Xc[c] = Xc[c].cat.add_categories("__NA__").fillna("__NA__")
        Xoc[c] = Xoc[c].cat.add_categories("__NA__").fillna("__NA__")
    t0 = time.time()
    for a in np.unique(apt):
        f = keep & (apt == a) & ~es
        s = keep & (apt == a) & es
        r = apt_out == a
        if f.sum() < MIN_TRAIN or s.sum() < 300 or not r.any():
            continue
        P = [lgb.train(_lgb(a, sd),
                       lgb.Dataset(X[f], y[f].clip(upper=CAP), categorical_feature=cat,
                                   params={"feature_pre_filter": False, "max_bin": 255}),
                       num_boost_round=6000,
                       valid_sets=[lgb.Dataset(X[s], y[s].clip(upper=CAP))],
                       callbacks=[lgb.early_stopping(150, verbose=False)]) for sd in SEEDS]
        lg[r] = np.mean([m.predict(Xout[r], num_iteration=m.best_iteration) for m in P], 0)
        m = CatBoostRegressor(**_cb(a, cat))
        m.fit(Pool(Xc[f], y[f].clip(upper=CAP), cat_features=cat),
              eval_set=Pool(Xc[s], y[s].clip(upper=CAP), cat_features=cat))
        cb[r] = m.predict(Xoc[r])
        print("  [%s %s] fait (%.0f min)" % (tag, a, (time.time() - t0) / 60), flush=True)
    return np.clip(lg, 60, CAP), np.clip(cb, 60, CAP)


def stage_mirror():
    meta, y, valid, art = _meta()
    X = pd.concat([pd.read_parquet(PROC / f) for f in TRAIN_F], axis=1)
    X["nbr_pred_rwy"] = np.load(EXP / "v16_nbr_train.npy")
    tr, va, es = meta["tr"].to_numpy(), meta["va"].to_numpy(), meta["es"].to_numpy()
    apt = X["AIRPORT"].astype(str).to_numpy()
    out = np.zeros(len(y), bool)
    out[np.where(va)[0]] = True
    lg, cb = _train_block(X, X[out], y, valid & tr, es, apt, apt[va], "mirror")
    np.savez(EXP / "v16_mirror.npz", lgbm=lg, cb=cb)
    print("[mirror] termine", flush=True)


def stage_prod():
    from dc26.pipeline import align_categories
    meta, y, valid, art = _meta()
    X = pd.concat([pd.read_parquet(PROC / f) for f in FULL_F], axis=1)
    Xr = align_categories(pd.concat([pd.read_parquet(PROC / f) for f in RANK_F], axis=1), X)
    assert len(Xr) == N_ROWS
    X["nbr_pred_rwy"] = np.load(EXP / "v16_nbr_train.npy")
    Xr["nbr_pred_rwy"] = np.load(EXP / "v16_nbr_rank.npy")
    es = meta["es"].to_numpy()
    apt = X["AIRPORT"].astype(str).to_numpy()
    lg, cb = _train_block(X, Xr, y, valid, es, apt, Xr["AIRPORT"].astype(str).to_numpy(), "prod")
    np.savez(EXP / "v16_rank.npz", lgbm=lg, cb=cb)
    print("[prod] termine", flush=True)


def stage_blend():
    from scipy.optimize import nnls
    from dc26.validation import rmse
    meta, y, valid, art = _meta()
    va = meta["va"].to_numpy()
    yv = y[va].to_numpy()
    apt_va = pd.read_parquet(PROC / "X_v2_mirror_t.parquet", columns=["AIRPORT"])["AIRPORT"].astype(str).to_numpy()[va]
    day = pd.read_parquet(PROC / "movements_2025.parquet", columns=["PHASE_mvt", "MVT_TIME_UTC_mvt"])
    day = day[day["PHASE_mvt"] == "DEP"]["MVT_TIME_UTC_mvt"].reset_index(drop=True).dt.day.to_numpy()[va]
    m16 = np.load(EXP / "v16_mirror.npz")
    n39 = np.load(EXP / "m39_preds.npz")
    ng = np.clip(n39["nn_global"], 60, CAP)
    na = np.clip(np.where(np.isfinite(n39["nn_airport"]), n39["nn_airport"], ng), 60, CAP)
    V = np.vstack([m16["lgbm"], m16["cb"], ng, na])
    names = ["lgbm", "cb", "nn_glob", "nn_apt"]
    fin = np.all(np.isfinite(V), axis=0)
    cl = ~art[va]
    W, t_raw_a, t_raw_b, n = {}, 0.0, 0.0, 0
    print("=== poids, ajustes jours IMPAIRS, mesures jours PAIRS (metrique BRUTE) ===")
    for a in sorted(set(apt_va)):
        mo = fin & (apt_va == a) & (day % 2 == 1)
        me = fin & (apt_va == a) & (day % 2 == 0)
        if mo.sum() < 500 or me.sum() < 500:
            continue
        w, _ = nnls(V[:, mo].T, yv[mo])
        w = w / w.sum() if w.sum() > 0 else np.array([1.0, 0, 0, 0])
        W[a] = [float(x) for x in w]
        p = np.clip(w @ V[:, me], 60, CAP)
        t_raw_a += ((yv[me] - np.clip(V[0, me], 60, CAP)) ** 2).sum()
        t_raw_b += ((yv[me] - p) ** 2).sum()
        n += me.sum()
        print("  %-8s %s | LGBM=%.1f -> melange=%.1f" % (
            a, " ".join("%s=%.2f" % (nm, x) for nm, x in zip(names, w) if x > 0.01),
            rmse(yv[me], np.clip(V[0, me], 60, CAP)), rmse(yv[me], p)))
    print("\n  jours pairs BRUT : LGBM seul=%.2f -> melange=%.2f" % (
        np.sqrt(t_raw_a / n), np.sqrt(t_raw_b / n)))
    (EXP / "v16_weights.json").write_text(json.dumps(W))

    pm = np.full(len(yv), np.nan)
    for a, w in W.items():
        m = apt_va == a
        if m.any():
            pm[m] = np.clip(np.array(w) @ np.nan_to_num(V[:, m]), 60, CAP)
    g = np.isfinite(pm)
    print("  miroir complet : BRUT=%.2f | nettoye=%.2f" % (
        rmse(yv[g], pm[g]), rmse(yv[g & cl], pm[g & cl])))

    d = np.load(EXP / "v16_rank.npz")
    nn = np.load(EXP / "v11_nn_preds.npz")
    rg = np.clip(nn["nn_global"], 60, CAP)
    ra = np.clip(np.where(np.isfinite(nn["nn_airport"]), nn["nn_airport"], rg), 60, CAP)
    R = np.vstack([d["lgbm"], d["cb"], rg, ra])
    apt_r = pd.read_parquet(PROC / "X_v2_ranking_t.parquet", columns=["AIRPORT"])["AIRPORT"].astype(str).to_numpy()
    pred = np.full(N_ROWS, np.nan)
    for a, w in W.items():
        r = apt_r == a
        if not r.any():
            continue
        M = R[:, r]
        wa = np.array(w, float)
        okc = np.all(np.isfinite(M), axis=1)
        if wa[~okc].sum() > 0:
            wa = np.where(okc, wa, 0.0)
            wa = wa / wa.sum() if wa.sum() > 0 else np.array([1.0, 0, 0, 0])
        pred[r] = wa @ np.nan_to_num(M)
    miss = ~np.isfinite(pred)
    if miss.any():
        pred[miss] = np.nan_to_num(R[0], nan=900.0)[miss]
        print("  repli sur %d lignes" % miss.sum())
    pred = np.clip(pred, 60, CAP)
    ids = pd.read_parquet(PROC / "rank_ids_t.parquet")["MVT_ID_mvt"]
    tpl = pd.read_parquet(ROOT / "data" / "submitting.parquet").drop(columns=["TAXITIME_SEC_mvt"])
    out = tpl.merge(pd.DataFrame({"MVT_ID_mvt": ids.values, "TAXITIME_SEC_mvt": pred}),
                    on="MVT_ID_mvt", how="left")
    assert out["TAXITIME_SEC_mvt"].notna().all() and len(out) == N_ROWS
    assert out["MVT_ID_mvt"].tolist() == tpl["MVT_ID_mvt"].tolist()
    assert out["TAXITIME_SEC_mvt"].between(60, CAP).all()
    out.to_parquet(ROOT / "submissions" / "reliable-hamburger_v16.parquet", index=False)
    t = out["TAXITIME_SEC_mvt"]
    print("v16 ecrite: n=%d mean=%.0f median=%.0f q99=%.0f" % (
        len(out), t.mean(), t.median(), t.quantile(.99)))


if __name__ == "__main__":
    {"nbr": stage_nbr, "mirror": stage_mirror, "prod": stage_prod,
     "blend": stage_blend}[sys.argv[1]]()
