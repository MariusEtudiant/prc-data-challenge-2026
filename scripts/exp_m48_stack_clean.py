"""M48: level-2 stacking done properly - identical level-1 configurations for the
out-of-fold predictions and for the holdout predictions.

M47 was inconclusive because the LightGBM out-of-fold came from an older feature set
while its holdout prediction came from the current one: the meta-model learned to
correct a model that was not the one it corrected at prediction time. Here every
level-1 model is fitted on exactly the same features in both roles.

Level 1 (all per airport, 5 month-block folds for OOF + one full fit for the holdout):
  tuned LightGBM | CatBoost | global network (from M47, already clean) | airport network
Level 2: ExtraTrees and LightGBM on [features + level-1 predictions + their spread]

stages: lgb | cb | nnair | stack
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
CAP, NN_CAP, MIN_TRAIN = 86400, 7200, 20000
FOLDS = [(1, 2, 3), (4, 5), (6, 7), (8, 9), (10, 11, 12)]


def _load():
    X = pd.concat([pd.read_parquet(PROC / f) for f in
                   ["X_v2_mirror_t.parquet", "flightctx_train2025.parquet",
                    "depregime_train2025.parquet", "aobtq_train2025.parquet",
                    "anomreg_train2025.parquet"]], axis=1)
    meta = pd.read_parquet(PROC / "meta_train2025.parquet")
    y = meta["y"].astype("float64")
    tr, va, es = meta["tr"].to_numpy(), meta["va"].to_numpy(), meta["es"].to_numpy()
    ok = (y > 0).to_numpy() & (y < 200000).to_numpy()
    art = np.zeros(len(y), bool)
    for k in (1, 2, 3):
        art |= (y - k * 86400).abs().to_numpy() < 3600
    ok &= ~art
    return X, meta, y, tr, va, es, ok, art


def _tuned(a):
    from dc26.mixture import REG_PARAMS
    p = EXP / ("optuna_" + a + ".json")
    b = dict(objective="regression", metric="rmse", verbosity=-1, num_threads=0,
             bagging_freq=1, seed=42)
    if not p.exists():
        return dict(REG_PARAMS)
    d = json.loads(p.read_text())
    b.update({k: v for k, v in d.items() if not k.startswith("_")})
    for k in ("num_leaves", "min_data_in_leaf", "max_bin"):
        b[k] = int(b[k])
    return b


def stage_lgb():
    import lightgbm as lgb
    X, meta, y, tr, va, es, ok, art = _load()
    cat = [c for c in X.columns if X[c].dtype.name == "category"]
    apt = X["AIRPORT"].astype(str).to_numpy()
    apt_va = apt[va]
    month = X["month"].to_numpy()
    base = ok & tr
    oof = np.full(len(X), np.nan)
    hold = np.full(va.sum(), np.nan)
    t0 = time.time()
    for a in np.unique(apt):
        sel = base & (apt == a)
        m_va = apt_va == a
        if sel.sum() < MIN_TRAIN or not m_va.any():
            continue
        p = _tuned(a)
        for months in FOLDS:
            h = sel & np.isin(month, months)
            f = sel & ~np.isin(month, months)
            if h.sum() < 500 or f.sum() < 5000:
                continue
            m = lgb.train(p, lgb.Dataset(X[f], y[f].clip(upper=CAP), categorical_feature=cat),
                          num_boost_round=6000, valid_sets=[lgb.Dataset(X[h], y[h].clip(upper=CAP))],
                          callbacks=[lgb.early_stopping(150, verbose=False)])
            oof[h] = m.predict(X[h], num_iteration=m.best_iteration)
        m = lgb.train(p, lgb.Dataset(X[sel & ~es], y[sel & ~es].clip(upper=CAP), categorical_feature=cat),
                      num_boost_round=6000,
                      valid_sets=[lgb.Dataset(X[sel & es], y[sel & es].clip(upper=CAP))],
                      callbacks=[lgb.early_stopping(150, verbose=False)])
        hold[m_va] = m.predict(X[va][m_va], num_iteration=m.best_iteration)
        print("  [%s] fait (%.0f min)" % (a, (time.time() - t0) / 60), flush=True)
    np.savez(EXP / "m48_lgb.npz", oof=oof, hold=hold)


def stage_cb():
    from catboost import CatBoostRegressor, Pool
    X, meta, y, tr, va, es, ok, art = _load()
    cat = [c for c in X.columns if X[c].dtype.name == "category"]
    for c in cat:
        X[c] = X[c].cat.add_categories("__NA__").fillna("__NA__")
    apt = X["AIRPORT"].astype(str).to_numpy()
    apt_va = apt[va]
    month = X["month"].to_numpy()
    base = ok & tr
    oof = np.full(len(X), np.nan)
    hold = np.full(va.sum(), np.nan)
    t0 = time.time()
    import torch
    gpu = torch.cuda.is_available()
    P = dict(loss_function="RMSE", iterations=700, learning_rate=0.12, depth=7,
             l2_leaf_reg=3.0, random_seed=42, verbose=False,
             max_ctr_complexity=1, cat_features=cat)
    if gpu:
        P.update(task_type="GPU", devices="0")
    else:
        P.update(thread_count=-1)
    print("CatBoost sur %s" % ("GPU" if gpu else "CPU"), flush=True)
    for a in np.unique(apt):
        sel = base & (apt == a)
        m_va = apt_va == a
        if sel.sum() < MIN_TRAIN or not m_va.any():
            continue
        for months in FOLDS:
            h = sel & np.isin(month, months)
            f = sel & ~np.isin(month, months)
            if h.sum() < 500 or f.sum() < 5000:
                continue
            m = CatBoostRegressor(**P)
            m.fit(Pool(X[f], y[f].clip(upper=CAP), cat_features=cat))
            oof[h] = m.predict(X[h])
        m = CatBoostRegressor(**P)
        m.fit(Pool(X[sel], y[sel].clip(upper=CAP), cat_features=cat))
        hold[m_va] = m.predict(X[va][m_va])
        print("  [%s] fait (%.0f min)" % (a, (time.time() - t0) / 60), flush=True)
    np.savez(EXP / "m48_cb.npz", oof=oof, hold=hold)


def stage_nnair():
    import torch
    import torch.nn as nn
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    X, meta, y, tr, va, es, ok, art = _load()
    cat_cols = [c for c in X.columns if X[c].dtype.name == "category"]
    num_cols = [c for c in X.columns if c not in cat_cols]
    apt = X["AIRPORT"].astype(str).to_numpy()
    apt_va = apt[va]
    month = X["month"].to_numpy()
    C = np.zeros((len(X), len(cat_cols)), dtype=np.int64)
    cards = []
    for j, c in enumerate(cat_cols):
        codes = X[c].cat.codes.to_numpy() + 1
        C[:, j] = codes
        cards.append(int(codes.max()) + 1)
    N = X[num_cols].to_numpy(dtype=np.float32)
    miss = (~np.isfinite(N)).astype(np.float32)
    base = ok & tr
    Nf = np.where(np.isfinite(N[base]), N[base], np.nan)
    mu = np.nan_to_num(np.nanmean(Nf, axis=0))
    sd = np.nanstd(Nf, axis=0)
    sd = np.where(np.isfinite(sd) & (sd > 1e-6), sd, 1.0)
    N = np.hstack([np.clip(np.where(np.isfinite(N), (N - mu) / sd, 0.0), -10, 10).astype(np.float32), miss])
    yt = y.clip(upper=NN_CAP).to_numpy(dtype=np.float32)
    ym, ys = float(yt[base].mean()), float(yt[base].std())
    tC = torch.from_numpy(C).to(dev); tN = torch.from_numpy(N).to(dev)
    tY = torch.from_numpy(((yt - ym) / ys).astype(np.float32)).to(dev)

    class Net(nn.Module):
        def __init__(self, seed):
            super().__init__()
            torch.manual_seed(seed)
            self.emb = nn.ModuleList([nn.Embedding(c, min(48, (c + 1) // 2)) for c in cards])
            d = sum(e.embedding_dim for e in self.emb) + N.shape[1]
            self.mlp = nn.Sequential(
                nn.Linear(d, 512), nn.BatchNorm1d(512), nn.SiLU(), nn.Dropout(0.10),
                nn.Linear(512, 256), nn.BatchNorm1d(256), nn.SiLU(),
                nn.Linear(256, 1))

        def forward(self, xc, xn):
            e = [em(xc[:, j]) for j, em in enumerate(self.emb)]
            return self.mlp(torch.cat(e + [xn], 1)).squeeze(1)

    def fit_pred(fit_mask, out_mask, bs=4096, epochs=30):
        idx = torch.from_numpy(np.where(fit_mask)[0]).to(dev)
        net = Net(42).to(dev)
        opt = torch.optim.AdamW(net.parameters(), lr=3e-3, weight_decay=1e-5)
        sched = torch.optim.lr_scheduler.OneCycleLR(
            opt, max_lr=3e-3, total_steps=epochs * (len(idx) // bs + 1), pct_start=0.2)
        for _ in range(epochs):
            net.train()
            perm = idx[torch.randperm(len(idx), device=dev)]
            for i in range(0, len(perm), bs):
                b = perm[i:i + bs]
                opt.zero_grad(set_to_none=True)
                nn.functional.mse_loss(net(tC[b], tN[b]), tY[b]).backward()
                opt.step(); sched.step()
        net.eval()
        oi = torch.from_numpy(np.where(out_mask)[0]).to(dev)
        out = []
        with torch.no_grad():
            for i in range(0, len(oi), 65536):
                b = oi[i:i + 65536]
                out.append(net(tC[b], tN[b]).cpu().numpy())
        return np.concatenate(out) * ys + ym

    oof = np.full(len(X), np.nan)
    hold = np.full(va.sum(), np.nan)
    t0 = time.time()
    for a in np.unique(apt):
        sel = base & (apt == a)
        m_va = apt_va == a
        if sel.sum() < MIN_TRAIN or not m_va.any():
            continue
        for months in FOLDS:
            h = sel & np.isin(month, months)
            f = sel & ~np.isin(month, months)
            if h.sum() < 500 or f.sum() < 5000:
                continue
            oof[h] = fit_pred(f, h)
        va_mask = np.zeros(len(X), bool)
        va_mask[np.where(va)[0][m_va]] = True
        hold[m_va] = fit_pred(sel, va_mask)
        print("  [%s] fait (%.0f min)" % (a, (time.time() - t0) / 60), flush=True)
    np.savez(EXP / "m48_nnair.npz", oof=oof, hold=hold)


def stage_stack():
    import lightgbm as lgb
    from sklearn.ensemble import ExtraTreesRegressor
    from dc26.validation import rmse, rmse_report
    from dc26 import log_experiment
    X, meta, y, tr, va, es, ok, art = _load()
    apt = X["AIRPORT"].astype(str).to_numpy(); apt_va = apt[va]
    y_va = y[va].to_numpy(); clean = ~art[va]
    vp = meta.loc[va, "val_part"].to_numpy()
    cat = [c for c in X.columns if X[c].dtype.name == "category"]

    src = {}
    for name, f in [("lgb", "m48_lgb.npz"), ("cb", "m48_cb.npz"), ("nnair", "m48_nnair.npz")]:
        p = EXP / f
        if p.exists():
            d = np.load(p)
            src[name] = (d["oof"], d["hold"])
    d47 = np.load(EXP / "m47_nn_oof.npz")
    src["nn"] = (d47["oof"], d47["va"])
    names = list(src)
    print("niveau 1 disponible: %s" % names, flush=True)

    base = ok & tr & np.all([np.isfinite(src[n][0]) for n in names], axis=0)
    print("lignes exploitables: %d" % base.sum(), flush=True)
    for n in names:
        okv = np.isfinite(src[n][1]) & clean
        print("  %-6s holdout nettoye=%.1f" % (n, rmse(y_va[okv], np.clip(src[n][1][okv], 60, CAP))), flush=True)

    def build(mask, which):
        Z = X[mask].copy()
        cols = [src[n][which][mask] for n in names]
        for n, c in zip(names, cols):
            Z["l1_" + n] = c
        Z["l1_mean"] = np.mean(cols, axis=0)
        Z["l1_spread"] = np.std(cols, axis=0)
        return Z

    va_pos = np.where(va)[0]
    for tag in ("et2", "lgb2"):
        pred = np.full(va.sum(), np.nan)
        for a in np.unique(apt):
            m_tr = base & (apt == a)
            m_va = apt_va == a
            if m_tr.sum() < MIN_TRAIN or not m_va.any():
                continue
            vm = np.zeros(len(X), bool); vm[va_pos[m_va]] = True
            Ztr = build(m_tr, 0)
            Zva = X[vm].copy()
            cols = [src[n][1][m_va] for n in names]
            for n, c in zip(names, cols):
                Zva["l1_" + n] = c
            Zva["l1_mean"] = np.mean(cols, axis=0)
            Zva["l1_spread"] = np.std(cols, axis=0)
            ytr = y[m_tr].clip(upper=CAP)
            if tag == "et2":
                for Z in (Ztr, Zva):
                    for c in cat:
                        Z[c] = Z[c].cat.codes.astype("float32")
                m = ExtraTreesRegressor(n_estimators=250, min_samples_leaf=5,
                                        max_features=0.4, n_jobs=-1, random_state=42)
                m.fit(Ztr.astype("float32").fillna(0.0), ytr)
                pred[m_va] = m.predict(Zva.astype("float32").fillna(0.0))
            else:
                st = m_tr & es; ft = m_tr & ~es
                m = lgb.train(dict(objective="regression", metric="rmse", learning_rate=0.04,
                                   num_leaves=63, min_data_in_leaf=200, feature_fraction=0.7,
                                   bagging_fraction=0.8, bagging_freq=1, verbosity=-1,
                                   num_threads=0, seed=42),
                              lgb.Dataset(build(ft, 0), y[ft].clip(upper=CAP), categorical_feature=cat),
                              num_boost_round=3000,
                              valid_sets=[lgb.Dataset(build(st, 0), y[st].clip(upper=CAP))],
                              callbacks=[lgb.early_stopping(120, verbose=False)])
                pred[m_va] = m.predict(Zva, num_iteration=m.best_iteration)
        pred = np.clip(pred, 60, CAP)
        rep = rmse_report(y_va, pred, vp)
        jul = vp == "jul"
        print("%-6s NIVEAU 2 : nettoye=%.1f | juillet=%.1f | brut=%.1f" % (
            tag, rmse(y_va[clean], pred[clean]), rmse(y_va[clean & jul], pred[clean & jul]),
            rep["overall"]), flush=True)
        np.save(EXP / ("m48_" + tag + "_va.npy"), pred)
        log_experiment("M48 empilement propre (%s)" % tag, model="L2 sur OOF coherents",
                       features="full + L1", val_rmse=rep["overall"], val_rmse_jan=rep["jan"],
                       val_rmse_jul=rep["jul"],
                       notes="nettoye %.1f" % rmse(y_va[clean], pred[clean]), keep="")


if __name__ == "__main__":
    {"lgb": stage_lgb, "cb": stage_cb, "nnair": stage_nnair, "stack": stage_stack}[sys.argv[1]]()
