"""M47: real level-2 stacking - the piece AutoGluon showed we were missing.

Its ExtraTrees at level 1 scored -430.8 on LIRF (worst of all models) and -378.2 at
level 2 (best of all). The family was never the point: a level-2 model receives the
raw features AND the level-1 out-of-fold predictions, so it can learn conditional
corrections ("when the queue is long and LightGBM says 900s, add 200") that a
weighted average cannot express. Our meta-model has always been a linear average.

Level 1 : per-airport tuned LGBM (OOF from M27) | per-airport CatBoost (OOF from M29)
          | global network (OOF computed here on GPU)
Level 2 : LightGBM and ExtraTrees, per airport, on [features + level-1 predictions]

stages: nnoof | stack
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
NPZ = ROOT / "experiments" / "m47_nn_oof.npz"
CAP, NN_CAP, MIN_TRAIN = 86400, 7200, 20000
FOLDS = [(1, 2, 3), (4, 5), (6, 7), (8, 9), (10, 11, 12)]
EPOCHS = 30


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


def stage_nnoof():
    """Out-of-fold predictions of the network, 5 month-block folds, on GPU."""
    import torch
    import torch.nn as nn
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    X, meta, y, tr, va, es, ok, art = _load()
    cat_cols = [c for c in X.columns if X[c].dtype.name == "category"]
    num_cols = [c for c in X.columns if c not in cat_cols]
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
    N = np.clip(np.where(np.isfinite(N), (N - mu) / sd, 0.0), -10, 10).astype(np.float32)
    N = np.hstack([N, miss])
    yt = y.clip(upper=NN_CAP).to_numpy(dtype=np.float32)
    ym, ys = float(yt[base].mean()), float(yt[base].std())
    tC = torch.from_numpy(C).to(dev)
    tN = torch.from_numpy(N).to(dev)
    tY = torch.from_numpy(((yt - ym) / ys).astype(np.float32)).to(dev)
    print("device %s" % dev, flush=True)

    class Net(nn.Module):
        def __init__(self, seed):
            super().__init__()
            torch.manual_seed(seed)
            self.emb = nn.ModuleList([nn.Embedding(c, min(48, (c + 1) // 2)) for c in cards])
            d = sum(e.embedding_dim for e in self.emb) + N.shape[1]
            self.mlp = nn.Sequential(
                nn.Linear(d, 1024), nn.BatchNorm1d(1024), nn.SiLU(), nn.Dropout(0.15),
                nn.Linear(1024, 512), nn.BatchNorm1d(512), nn.SiLU(), nn.Dropout(0.10),
                nn.Linear(512, 256), nn.BatchNorm1d(256), nn.SiLU(),
                nn.Linear(256, 1))

        def forward(self, xc, xn):
            e = [em(xc[:, j]) for j, em in enumerate(self.emb)]
            return self.mlp(torch.cat(e + [xn], 1)).squeeze(1)

    def predict(net, idx, bs=65536):
        net.eval()
        out = []
        with torch.no_grad():
            for i in range(0, len(idx), bs):
                b = idx[i:i + bs]
                out.append(net(tC[b], tN[b]).cpu().numpy())
        return np.concatenate(out) * ys + ym

    def train(fit_mask, stop_mask, seed=42, bs=8192):
        idx_fit = torch.from_numpy(np.where(fit_mask)[0]).to(dev)
        idx_stop = torch.from_numpy(np.where(stop_mask)[0]).to(dev)
        y_stop = yt[np.where(stop_mask)[0]]
        net = Net(seed).to(dev)
        opt = torch.optim.AdamW(net.parameters(), lr=3e-3, weight_decay=1e-5)
        sched = torch.optim.lr_scheduler.OneCycleLR(
            opt, max_lr=3e-3, total_steps=EPOCHS * (len(idx_fit) // bs + 1), pct_start=0.2)
        best, state, pat = 1e18, None, 0
        for ep in range(EPOCHS):
            net.train()
            perm = idx_fit[torch.randperm(len(idx_fit), device=dev)]
            for i in range(0, len(perm), bs):
                b = perm[i:i + bs]
                opt.zero_grad(set_to_none=True)
                nn.functional.mse_loss(net(tC[b], tN[b]), tY[b]).backward()
                opt.step()
                sched.step()
            r = float(np.sqrt(np.mean((predict(net, idx_stop) - y_stop) ** 2)))
            if r < best - 0.3:
                best, state, pat = r, {k: v.clone() for k, v in net.state_dict().items()}, 0
            else:
                pat += 1
                if pat >= 5:
                    break
        net.load_state_dict(state)
        return net, best

    t0 = time.time()
    oof = np.full(len(X), np.nan)
    for months in FOLDS:
        hold = base & np.isin(month, months)
        fit = base & ~np.isin(month, months) & ~es
        stop = base & ~np.isin(month, months) & es
        if stop.sum() < 500:
            stop = hold
        net, b = train(fit, stop)
        oof[hold] = predict(net, torch.from_numpy(np.where(hold)[0]).to(dev))
        print("  fold %s ES=%.1f (%.0fs)" % (months, b, time.time() - t0), flush=True)
    net, b = train(base & ~es, base & es)
    va_pred = predict(net, torch.from_numpy(np.where(va)[0]).to(dev))
    np.savez(NPZ, oof=oof, va=va_pred)
    print("[nnoof] termine en %.0f min" % ((time.time() - t0) / 60), flush=True)


def stage_stack():
    import lightgbm as lgb
    from sklearn.ensemble import ExtraTreesRegressor
    from dc26.validation import rmse, rmse_report
    from dc26 import log_experiment
    X, meta, y, tr, va, es, ok, art = _load()
    apt = X["AIRPORT"].astype(str).to_numpy()
    apt_va = apt[va]
    y_va = y[va].to_numpy()
    clean = ~art[va]
    vp = meta.loc[va, "val_part"].to_numpy()

    # ---- level-1: out-of-fold on training rows, full-model predictions on holdout
    l1_oof = {
        "lgb": np.load(ROOT / "experiments" / "m27_oof.npy"),
        "cb": np.load(ROOT / "experiments" / "m29_cb_oof.npy"),
        "nn": np.load(NPZ)["oof"],
    }
    m26 = np.load(ROOT / "experiments" / "m26_preds.npz", allow_pickle=True)
    l1_va = {
        "lgb": np.load(ROOT / "experiments" / "m36_gbdt_va.npy"),
        "cb": m26["air_cb_va"],
        "nn": np.load(NPZ)["va"],
    }
    names = list(l1_oof)
    usable = ok & tr & np.all([np.isfinite(l1_oof[n]) for n in names], axis=0)
    print("lignes exploitables pour le niveau 2: %d" % usable.sum(), flush=True)

    def add_l1(base_df, mask, source):
        Z = base_df[mask].copy()
        for n in names:
            Z["l1_" + n] = source[n][mask]
        Z["l1_mean"] = np.mean([source[n][mask] for n in names], axis=0)
        Z["l1_spread"] = np.std([source[n][mask] for n in names], axis=0)
        return Z

    cat = [c for c in X.columns if X[c].dtype.name == "category"]
    for tag in ("lgb2", "et2"):
        pred = np.full(va.sum(), np.nan)
        for a in np.unique(apt):
            m_tr = usable & (apt == a)
            m_va = apt_va == a
            if m_tr.sum() < MIN_TRAIN or not m_va.any():
                continue
            Ztr = add_l1(X, m_tr, l1_oof)
            va_idx = np.zeros(len(X), bool)
            va_idx[np.where(va)[0][m_va]] = True
            Zva = X[va_idx].copy()
            for n in names:
                Zva["l1_" + n] = l1_va[n][m_va]
            Zva["l1_mean"] = np.mean([l1_va[n][m_va] for n in names], axis=0)
            Zva["l1_spread"] = np.std([l1_va[n][m_va] for n in names], axis=0)
            ytr = y[m_tr].clip(upper=CAP)
            if tag == "lgb2":
                stop = m_tr & es
                fit = m_tr & ~es
                m = lgb.train(dict(objective="regression", metric="rmse", learning_rate=0.04,
                                   num_leaves=63, min_data_in_leaf=100, feature_fraction=0.8,
                                   bagging_fraction=0.8, bagging_freq=1, verbosity=-1,
                                   num_threads=0, seed=42),
                              lgb.Dataset(add_l1(X, fit, l1_oof), y[fit].clip(upper=CAP),
                                          categorical_feature=cat),
                              num_boost_round=3000,
                              valid_sets=[lgb.Dataset(add_l1(X, stop, l1_oof),
                                                      y[stop].clip(upper=CAP))],
                              callbacks=[lgb.early_stopping(120, verbose=False)])
                pred[m_va] = m.predict(Zva, num_iteration=m.best_iteration)
            else:
                Zn = Ztr.copy(); Zvn = Zva.copy()
                for c in cat:
                    Zn[c] = Zn[c].cat.codes.astype("float32")
                    Zvn[c] = Zvn[c].cat.codes.astype("float32")
                Zn = Zn.astype("float32").fillna(0.0)
                Zvn = Zvn.astype("float32").fillna(0.0)
                m = ExtraTreesRegressor(n_estimators=200, min_samples_leaf=5,
                                        max_features=0.4, n_jobs=-1, random_state=42)
                m.fit(Zn, ytr)
                pred[m_va] = m.predict(Zvn)
        pred = np.clip(pred, 60, CAP)
        rep = rmse_report(y_va, pred, vp)
        print("%-8s niveau 2 : nettoye=%.1f | brut=%.1f (jan %.1f jul %.1f)" % (
            tag, rmse(y_va[clean], pred[clean]), rep["overall"], rep["jan"], rep["jul"]), flush=True)
        np.save(ROOT / "experiments" / ("m47_" + tag + "_va.npy"), pred)
        log_experiment("M47 empilement niveau 2 (%s)" % tag, model="L2 sur OOF de 3 modeles",
                       features="full + predictions L1", val_rmse=rep["overall"],
                       val_rmse_jan=rep["jan"], val_rmse_jul=rep["jul"],
                       notes="nettoye %.1f" % rmse(y_va[clean], pred[clean]), keep="")


if __name__ == "__main__":
    {"nnoof": stage_nnoof, "stack": stage_stack}[sys.argv[1]]()
