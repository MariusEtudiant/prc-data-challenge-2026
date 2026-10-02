"""M36: entity-embedding neural network - a genuinely different inductive bias.

Every model tried so far is a tree ensemble. Trees approximate the identity in
staircases, which is why handing d_aobt3 as an offset (M16) already bought 1.7s.
A network represents smooth relations natively, so what matters for the blend is
not its standalone score but how DIFFERENTLY it errs.

stages: nn | blend
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
NPZ = ROOT / "experiments" / "m36_preds.npz"
CAP, NN_CAP, MIN_TRAIN = 86400, 7200, 20000


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


def stage_nn():
    import torch
    import torch.nn as nn
    torch.manual_seed(42)
    torch.set_num_threads(12)
    X, meta, y, tr, va, es, ok, art = _load()
    cat_cols = [c for c in X.columns if X[c].dtype.name == "category"]
    num_cols = [c for c in X.columns if c not in cat_cols]

    C = np.zeros((len(X), len(cat_cols)), dtype=np.int64)
    cards = []
    for j, c in enumerate(cat_cols):
        codes = X[c].cat.codes.to_numpy() + 1        # 0 is the missing bucket
        C[:, j] = codes
        cards.append(int(codes.max()) + 1)
    N = X[num_cols].to_numpy(dtype=np.float32)
    miss = (~np.isfinite(N)).astype(np.float32)
    fit = ok & tr & ~es
    Nf = np.where(np.isfinite(N[fit]), N[fit], np.nan)
    mu = np.nanmean(Nf, axis=0)
    sd = np.nanstd(Nf, axis=0)
    sd = np.where(np.isfinite(sd) & (sd > 1e-6), sd, 1.0)
    mu = np.where(np.isfinite(mu), mu, 0.0)
    N = np.where(np.isfinite(N), (N - mu) / sd, 0.0).astype(np.float32)
    N = np.clip(N, -10, 10)
    N = np.hstack([N, miss])
    print("numeriques %d (+%d indicateurs) | categorielles %d" %
          (len(num_cols), miss.shape[1], len(cat_cols)), flush=True)

    yt = y.clip(upper=NN_CAP).to_numpy(dtype=np.float32)
    ym, ys = float(yt[fit].mean()), float(yt[fit].std())
    yz = (yt - ym) / ys

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.emb = nn.ModuleList([nn.Embedding(c, min(32, (c + 1) // 2)) for c in cards])
            d = sum(e.embedding_dim for e in self.emb) + N.shape[1]
            self.mlp = nn.Sequential(
                nn.Linear(d, 512), nn.BatchNorm1d(512), nn.ReLU(), nn.Dropout(0.15),
                nn.Linear(512, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.10),
                nn.Linear(256, 128), nn.BatchNorm1d(128), nn.ReLU(),
                nn.Linear(128, 1))

        def forward(self, xc, xn):
            e = [emb(xc[:, j]) for j, emb in enumerate(self.emb)]
            return self.mlp(torch.cat(e + [xn], 1)).squeeze(1)

    net = Net()
    opt = torch.optim.AdamW(net.parameters(), lr=2e-3, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=12)
    idx_fit = np.where(fit)[0]
    idx_es = np.where(ok & tr & es)[0]
    Xc_es = torch.from_numpy(C[idx_es])
    Xn_es = torch.from_numpy(N[idx_es])
    y_es = yt[idx_es]

    def predict(idx):
        net.eval()
        out = []
        with torch.no_grad():
            for i in range(0, len(idx), 16384):
                b = idx[i:i + 16384]
                out.append(net(torch.from_numpy(C[b]), torch.from_numpy(N[b])).numpy())
        return np.concatenate(out) * ys + ym

    best, best_state, patience, bs = 1e18, None, 0, 4096
    for ep in range(12):
        net.train()
        t0 = time.time()
        perm = np.random.permutation(idx_fit)
        tot = 0.0
        for i in range(0, len(perm), bs):
            b = perm[i:i + bs]
            opt.zero_grad()
            loss = nn.functional.mse_loss(
                net(torch.from_numpy(C[b]), torch.from_numpy(N[b])),
                torch.from_numpy(yz[b]))
            loss.backward()
            opt.step()
            tot += float(loss) * len(b)
        sched.step()
        r = float(np.sqrt(np.mean((predict(idx_es) - y_es) ** 2)))
        print("  epoque %2d  train_mse=%.4f  ES_rmse=%.1f  (%.0fs)" %
              (ep + 1, tot / len(perm), r, time.time() - t0), flush=True)
        if r < best - 0.5:
            best = r
            best_state = {k: v.clone() for k, v in net.state_dict().items()}
            patience = 0
        else:
            patience += 1
            if patience >= 3:
                break
    net.load_state_dict(best_state)
    np.savez(NPZ, nn_va=predict(np.where(va)[0]), nn_es=predict(idx_es), best_es=best)
    print("[nn] meilleur ES rmse=%.1f" % best, flush=True)


def stage_blend():
    import lightgbm as lgb
    from dc26.validation import rmse, rmse_report
    from dc26.mixture import REG_PARAMS
    from dc26 import log_experiment
    X, meta, y, tr, va, es, ok, art = _load()
    apt = X["AIRPORT"].astype(str).to_numpy()
    apt_va = apt[va]
    y_va = y[va].to_numpy()
    clean = ~art[va]
    vp = meta.loc[va, "val_part"].to_numpy()
    cat = [c for c in X.columns if X[c].dtype.name == "category"]
    d = np.load(NPZ)

    def tuned(a):
        p = ROOT / "experiments" / ("optuna_" + a + ".json")
        b = dict(objective="regression", metric="rmse", verbosity=-1, num_threads=0,
                 bagging_freq=1, seed=42)
        if not p.exists():
            return dict(REG_PARAMS)
        j = json.loads(p.read_text())
        b.update({k: v for k, v in j.items() if not k.startswith("_")})
        for k in ("num_leaves", "min_data_in_leaf", "max_bin"):
            b[k] = int(b[k])
        return b

    g_va = np.full(va.sum(), np.nan)
    for a in np.unique(apt):
        f, s = ok & tr & (apt == a) & ~es, ok & tr & (apt == a) & es
        sel = apt_va == a
        if f.sum() < MIN_TRAIN or not sel.any():
            continue
        m = lgb.train(tuned(a), lgb.Dataset(X[f], y[f].clip(upper=CAP), categorical_feature=cat),
                      num_boost_round=6000,
                      valid_sets=[lgb.Dataset(X[s], y[s].clip(upper=CAP))],
                      callbacks=[lgb.early_stopping(150, verbose=False)])
        g_va[sel] = m.predict(X[va][sel], num_iteration=m.best_iteration)
    g_va = np.clip(g_va, 60, CAP)
    np.save(ROOT / 'experiments' / 'm36_gbdt_va.npy', g_va)
    nn_va = np.clip(d["nn_va"], 60, CAP)

    print("GBDT seul : nettoye=%.1f | brut=%.1f" %
          (rmse(y_va[clean], g_va[clean]), rmse_report(y_va, g_va, vp)["overall"]), flush=True)
    print("NN seul   : nettoye=%.1f | brut=%.1f" %
          (rmse(y_va[clean], nn_va[clean]), rmse_report(y_va, nn_va, vp)["overall"]), flush=True)
    e1 = g_va[clean] - y_va[clean]
    e2 = nn_va[clean] - y_va[clean]
    print("correlation des erreurs GBDT/NN : %.3f" % np.corrcoef(e1, e2)[0, 1], flush=True)

    # fair comparison: the NN was trained on a target clipped at 2h, so judge it
    # where it can actually operate, and blend only in that regime
    bulk = clean & (y_va <= 7200)
    print("sur le GROS DU TRAFIC (y<=2h, %d lignes): GBDT=%.1f  NN=%.1f  corr=%.3f" % (
        bulk.sum(), rmse(y_va[bulk], g_va[bulk]), rmse(y_va[bulk], nn_va[bulk]),
        np.corrcoef(g_va[bulk] - y_va[bulk], nn_va[bulk] - y_va[bulk])[0, 1]), flush=True)

    best = None
    for w in np.arange(0, 0.55, 0.05):
        # regime-aware: the network only contributes where the trees stay in the bulk
        gate = (g_va <= 7200).astype(float)
        p = g_va + w * gate * (nn_va - g_va)
        c = rmse(y_va[clean], p[clean])
        print("  w_NN=%.2f : nettoye=%.1f | brut=%.1f | bulk=%.1f" %
              (w, c, rmse_report(y_va, p, vp)["overall"], rmse(y_va[bulk], p[bulk])), flush=True)
        if best is None or c < best[1]:
            best = (w, c)
    w = best[0]
    gate = (g_va <= 7200).astype(float)
    rep = rmse_report(y_va, g_va + w * gate * (nn_va - g_va), vp)
    print("\nmeilleur melange w_NN=%.2f -> nettoye=%.1f brut=%.1f" % (w, best[1], rep["overall"]), flush=True)
    log_experiment("M36 melange GBDT + reseau", model="LGBM + MLP embeddings",
                   features="full", val_rmse=rep["overall"], val_rmse_jan=rep["jan"],
                   val_rmse_jul=rep["jul"], notes="w_NN=%.2f nettoye=%.1f" % (w, best[1]),
                   keep="KEEP" if w > 0 else "DROP")


if __name__ == "__main__":
    {"nn": stage_nn, "blend": stage_blend}[sys.argv[1]]()
