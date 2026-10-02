"""M39: the network path pushed as far as the GPU allows.

Two extensions over M38:
  - global network averaged over 10 seeds instead of 3 (variance reduction is what
    buys weight in a blend),
  - one network per airport, since LIRF and LFPG carry two thirds of the error and
    a global network has to compromise between very different airports.

Blend weights are then fitted per airport over {trees, global net, airport net}.

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
NPZ = ROOT / "experiments" / "m39_preds.npz"
CAP, NN_CAP, MIN_TRAIN = 86400, 7200, 20000
SEEDS_GLOBAL = [42, 1337, 2024, 7, 99, 314, 2718, 1618, 577, 8128]
SEEDS_AIRPORT = [42, 1337, 2024]
EPOCHS = 40


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
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    X, meta, y, tr, va, es, ok, art = _load()
    cat_cols = [c for c in X.columns if X[c].dtype.name == "category"]
    num_cols = [c for c in X.columns if c not in cat_cols]
    apt = X["AIRPORT"].astype(str).to_numpy()

    C = np.zeros((len(X), len(cat_cols)), dtype=np.int64)
    cards = []
    for j, c in enumerate(cat_cols):
        codes = X[c].cat.codes.to_numpy() + 1
        C[:, j] = codes
        cards.append(int(codes.max()) + 1)
    N = X[num_cols].to_numpy(dtype=np.float32)
    miss = (~np.isfinite(N)).astype(np.float32)
    fit_all = ok & tr & ~es
    Nf = np.where(np.isfinite(N[fit_all]), N[fit_all], np.nan)
    mu = np.nan_to_num(np.nanmean(Nf, axis=0))
    sd = np.nanstd(Nf, axis=0)
    sd = np.where(np.isfinite(sd) & (sd > 1e-6), sd, 1.0)
    N = np.clip(np.where(np.isfinite(N), (N - mu) / sd, 0.0), -10, 10).astype(np.float32)
    N = np.hstack([N, miss])

    yt = y.clip(upper=NN_CAP).to_numpy(dtype=np.float32)
    ym, ys = float(yt[fit_all].mean()), float(yt[fit_all].std())
    tC = torch.from_numpy(C).to(dev)
    tN = torch.from_numpy(N).to(dev)
    tY = torch.from_numpy(((yt - ym) / ys).astype(np.float32)).to(dev)
    print("device %s | donnees GPU %.1f Go" % (dev, (tC.nelement() * 8 + tN.nelement() * 4) / 1e9), flush=True)

    class Net(nn.Module):
        def __init__(self, seed, width=1024):
            super().__init__()
            torch.manual_seed(seed)
            self.emb = nn.ModuleList([nn.Embedding(c, min(48, (c + 1) // 2)) for c in cards])
            d = sum(e.embedding_dim for e in self.emb) + N.shape[1]
            w = width
            self.mlp = nn.Sequential(
                nn.Linear(d, w), nn.BatchNorm1d(w), nn.SiLU(), nn.Dropout(0.15),
                nn.Linear(w, w // 2), nn.BatchNorm1d(w // 2), nn.SiLU(), nn.Dropout(0.10),
                nn.Linear(w // 2, w // 4), nn.BatchNorm1d(w // 4), nn.SiLU(), nn.Dropout(0.05),
                nn.Linear(w // 4, 128), nn.BatchNorm1d(128), nn.SiLU(),
                nn.Linear(128, 1))

        def forward(self, xc, xn):
            e = [emb(xc[:, j]) for j, emb in enumerate(self.emb)]
            return self.mlp(torch.cat(e + [xn], 1)).squeeze(1)

    def predict(net, idx, bs=65536):
        net.eval()
        out = []
        with torch.no_grad():
            for i in range(0, len(idx), bs):
                b = idx[i:i + bs]
                out.append(net(tC[b], tN[b]).cpu().numpy())
        return np.concatenate(out) * ys + ym

    def train_one(seed, fit_mask, stop_mask, width, epochs, bs=8192):
        idx_fit = torch.from_numpy(np.where(fit_mask)[0]).to(dev)
        idx_stop = torch.from_numpy(np.where(stop_mask)[0]).to(dev)
        y_stop = yt[np.where(stop_mask)[0]]
        net = Net(seed, width).to(dev)
        opt = torch.optim.AdamW(net.parameters(), lr=3e-3, weight_decay=1e-5)
        steps = epochs * (len(idx_fit) // bs + 1)
        sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=3e-3, total_steps=steps, pct_start=0.2)
        best, state, pat = 1e18, None, 0
        for ep in range(epochs):
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
                if pat >= 6:
                    break
        net.load_state_dict(state)
        return net, best

    idx_va = torch.from_numpy(np.where(va)[0]).to(dev)
    stop_all = ok & tr & es

    # ---- global network, 10 seeds
    t0 = time.time()
    Pg, sg = [], []
    for s in SEEDS_GLOBAL:
        net, b = train_one(s, fit_all, stop_all, 1024, EPOCHS)
        Pg.append(predict(net, idx_va))
        sg.append(b)
        print("  [global seed %4d] ES=%.1f  (%.0fs cumulees)" % (s, b, time.time() - t0), flush=True)
    nn_global = np.mean(Pg, axis=0)

    # ---- one network per airport
    nn_airport = np.full(va.sum(), np.nan)
    apt_va = apt[va]
    for a in np.unique(apt):
        f, st = fit_all & (apt == a), stop_all & (apt == a)
        sel = apt_va == a
        if f.sum() < MIN_TRAIN or st.sum() < 500 or not sel.any():
            continue
        t1 = time.time()
        P = []
        for s in SEEDS_AIRPORT:
            net, b = train_one(s, f, st, 512, EPOCHS, bs=4096)
            P.append(predict(net, idx_va)[sel])
        nn_airport[sel] = np.mean(P, axis=0)
        print("  [%s] %d graines, n_train=%d  (%.0fs)" % (a, len(SEEDS_AIRPORT), f.sum(), time.time() - t1), flush=True)

    np.savez(NPZ, nn_global=nn_global, nn_airport=nn_airport, scores_global=np.array(sg))
    print("[nn] termine en %.0f min" % ((time.time() - t0) / 60), flush=True)


def stage_blend():
    from dc26.validation import rmse, rmse_report
    from dc26 import log_experiment
    X, meta, y, tr, va, es, ok, art = _load()
    apt_va = X["AIRPORT"].astype(str).to_numpy()[va]
    y_va = y[va].to_numpy()
    clean = ~art[va]
    bulk = clean & (y_va <= 7200)
    vp = meta.loc[va, "val_part"].to_numpy()
    g = np.clip(np.load(ROOT / "experiments" / "m36_gbdt_va.npy"), 60, CAP)
    d = np.load(NPZ)
    ng = np.clip(d["nn_global"], 60, CAP)
    na = np.clip(np.where(np.isfinite(d["nn_airport"]), d["nn_airport"], d["nn_global"]), 60, CAP)

    for tag, p in [("GBDT", g), ("NN global x10", ng), ("NN par aeroport", na)]:
        print("%-18s nettoye=%.1f | bulk=%.1f | corr/GBDT=%.3f" % (
            tag, rmse(y_va[clean], p[clean]), rmse(y_va[bulk], p[bulk]),
            np.corrcoef(g[clean] - y_va[clean], p[clean] - y_va[clean])[0, 1]), flush=True)

    gate = (g <= 7200).astype(float)
    grid = np.arange(0, 0.65, 0.05)
    weights = {}
    pred = g.copy()
    for a in np.unique(apt_va):
        m = (apt_va == a) & clean
        best = None
        for w1 in grid:
            for w2 in grid:
                if w1 + w2 > 0.75:
                    continue
                p = g + gate * (w1 * (ng - g) + w2 * (na - g))
                c = rmse(y_va[m], p[m])
                if best is None or c < best[2]:
                    best = (w1, w2, c)
        w1, w2, c = best
        weights[a] = [float(w1), float(w2)]
        sel = apt_va == a
        pred[sel] = (g + gate * (w1 * (ng - g) + w2 * (na - g)))[sel]
        print("  [%s] w_global=%.2f w_aeroport=%.2f -> %.1f (etait %.1f)" % (
            a, w1, w2, c, rmse(y_va[m], g[m])), flush=True)
    (ROOT / "experiments" / "m39_nn_weights.json").write_text(json.dumps(weights))
    rep = rmse_report(y_va, pred, vp)
    print("\nMELANGE FINAL: nettoye=%.1f | brut=%.1f | bulk=%.1f" % (
        rmse(y_va[clean], pred[clean]), rep["overall"], rmse(y_va[bulk], pred[bulk])), flush=True)
    log_experiment("M39 arbres + reseaux (global x10 + par aeroport)",
                   model="LGBM + MLP", features="full",
                   val_rmse=rep["overall"], val_rmse_jan=rep["jan"], val_rmse_jul=rep["jul"],
                   notes="nettoye %.1f" % rmse(y_va[clean], pred[clean]), keep="KEEP")


if __name__ == "__main__":
    {"nn": stage_nn, "blend": stage_blend}[sys.argv[1]]()
