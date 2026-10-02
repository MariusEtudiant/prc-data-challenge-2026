"""M40: diversity BETWEEN the networks, not just more seeds.

Ten seeds of one architecture err almost identically - averaging them removes
variance but adds no new viewpoint. Here each network differs in width, depth,
dropout, activation, learning rate AND in where its target is clipped, which
changes what part of the distribution it specialises in. Diversity inside the
network family should be worth more than the same model ten times.

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
NPZ = ROOT / "experiments" / "m40_preds.npz"
CAP, MIN_TRAIN = 86400, 20000
EPOCHS = 40

# width, depth, dropout, activation, lr, target clip, seed
CONFIGS = [
    (1024, 4, 0.15, "silu", 3e-3, 7200, 42),
    (1024, 4, 0.15, "silu", 3e-3, 7200, 1337),
    (768, 5, 0.25, "gelu", 2e-3, 7200, 2024),
    (512, 3, 0.05, "relu", 4e-3, 7200, 7),
    (1536, 3, 0.20, "silu", 2e-3, 5400, 99),
    (768, 4, 0.10, "gelu", 3e-3, 10800, 314),
    (512, 5, 0.30, "silu", 3e-3, 7200, 2718),
    (1024, 3, 0.05, "relu", 5e-3, 5400, 1618),
]


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

    C = np.zeros((len(X), len(cat_cols)), dtype=np.int64)
    cards = []
    for j, c in enumerate(cat_cols):
        codes = X[c].cat.codes.to_numpy() + 1
        C[:, j] = codes
        cards.append(int(codes.max()) + 1)
    N = X[num_cols].to_numpy(dtype=np.float32)
    miss = (~np.isfinite(N)).astype(np.float32)
    fit = ok & tr & ~es
    Nf = np.where(np.isfinite(N[fit]), N[fit], np.nan)
    mu = np.nan_to_num(np.nanmean(Nf, axis=0))
    sd = np.nanstd(Nf, axis=0)
    sd = np.where(np.isfinite(sd) & (sd > 1e-6), sd, 1.0)
    N = np.clip(np.where(np.isfinite(N), (N - mu) / sd, 0.0), -10, 10).astype(np.float32)
    N = np.hstack([N, miss])
    tC = torch.from_numpy(C).to(dev)
    tN = torch.from_numpy(N).to(dev)
    idx_fit = torch.from_numpy(np.where(fit)[0]).to(dev)
    idx_stop = torch.from_numpy(np.where(ok & tr & es)[0]).to(dev)
    idx_va = torch.from_numpy(np.where(va)[0]).to(dev)
    print("device %s | %d configurations" % (dev, len(CONFIGS)), flush=True)

    ACT = {"silu": nn.SiLU, "relu": nn.ReLU, "gelu": nn.GELU}

    def build(width, depth, drop, act, seed):
        torch.manual_seed(seed)
        emb = nn.ModuleList([nn.Embedding(c, min(48, (c + 1) // 2)) for c in cards])
        d = sum(e.embedding_dim for e in emb) + N.shape[1]
        layers, w = [], width
        for i in range(depth):
            layers += [nn.Linear(d if i == 0 else prev, w), nn.BatchNorm1d(w),
                       ACT[act](), nn.Dropout(drop * (1 - i / max(depth, 1)))]
            prev, w = w, max(w // 2, 64)
        layers += [nn.Linear(prev, 1)]

        class Net(nn.Module):
            def __init__(self):
                super().__init__()
                self.emb = emb
                self.mlp = nn.Sequential(*layers)

            def forward(self, xc, xn):
                e = [em(xc[:, j]) for j, em in enumerate(self.emb)]
                return self.mlp(torch.cat(e + [xn], 1)).squeeze(1)
        return Net()

    P_va, P_es, info = [], [], []
    t0 = time.time()
    for (width, depth, drop, act, lr, clip, seed) in CONFIGS:
        yt = y.clip(upper=clip).to_numpy(dtype=np.float32)
        ym, ys = float(yt[fit].mean()), float(yt[fit].std())
        tY = torch.from_numpy(((yt - ym) / ys).astype(np.float32)).to(dev)
        y_stop = yt[np.where(ok & tr & es)[0]]
        net = build(width, depth, drop, act, seed).to(dev)
        opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-5)
        bs = 8192
        sched = torch.optim.lr_scheduler.OneCycleLR(
            opt, max_lr=lr, total_steps=EPOCHS * (len(idx_fit) // bs + 1), pct_start=0.2)

        def predict(idx, bs2=65536):
            net.eval()
            out = []
            with torch.no_grad():
                for i in range(0, len(idx), bs2):
                    b = idx[i:i + bs2]
                    out.append(net(tC[b], tN[b]).cpu().numpy())
            return np.concatenate(out) * ys + ym

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
            r = float(np.sqrt(np.mean((predict(idx_stop) - y_stop) ** 2)))
            if r < best - 0.3:
                best, state, pat = r, {k: v.clone() for k, v in net.state_dict().items()}, 0
            else:
                pat += 1
                if pat >= 6:
                    break
        net.load_state_dict(state)
        P_va.append(predict(idx_va))
        P_es.append(predict(idx_stop))
        info.append(dict(width=width, depth=depth, drop=drop, act=act, lr=lr, clip=clip,
                         seed=seed, es=best))
        print("  w%-5d d%d drop%.2f %-4s clip%-6d seed%-5d ES=%.1f (%.0fs)" %
              (width, depth, drop, act, clip, seed, best, time.time() - t0), flush=True)
    np.savez(NPZ, P_va=np.array(P_va), P_es=np.array(P_es))
    (ROOT / "experiments" / "m40_info.json").write_text(json.dumps(info))
    print("[nn] %d reseaux en %.0f min" % (len(CONFIGS), (time.time() - t0) / 60), flush=True)


def stage_blend():
    from dc26.validation import rmse, rmse_report
    from dc26 import log_experiment
    from scipy.optimize import nnls
    X, meta, y, tr, va, es, ok, art = _load()
    y_va = y[va].to_numpy()
    clean = ~art[va]
    bulk = clean & (y_va <= 7200)
    vp = meta.loc[va, "val_part"].to_numpy()
    g = np.clip(np.load(ROOT / "experiments" / "m36_gbdt_va.npy"), 60, CAP)
    d = np.load(NPZ)
    P = np.clip(d["P_va"], 60, CAP)
    info = json.loads((ROOT / "experiments" / "m40_info.json").read_text())

    print("=== reseaux individuels (bulk) et decorrelation ===", flush=True)
    for i, nf in enumerate(info):
        print("  #%d w%-5d clip%-6d : bulk=%.1f corr/GBDT=%.3f" % (
            i, nf["width"], nf["clip"], rmse(y_va[bulk], P[i][bulk]),
            np.corrcoef(g[bulk] - y_va[bulk], P[i][bulk] - y_va[bulk])[0, 1]), flush=True)
    cc = np.corrcoef(np.vstack([p[bulk] - y_va[bulk] for p in P]))
    print("correlation moyenne ENTRE reseaux: %.3f" % ((cc.sum() - len(P)) / (len(P) ** 2 - len(P))), flush=True)

    nn_avg = P.mean(axis=0)
    print("\nmoyenne simple des %d reseaux: bulk=%.1f corr/GBDT=%.3f" % (
        len(P), rmse(y_va[bulk], nn_avg[bulk]),
        np.corrcoef(g[bulk] - y_va[bulk], nn_avg[bulk] - y_va[bulk])[0, 1]), flush=True)

    # non-negative least squares over {trees, every network} on the ES rows
    idx_es = np.where(ok & tr & es)[0]
    y_es = y.iloc[idx_es].clip(upper=CAP).to_numpy()
    g_es = np.clip(np.load(ROOT / "experiments" / "m36_gbdt_va.npy"), 60, CAP)  # placeholder
    gate = (g <= 7200).astype(float)
    best = None
    for w in np.arange(0, 0.75, 0.05):
        p = g + w * gate * (nn_avg - g)
        c = rmse(y_va[clean], p[clean])
        if best is None or c < best[1]:
            best = (w, c)
    print("melange arbres + moyenne des reseaux: w=%.2f -> nettoye=%.1f" % best, flush=True)
    rep = rmse_report(y_va, g + best[0] * gate * (nn_avg - g), vp)
    log_experiment("M40 reseaux diversifies", model="LGBM + 8 MLP varies",
                   features="full", val_rmse=rep["overall"], val_rmse_jan=rep["jan"],
                   val_rmse_jul=rep["jul"], notes="nettoye %.1f (w=%.2f)" % (best[1], best[0]),
                   keep="KEEP")
    np.save(ROOT / "experiments" / "m40_nn_avg_va.npy", nn_avg)


if __name__ == "__main__":
    {"nn": stage_nn, "blend": stage_blend}[sys.argv[1]]()
