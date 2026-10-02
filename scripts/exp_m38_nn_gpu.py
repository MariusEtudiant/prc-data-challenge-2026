"""M38: the network that actually works, trained properly on GPU.

M36 showed a plain MLP reaching 225.7 on the bulk against 214.0 for the tuned
per-airport trees, with an error correlation of 0.868 - enough for the blend to
gain 2.6s. It was still improving when the CPU budget stopped it at 12 epochs.

Here: wider network, longer schedule, several seeds averaged (variance reduction
matters most in a blend), all on GPU. The FT-Transformer was dropped - 30 min per
epoch under memory pressure and tracking worse than this MLP at equal epochs.

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
NPZ = ROOT / "experiments" / "m38_preds.npz"
CAP, NN_CAP, MIN_TRAIN = 86400, 7200, 20000
SEEDS = [42, 1337, 2024]
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
    print("device:", dev, flush=True)
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

    yt = y.clip(upper=NN_CAP).to_numpy(dtype=np.float32)
    ym, ys = float(yt[fit].mean()), float(yt[fit].std())
    yz = ((yt - ym) / ys).astype(np.float32)

    # everything lives on the GPU: 2M x ~190 float32 is about 1.5 GB
    tC = torch.from_numpy(C).to(dev)
    tN = torch.from_numpy(N).to(dev)
    tY = torch.from_numpy(yz).to(dev)
    print("donnees sur GPU: %.1f Go" % ((tC.element_size() * tC.nelement()
                                         + tN.element_size() * tN.nelement()) / 1e9), flush=True)

    idx_fit = torch.from_numpy(np.where(fit)[0]).to(dev)
    idx_es = torch.from_numpy(np.where(ok & tr & es)[0]).to(dev)
    idx_va = torch.from_numpy(np.where(va)[0]).to(dev)
    y_es_np = yt[np.where(ok & tr & es)[0]]

    class Net(nn.Module):
        def __init__(self, seed):
            super().__init__()
            torch.manual_seed(seed)
            self.emb = nn.ModuleList([nn.Embedding(c, min(48, (c + 1) // 2)) for c in cards])
            d = sum(e.embedding_dim for e in self.emb) + N.shape[1]
            self.mlp = nn.Sequential(
                nn.Linear(d, 1024), nn.BatchNorm1d(1024), nn.SiLU(), nn.Dropout(0.15),
                nn.Linear(1024, 512), nn.BatchNorm1d(512), nn.SiLU(), nn.Dropout(0.10),
                nn.Linear(512, 256), nn.BatchNorm1d(256), nn.SiLU(), nn.Dropout(0.05),
                nn.Linear(256, 128), nn.BatchNorm1d(128), nn.SiLU(),
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

    P_va, P_es, scores = [], [], []
    for seed in SEEDS:
        net = Net(seed).to(dev)
        opt = torch.optim.AdamW(net.parameters(), lr=3e-3, weight_decay=1e-5)
        sched = torch.optim.lr_scheduler.OneCycleLR(
            opt, max_lr=3e-3, total_steps=EPOCHS * (len(idx_fit) // 8192 + 1), pct_start=0.2)
        best, best_state, patience, bs = 1e18, None, 0, 8192
        t0 = time.time()
        for ep in range(EPOCHS):
            net.train()
            perm = idx_fit[torch.randperm(len(idx_fit), device=dev)]
            for i in range(0, len(perm), bs):
                b = perm[i:i + bs]
                opt.zero_grad(set_to_none=True)
                loss = nn.functional.mse_loss(net(tC[b], tN[b]), tY[b])
                loss.backward()
                opt.step()
                sched.step()
            r = float(np.sqrt(np.mean((predict(net, idx_es) - y_es_np) ** 2)))
            if ep % 5 == 0 or ep == EPOCHS - 1:
                print("  seed %d epoque %2d  ES_rmse=%.1f  (%.0fs cumulees)" %
                      (seed, ep + 1, r, time.time() - t0), flush=True)
            if r < best - 0.3:
                best, best_state, patience = r, {k: v.clone() for k, v in net.state_dict().items()}, 0
            else:
                patience += 1
                if patience >= 6:
                    break
        net.load_state_dict(best_state)
        P_va.append(predict(net, idx_va))
        P_es.append(predict(net, idx_es))
        scores.append(best)
        print("  [seed %d] meilleur ES=%.1f en %.0fs" % (seed, best, time.time() - t0), flush=True)
    np.savez(NPZ, nn_va=np.mean(P_va, axis=0), nn_es=np.mean(P_es, axis=0),
             scores=np.array(scores))
    print("[nn] ES par seed: %s | moyenne des predictions sauvegardee" %
          np.round(scores, 1), flush=True)


def stage_blend():
    from dc26.validation import rmse, rmse_report
    from dc26 import log_experiment
    X, meta, y, tr, va, es, ok, art = _load()
    y_va = y[va].to_numpy()
    clean = ~art[va]
    bulk = clean & (y_va <= 7200)
    vp = meta.loc[va, "val_part"].to_numpy()
    g = np.clip(np.load(ROOT / "experiments" / "m36_gbdt_va.npy"), 60, CAP)
    nn_old = np.clip(np.load(ROOT / "experiments" / "m36_preds.npz")["nn_va"], 60, CAP)
    nn_new = np.clip(np.load(NPZ)["nn_va"], 60, CAP)

    for tag, p in [("GBDT", g), ("MLP v1 (CPU)", nn_old), ("MLP v2 (GPU, 3 seeds)", nn_new)]:
        print("%-24s nettoye=%.1f | bulk=%.1f | corr err/GBDT=%.3f" % (
            tag, rmse(y_va[clean], p[clean]), rmse(y_va[bulk], p[bulk]),
            np.corrcoef(g[clean] - y_va[clean], p[clean] - y_va[clean])[0, 1]), flush=True)

    gate = (g <= 7200).astype(float)
    best = None
    for w in np.arange(0, 0.65, 0.05):
        p = g + w * gate * (nn_new - g)
        c = rmse(y_va[clean], p[clean])
        print("  w_NN=%.2f : nettoye=%.1f | bulk=%.1f" % (w, c, rmse(y_va[bulk], p[bulk])), flush=True)
        if best is None or c < best[1]:
            best = (w, c)
    # three-way: GBDT + both networks
    best3 = None
    for w1 in np.arange(0, 0.5, 0.05):
        for w2 in np.arange(0, 0.5, 0.05):
            if w1 + w2 > 0.7:
                continue
            p = g + gate * (w1 * (nn_new - g) + w2 * (nn_old - g))
            c = rmse(y_va[clean], p[clean])
            if best3 is None or c < best3[2]:
                best3 = (w1, w2, c)
    w = best[0]
    rep = rmse_report(y_va, g + w * gate * (nn_new - g), vp)
    print("\nmelange 2 modeles : w_NN=%.2f -> nettoye=%.1f (leaderboard projete ~%.0f)" %
          (w, best[1], best[1] * 1.04), flush=True)
    print("melange 3 modeles : w_v2=%.2f w_v1=%.2f -> nettoye=%.1f (projete ~%.0f)" %
          (best3[0], best3[1], best3[2], best3[2] * 1.04), flush=True)
    log_experiment("M38 melange GBDT + MLP GPU multi-seed", model="LGBM + MLP x3 seeds",
                   features="full", val_rmse=rep["overall"], val_rmse_jan=rep["jan"],
                   val_rmse_jul=rep["jul"],
                   notes="nettoye %.1f (w=%.2f), 3-modeles %.1f" % (best[1], w, best3[2]),
                   keep="KEEP")


if __name__ == "__main__":
    {"nn": stage_nn, "blend": stage_blend}[sys.argv[1]]()
