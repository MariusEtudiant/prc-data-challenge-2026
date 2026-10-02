"""M37: FT-Transformer (Gorishniy et al. 2021) - attention over feature tokens.

Third inductive bias after trees and the MLP. Each feature - numerical or
categorical - becomes a token; a [CLS] token attends over them. Unlike an MLP it
models feature-to-feature interactions explicitly and adaptively, which is the
part trees do by hard splitting and MLPs by dense mixing.

Token budget is the cost driver, so only the strongest features are tokenised
(kept by LightGBM gain) plus every categorical.

stages: ftt | blend
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
NPZ = ROOT / "experiments" / "m37_preds.npz"
CAP, NN_CAP, MIN_TRAIN = 86400, 7200, 20000
N_NUM_TOKENS = 64          # strongest numerical features kept as tokens
D_TOKEN, N_LAYERS, N_HEADS = 96, 4, 8


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


def _top_numeric(X, y, fit, num_cols, k):
    """Rank numerical features by LightGBM gain on a subsample."""
    import lightgbm as lgb
    idx = np.where(fit)[0]
    if len(idx) > 300000:
        idx = np.random.RandomState(0).choice(idx, 300000, replace=False)
    m = lgb.train(dict(objective="regression", metric="rmse", num_leaves=127,
                       learning_rate=0.1, verbosity=-1, num_threads=12, seed=0),
                  lgb.Dataset(X.iloc[idx][num_cols], y.iloc[idx].clip(upper=NN_CAP)),
                  num_boost_round=200)
    imp = pd.Series(m.feature_importance("gain"), index=num_cols).sort_values(ascending=False)
    return list(imp.head(k).index)


def stage_ftt():
    import torch
    import torch.nn as nn
    torch.manual_seed(42)
    torch.set_num_threads(12)
    DEV = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('device:', DEV, flush=True)
    X, meta, y, tr, va, es, ok, art = _load()
    cat_cols = [c for c in X.columns if X[c].dtype.name == "category"]
    num_all = [c for c in X.columns if c not in cat_cols]
    fit = ok & tr & ~es
    num_cols = _top_numeric(X, y, fit, num_all, N_NUM_TOKENS)
    print("tokens: %d numeriques + %d categorielles + CLS = %d" %
          (len(num_cols), len(cat_cols), len(num_cols) + len(cat_cols) + 1), flush=True)

    C = np.zeros((len(X), len(cat_cols)), dtype=np.int64)
    cards = []
    for j, c in enumerate(cat_cols):
        codes = X[c].cat.codes.to_numpy() + 1
        C[:, j] = codes
        cards.append(int(codes.max()) + 1)
    N = X[num_cols].to_numpy(dtype=np.float32)
    Nf = np.where(np.isfinite(N[fit]), N[fit], np.nan)
    mu = np.nan_to_num(np.nanmean(Nf, axis=0))
    sd = np.nanstd(Nf, axis=0)
    sd = np.where(np.isfinite(sd) & (sd > 1e-6), sd, 1.0)
    N = np.clip(np.where(np.isfinite(N), (N - mu) / sd, 0.0), -10, 10).astype(np.float32)

    yt = y.clip(upper=NN_CAP).to_numpy(dtype=np.float32)
    ym, ys = float(yt[fit].mean()), float(yt[fit].std())
    yz = (yt - ym) / ys

    class FTT(nn.Module):
        def __init__(self, n_num, cards, d=D_TOKEN, nl=N_LAYERS, nh=N_HEADS):
            super().__init__()
            self.w_num = nn.Parameter(torch.randn(n_num, d) * 0.02)
            self.b_num = nn.Parameter(torch.zeros(n_num, d))
            self.emb = nn.ModuleList([nn.Embedding(c, d) for c in cards])
            self.cls = nn.Parameter(torch.randn(1, 1, d) * 0.02)
            layer = nn.TransformerEncoderLayer(d_model=d, nhead=nh, dim_feedforward=d * 4,
                                               dropout=0.1, batch_first=True,
                                               norm_first=True, activation="gelu")
            self.tr = nn.TransformerEncoder(layer, nl)
            self.head = nn.Sequential(nn.LayerNorm(d), nn.ReLU(), nn.Linear(d, 1))

        def forward(self, xc, xn):
            t_num = xn.unsqueeze(-1) * self.w_num + self.b_num          # B x n_num x d
            t_cat = torch.stack([e(xc[:, j]) for j, e in enumerate(self.emb)], 1)
            cls = self.cls.expand(xn.size(0), -1, -1)
            h = self.tr(torch.cat([cls, t_num, t_cat], 1))
            return self.head(h[:, 0]).squeeze(1)

    net = FTT(len(num_cols), cards).to(DEV)
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=15)
    idx_fit = np.where(fit)[0]
    idx_es = np.where(ok & tr & es)[0]
    y_es = yt[idx_es]

    def predict(idx, bs=16384):
        net.eval()
        out = []
        with torch.no_grad():
            for i in range(0, len(idx), bs):
                b = idx[i:i + bs]
                out.append(net(torch.from_numpy(C[b]).to(DEV), torch.from_numpy(N[b]).to(DEV)).cpu().numpy())
        return np.concatenate(out) * ys + ym

    best, best_state, patience, bs = 1e18, None, 0, 2048
    for ep in range(15):
        net.train()
        t0 = time.time()
        perm = np.random.permutation(idx_fit)
        tot = 0.0
        for i in range(0, len(perm), bs):
            b = perm[i:i + bs]
            opt.zero_grad()
            loss = nn.functional.mse_loss(
                net(torch.from_numpy(C[b]).to(DEV), torch.from_numpy(N[b]).to(DEV)),
                torch.from_numpy(yz[b]).to(DEV))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
            tot += float(loss) * len(b)
        sched.step()
        r = float(np.sqrt(np.mean((predict(idx_es) - y_es) ** 2)))
        print("  epoque %d  train_mse=%.4f  ES_rmse=%.1f  (%.0f min)" %
              (ep + 1, tot / len(perm), r, (time.time() - t0) / 60), flush=True)
        if r < best - 0.5:
            best = r
            best_state = {k: v.clone() for k, v in net.state_dict().items()}
            patience = 0
        else:
            patience += 1
            if patience >= 2:
                break
    net.load_state_dict(best_state)
    np.savez(NPZ, ftt_va=predict(np.where(va)[0]), ftt_es=predict(idx_es), best_es=best)
    print("[ftt] meilleur ES rmse=%.1f" % best, flush=True)


def stage_blend():
    from dc26.validation import rmse, rmse_report
    from dc26 import log_experiment
    X, meta, y, tr, va, es, ok, art = _load()
    y_va = y[va].to_numpy()
    clean = ~art[va]
    vp = meta.loc[va, "val_part"].to_numpy()
    ftt = np.clip(np.load(NPZ)["ftt_va"], 60, CAP)
    mlp = np.clip(np.load(ROOT / "experiments" / "m36_preds.npz")["nn_va"], 60, CAP)
    gbdt = np.clip(np.load(ROOT / "experiments" / "m36_gbdt_va.npy"), 60, CAP)

    for tag, p in [("GBDT", gbdt), ("MLP", mlp), ("FT-Transformer", ftt)]:
        print("%-15s nettoye=%.1f | brut=%.1f" % (
            tag, rmse(y_va[clean], p[clean]), rmse_report(y_va, p, vp)["overall"]), flush=True)
    e = {k: v[clean] - y_va[clean] for k, v in [("g", gbdt), ("m", mlp), ("f", ftt)]}
    print("correlations d'erreurs: GBDT/MLP=%.3f GBDT/FTT=%.3f MLP/FTT=%.3f" % (
        np.corrcoef(e["g"], e["m"])[0, 1], np.corrcoef(e["g"], e["f"])[0, 1],
        np.corrcoef(e["m"], e["f"])[0, 1]), flush=True)

    grid = np.arange(0, 0.45, 0.05)
    best = None
    for wm in grid:
        for wf in grid:
            if wm + wf > 0.6:
                continue
            p = (1 - wm - wf) * gbdt + wm * mlp + wf * ftt
            c = rmse(y_va[clean], p)
            if best is None or c < best[2]:
                best = (wm, wf, c)
    wm, wf, c = best
    rep = rmse_report(y_va, (1 - wm - wf) * gbdt + wm * mlp + wf * ftt, vp)
    print("\nmeilleur: w_MLP=%.2f w_FTT=%.2f -> nettoye=%.1f brut=%.1f" % (wm, wf, c, rep["overall"]), flush=True)
    log_experiment("M37 melange 3 familles (GBDT+MLP+FTT)", model="LGBM+MLP+FT-Transformer",
                   features="full", val_rmse=rep["overall"], val_rmse_jan=rep["jan"],
                   val_rmse_jul=rep["jul"], notes="wm=%.2f wf=%.2f nettoye=%.1f" % (wm, wf, c),
                   keep="KEEP" if (wm + wf) > 0 else "DROP")


if __name__ == "__main__":
    {"ftt": stage_ftt, "blend": stage_blend}[sys.argv[1]]()
