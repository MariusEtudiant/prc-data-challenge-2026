"""Submission v11 = v10's tree ensemble + the neural networks (M39).

The tree part is reused as-is from the v10 run (experiments/v10_rank_preds.npz), so
only the networks are trained here, on the full 2025 year: a global one averaged
over 10 seeds and one per airport over 3 seeds. Blend weights per airport come from
the mirror fit (experiments/m39_nn_weights.json), applied through the same gate:
networks only contribute where the trees stay in the ordinary regime.

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
NPZ = ROOT / "experiments" / "v11_nn_preds.npz"
CAP, NN_CAP, MIN_TRAIN = 86400, 7200, 20000
SEEDS_GLOBAL = [42, 1337, 2024, 7, 99, 314, 2718, 1618, 577, 8128]
SEEDS_AIRPORT = [42, 1337, 2024]
EPOCHS = 40

TRAIN_FILES = ["X_v2_full_t.parquet", "flightctx_train2025.parquet",
               "depregime_train2025.parquet", "aobtq_full2025.parquet",
               "anomreg_train2025.parquet"]
RANK_FILES = ["X_v2_ranking_t.parquet", "flightctx_ranking2026.parquet",
              "depregime_ranking2026.parquet", "aobtq_ranking2026.parquet",
              "anomreg_ranking2026.parquet"]


def _data():
    from dc26.pipeline import align_categories
    X = pd.concat([pd.read_parquet(PROC / f) for f in TRAIN_FILES], axis=1)
    Xr = pd.concat([pd.read_parquet(PROC / f) for f in RANK_FILES], axis=1)
    Xr = align_categories(Xr, X)
    meta = pd.read_parquet(PROC / "meta_train2025.parquet")
    y = meta["y"].astype("float64")
    keep = (y > 0).to_numpy() & (y < 200000).to_numpy()
    art = np.zeros(len(y), bool)
    for k in (1, 2, 3):
        art |= (y - k * 86400).abs().to_numpy() < 3600
    keep &= ~art
    es = meta["es"].to_numpy()
    return X, Xr, y, keep, es


def stage_nn():
    import torch
    import torch.nn as nn
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    X, Xr, y, keep, es = _data()
    cat_cols = [c for c in X.columns if X[c].dtype.name == "category"]
    num_cols = [c for c in X.columns if c not in cat_cols]
    n_tr, n_rk = len(X), len(Xr)
    apt = X["AIRPORT"].astype(str).to_numpy()
    apt_r = Xr["AIRPORT"].astype(str).to_numpy()

    # categorical codes share the training categories (align_categories guarantees it)
    C = np.zeros((n_tr + n_rk, len(cat_cols)), dtype=np.int64)
    cards = []
    for j, c in enumerate(cat_cols):
        a = X[c].cat.codes.to_numpy() + 1
        b = Xr[c].cat.codes.to_numpy() + 1
        C[:n_tr, j], C[n_tr:, j] = a, b
        cards.append(int(max(a.max(), b.max())) + 1)
    Nall = np.vstack([X[num_cols].to_numpy(dtype=np.float32),
                      Xr[num_cols].to_numpy(dtype=np.float32)])
    miss = (~np.isfinite(Nall)).astype(np.float32)
    fit_all = np.zeros(n_tr + n_rk, bool)
    fit_all[:n_tr] = keep & ~es
    Nf = np.where(np.isfinite(Nall[fit_all]), Nall[fit_all], np.nan)
    mu = np.nan_to_num(np.nanmean(Nf, axis=0))
    sd = np.nanstd(Nf, axis=0)
    sd = np.where(np.isfinite(sd) & (sd > 1e-6), sd, 1.0)
    Nall = np.clip(np.where(np.isfinite(Nall), (Nall - mu) / sd, 0.0), -10, 10).astype(np.float32)
    Nall = np.hstack([Nall, miss])

    yt = np.zeros(n_tr + n_rk, dtype=np.float32)
    yt[:n_tr] = y.clip(upper=NN_CAP).to_numpy()
    ym, ys = float(yt[fit_all].mean()), float(yt[fit_all].std())
    tC = torch.from_numpy(C).to(dev)
    tN = torch.from_numpy(Nall).to(dev)
    tY = torch.from_numpy(((yt - ym) / ys).astype(np.float32)).to(dev)
    print("device %s | %.1f Go sur GPU | %d colonnes" % (
        dev, (tC.nelement() * 8 + tN.nelement() * 4) / 1e9, Nall.shape[1]), flush=True)

    class Net(nn.Module):
        def __init__(self, seed, width=1024):
            super().__init__()
            torch.manual_seed(seed)
            self.emb = nn.ModuleList([nn.Embedding(c, min(48, (c + 1) // 2)) for c in cards])
            d = sum(e.embedding_dim for e in self.emb) + Nall.shape[1]
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

    def train_one(seed, fit_mask, stop_mask, width, bs=8192):
        idx_fit = torch.from_numpy(np.where(fit_mask)[0]).to(dev)
        idx_stop = torch.from_numpy(np.where(stop_mask)[0]).to(dev)
        y_stop = yt[np.where(stop_mask)[0]]
        net = Net(seed, width).to(dev)
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
                if pat >= 6:
                    break
        net.load_state_dict(state)
        return net, best

    idx_rank = torch.from_numpy(np.arange(n_tr, n_tr + n_rk)).to(dev)
    stop_all = np.zeros(n_tr + n_rk, bool)
    stop_all[:n_tr] = keep & es

    t0 = time.time()
    P = []
    for s in SEEDS_GLOBAL:
        net, b = train_one(s, fit_all, stop_all, 1024)
        P.append(predict(net, idx_rank))
        print("  [global %4d] ES=%.1f (%.0fs)" % (s, b, time.time() - t0), flush=True)
    nn_global = np.mean(P, axis=0)

    nn_airport = np.full(n_rk, np.nan)
    for a in np.unique(apt):
        f = fit_all.copy(); f[:n_tr] &= (apt == a)
        st = stop_all.copy(); st[:n_tr] &= (apt == a)
        sel = apt_r == a
        if f.sum() < MIN_TRAIN or st.sum() < 500 or not sel.any():
            continue
        t1 = time.time()
        Pa = []
        for s in SEEDS_AIRPORT:
            net, b = train_one(s, f, st, 512, bs=4096)
            Pa.append(predict(net, idx_rank)[sel])
        nn_airport[sel] = np.mean(Pa, axis=0)
        print("  [%s] n_train=%d (%.0fs)" % (a, f.sum(), time.time() - t1), flush=True)

    np.savez(NPZ, nn_global=nn_global, nn_airport=nn_airport)
    print("[nn] termine en %.0f min" % ((time.time() - t0) / 60), flush=True)


def stage_blend():
    X, Xr, y, keep, es = _data()
    apt_r = Xr["AIRPORT"].astype(str).to_numpy()
    d10 = np.load(ROOT / "experiments" / "v10_rank_preds.npz", allow_pickle=True)
    w_all = json.loads((ROOT / "experiments" / "m26_weights.json").read_text())
    P = np.vstack([d10["air_lgbm"], d10["air_cb"], d10["glob_lgbm"]])
    g = np.full(len(Xr), np.nan)
    for a, w in w_all.items():
        r = apt_r == a
        if not r.any():
            continue
        w = np.array(w, dtype=float)
        okc = np.all(np.isfinite(P[:, r]), axis=1)
        if w[~okc].sum() > 0:
            w = np.where(okc, w, 0.0)
            w = w / w.sum() if w.sum() > 0 else np.array([0.0, 0.0, 1.0])
        g[r] = w @ P[:, r]
    miss = ~np.isfinite(g)
    g[miss] = d10["glob_lgbm"][miss]
    g = np.clip(g, 60, CAP)

    dn = np.load(NPZ)
    ng = np.clip(dn["nn_global"], 60, CAP)
    na = np.clip(np.where(np.isfinite(dn["nn_airport"]), dn["nn_airport"], dn["nn_global"]), 60, CAP)
    wnn = json.loads((ROOT / "experiments" / "m39_nn_weights.json").read_text())

    gate = (g <= 7200).astype(float)
    pred = g.copy()
    for a, (w1, w2) in wnn.items():
        r = apt_r == a
        if not r.any():
            continue
        pred[r] = (g + gate * (w1 * (ng - g) + w2 * (na - g)))[r]
        print("  [%s] w_global=%.2f w_aeroport=%.2f | n=%d" % (a, w1, w2, int(r.sum())), flush=True)
    pred = np.clip(pred, 60, CAP)
    print("ecart moyen absolu vs v10 (arbres seuls): %.0fs" % np.mean(np.abs(pred - g)), flush=True)

    ids = pd.read_parquet(PROC / "rank_ids_t.parquet")["MVT_ID_mvt"]
    tpl = pd.read_parquet(ROOT / "data" / "submitting.parquet").drop(columns=["TAXITIME_SEC_mvt"])
    out = tpl.merge(pd.DataFrame({"MVT_ID_mvt": ids.values, "TAXITIME_SEC_mvt": pred}),
                    on="MVT_ID_mvt", how="left")
    assert out["TAXITIME_SEC_mvt"].notna().all() and len(out) == 215876
    out.to_parquet(ROOT / "submissions" / "reliable-hamburger_v11.parquet", index=False)
    t = out["TAXITIME_SEC_mvt"]
    print("v11 ecrite: mean=%.0f median=%.0f q99=%.0f | >1h=%d >2h=%d" % (
        t.mean(), t.median(), t.quantile(.99), (t > 3600).sum(), (t > 7200).sum()), flush=True)


if __name__ == "__main__":
    {"nn": stage_nn, "blend": stage_blend}[sys.argv[1]]()
