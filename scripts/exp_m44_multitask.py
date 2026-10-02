"""M44: multi-task learning - the 2 million labelled arrivals we have been throwing away.

We train on 2.08M departures and ignore 2.08M arrivals whose taxi-in times ARE
labelled, in 2025 and in the 2026 ranking file alike. Arrivals share everything
structural with departures: which stand is far from which runway, which taxi routes
are slow, how an airport behaves on a given day. A shared trunk can learn that
geometry from twice the data, and the 2026 arrivals inject test-period information
into the learned representation legitimately.

Design: shared embeddings + trunk, one output head per phase, arrivals down-weighted
so the easier task does not capture the capacity. Departure-only features (d_aobt3,
queue, turnaround, departure regime) are NaN for arrivals; the missing-indicators and
the phase flag let the network condition on it.

stages: feats | nn | blend
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
NPZ = ROOT / "experiments" / "m44_preds.npz"
ARR_X = PROC / "X_arrivals_multitask.parquet"
CAP, NN_CAP = 86400, 7200
ARR_WEIGHT = 0.3
EPOCHS = 30
SEEDS = [42, 1337, 2024]


def _dep():
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


def stage_feats():
    """Build the arrival rows on the shared part of the feature space (one year at a
    time, written to disk, memory freed in between)."""
    import gc
    from dc26 import load_movements_2025, load_ranking_2026
    from dc26.weather import build_weather_features, load_metar
    from dc26.pru_features import build_pru_features
    from dc26.anomaly_regime import build_anomaly_regime
    from dc26.features import build_features_v1

    Xd, *_ = _dep()
    cols_d = list(Xd.columns)
    cats = {c: Xd[c].cat.categories for c in cols_d if Xd[c].dtype.name == "category"}
    del Xd
    gc.collect()
    met = load_metar()

    parts = []
    for tag, loader in [("2025", load_movements_2025), ("2026", load_ranking_2026)]:
        mv = loader()
        arr = mv[mv["PHASE_mvt"] == "ARR"].copy()
        arr = arr[arr["TAXITIME_SEC_mvt"].between(30, 7200)]
        base = build_features_v1(arr)
        for c in ["d_aobt3", "d_eobt1", "d_lobt", "d_iobt", "d_aobt3_vs_eobt1"]:
            if c in base:
                base[c] = np.nan
        A = pd.concat([base, build_weather_features(arr, met), build_pru_features(arr),
                       build_anomaly_regime(arr, mv)], axis=1)
        del base
        gc.collect()
        A["phase_dep"] = 0.0
        A["taxi"] = arr["TAXITIME_SEC_mvt"].to_numpy(dtype="float32")
        del arr, mv
        gc.collect()
        for c in cols_d:
            if c not in A.columns:
                A[c] = np.nan
            elif c in cats:
                A[c] = pd.Categorical(A[c].astype(str), categories=cats[c])
        for c in A.columns:
            if A[c].dtype == "float64":
                A[c] = A[c].astype("float32")
        A = A[cols_d + ["phase_dep", "taxi"]].reset_index(drop=True)
        out = PROC / ("X_arrivals_" + tag + ".parquet")
        A.to_parquet(out, index=False)
        parts.append(out)
        print("[feats] arrivees %s: %s -> %s" % (tag, A.shape, out.name), flush=True)
        del A
        gc.collect()
    print("[feats] termine", flush=True)


def stage_nn():
    import torch
    import torch.nn as nn
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    Xd, meta, y, tr, va, es, ok, art = _dep()
    A = pd.concat([pd.read_parquet(PROC / "X_arrivals_2025.parquet"),
                   pd.read_parquet(PROC / "X_arrivals_2026.parquet")],
                  ignore_index=True)
    cols = list(Xd.columns)
    cat_cols = [c for c in cols if Xd[c].dtype.name == "category"]
    num_cols = [c for c in cols if c not in cat_cols]

    n_d, n_a = len(Xd), len(A)
    C = np.zeros((n_d + n_a, len(cat_cols)), dtype=np.int64)
    cards = []
    for j, c in enumerate(cat_cols):
        a = Xd[c].cat.codes.to_numpy() + 1
        b = A[c].cat.codes.to_numpy() + 1 if A[c].dtype.name == "category" else np.zeros(n_a, int)
        C[:n_d, j], C[n_d:, j] = a, b
        cards.append(int(max(a.max(), b.max())) + 1)
    N = np.vstack([Xd[num_cols].to_numpy(dtype=np.float32),
                   A[num_cols].to_numpy(dtype=np.float32)])
    phase = np.concatenate([np.ones(n_d, np.float32), np.zeros(n_a, np.float32)])
    miss = (~np.isfinite(N)).astype(np.float32)

    fit_d = np.zeros(n_d + n_a, bool); fit_d[:n_d] = ok & tr & ~es
    Nf = np.where(np.isfinite(N[fit_d]), N[fit_d], np.nan)
    mu = np.nan_to_num(np.nanmean(Nf, axis=0))
    sd = np.nanstd(Nf, axis=0)
    sd = np.where(np.isfinite(sd) & (sd > 1e-6), sd, 1.0)
    N = np.clip(np.where(np.isfinite(N), (N - mu) / sd, 0.0), -10, 10).astype(np.float32)
    N = np.hstack([N, miss, phase[:, None]])

    yt = np.concatenate([y.clip(upper=NN_CAP).to_numpy(dtype=np.float32),
                         np.clip(A["taxi"].to_numpy(dtype=np.float32), 0, NN_CAP)])
    ym, ys = float(yt[fit_d].mean()), float(yt[fit_d].std())
    is_dep = torch.from_numpy(phase).to(dev)
    tC = torch.from_numpy(C).to(dev)
    tN = torch.from_numpy(N).to(dev)
    tY = torch.from_numpy(((yt - ym) / ys).astype(np.float32)).to(dev)
    print("device %s | departs %d + arrivees %d = %d lignes | %.1f Go" % (
        dev, n_d, n_a, n_d + n_a, (tC.nelement() * 8 + tN.nelement() * 4) / 1e9), flush=True)

    fit_arr = np.zeros(n_d + n_a, bool)
    fit_arr[n_d:] = True                       # every arrival, 2025 and 2026
    idx_fit = torch.from_numpy(np.where(fit_d | fit_arr)[0]).to(dev)
    idx_stop = torch.from_numpy(np.where(np.concatenate(
        [ok & tr & es, np.zeros(n_a, bool)]))[0]).to(dev)
    idx_va = torch.from_numpy(np.where(np.concatenate([va, np.zeros(n_a, bool)]))[0]).to(dev)
    y_stop = yt[np.where(np.concatenate([ok & tr & es, np.zeros(n_a, bool)]))[0]]

    class Net(nn.Module):
        def __init__(self, seed):
            super().__init__()
            torch.manual_seed(seed)
            self.emb = nn.ModuleList([nn.Embedding(c, min(48, (c + 1) // 2)) for c in cards])
            d = sum(e.embedding_dim for e in self.emb) + N.shape[1]
            self.trunk = nn.Sequential(
                nn.Linear(d, 1024), nn.BatchNorm1d(1024), nn.SiLU(), nn.Dropout(0.15),
                nn.Linear(1024, 512), nn.BatchNorm1d(512), nn.SiLU(), nn.Dropout(0.10),
                nn.Linear(512, 256), nn.BatchNorm1d(256), nn.SiLU())
            self.head_dep = nn.Sequential(nn.Linear(256, 128), nn.SiLU(), nn.Linear(128, 1))
            self.head_arr = nn.Sequential(nn.Linear(256, 128), nn.SiLU(), nn.Linear(128, 1))

        def forward(self, xc, xn, dep_mask):
            e = [em(xc[:, j]) for j, em in enumerate(self.emb)]
            h = self.trunk(torch.cat(e + [xn], 1))
            return (dep_mask * self.head_dep(h).squeeze(1)
                    + (1 - dep_mask) * self.head_arr(h).squeeze(1))

    P_va, scores = [], []
    t0 = time.time()
    for seed in SEEDS:
        net = Net(seed).to(dev)
        opt = torch.optim.AdamW(net.parameters(), lr=3e-3, weight_decay=1e-5)
        bs = 8192
        sched = torch.optim.lr_scheduler.OneCycleLR(
            opt, max_lr=3e-3, total_steps=EPOCHS * (len(idx_fit) // bs + 1), pct_start=0.2)

        def predict(idx, bs2=65536):
            net.eval()
            out = []
            with torch.no_grad():
                for i in range(0, len(idx), bs2):
                    b = idx[i:i + bs2]
                    out.append(net(tC[b], tN[b], is_dep[b]).cpu().numpy())
            return np.concatenate(out) * ys + ym

        best, state, pat = 1e18, None, 0
        for ep in range(EPOCHS):
            net.train()
            perm = idx_fit[torch.randperm(len(idx_fit), device=dev)]
            for i in range(0, len(perm), bs):
                b = perm[i:i + bs]
                opt.zero_grad(set_to_none=True)
                pred = net(tC[b], tN[b], is_dep[b])
                w = is_dep[b] + (1 - is_dep[b]) * ARR_WEIGHT
                loss = (w * (pred - tY[b]) ** 2).mean()
                loss.backward()
                opt.step()
                sched.step()
            r = float(np.sqrt(np.mean((predict(idx_stop) - y_stop) ** 2)))
            if r < best - 0.3:
                best, state, pat = r, {k: v.clone() for k, v in net.state_dict().items()}, 0
            else:
                pat += 1
                if pat >= 5:
                    break
        net.load_state_dict(state)
        P_va.append(predict(idx_va))
        scores.append(best)
        print("  [seed %d] ES=%.1f (%.0fs)" % (seed, best, time.time() - t0), flush=True)
    np.savez(NPZ, nn_va=np.mean(P_va, axis=0), scores=np.array(scores))
    print("[nn] ES par seed %s" % np.round(scores, 1), flush=True)


def stage_blend():
    from dc26.validation import rmse, rmse_report
    from dc26 import log_experiment
    Xd, meta, y, tr, va, es, ok, art = _dep()
    y_va = y[va].to_numpy()
    clean = ~art[va]
    bulk = clean & (y_va <= 7200)
    vp = meta.loc[va, "val_part"].to_numpy()
    g = np.clip(np.load(ROOT / "experiments" / "m36_gbdt_va.npy"), 60, CAP)
    mono = np.clip(np.load(NPZ)["nn_va"], 60, CAP)
    solo = np.clip(np.load(ROOT / "experiments" / "m39_preds.npz")["nn_global"], 60, CAP)

    for tag, p in [("GBDT", g), ("NN mono-tache", solo), ("NN MULTI-TACHE", mono)]:
        print("%-18s bulk=%.1f | corr/GBDT=%.3f" % (
            tag, rmse(y_va[bulk], p[bulk]),
            np.corrcoef(g[bulk] - y_va[bulk], p[bulk] - y_va[bulk])[0, 1]), flush=True)
    print("correlation NN mono / NN multi: %.3f" % np.corrcoef(
        solo[bulk] - y_va[bulk], mono[bulk] - y_va[bulk])[0, 1], flush=True)

    gate = (g <= 7200).astype(float)
    for tag, nnp in [("mono-tache", solo), ("multi-tache", mono)]:
        best = min(((w, rmse(y_va[clean], (g + w * gate * (nnp - g))[clean]))
                    for w in np.arange(0, 0.7, 0.05)), key=lambda t: t[1])
        print("  arbres + %s : w=%.2f -> nettoye=%.1f" % (tag, best[0], best[1]), flush=True)
    best3 = None
    for w1 in np.arange(0, 0.6, 0.05):
        for w2 in np.arange(0, 0.6, 0.05):
            if w1 + w2 > 0.7:
                continue
            p = g + gate * (w1 * (solo - g) + w2 * (mono - g))
            c = rmse(y_va[clean], p[clean])
            if best3 is None or c < best3[2]:
                best3 = (w1, w2, c)
    print("  arbres + les deux : w_mono=%.2f w_multi=%.2f -> nettoye=%.1f" % best3, flush=True)
    p = g + gate * (best3[0] * (solo - g) + best3[1] * (mono - g))
    rep = rmse_report(y_va, p, vp)
    log_experiment("M44 apprentissage multi-taches (arrivees)", model="LGBM + MLP multi-tache",
                   features="full + 2M arrivees etiquetees", val_rmse=rep["overall"],
                   val_rmse_jan=rep["jan"], val_rmse_jul=rep["jul"],
                   notes="nettoye %.1f" % best3[2], keep="")


if __name__ == "__main__":
    {"feats": stage_feats, "nn": stage_nn, "blend": stage_blend}[sys.argv[1]]()
