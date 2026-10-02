"""M95: K-fold bagging against our single-split production design, like for like.

Production today fits each per-airport model on tr&~es with NO early stopping - it
reuses the mirror's best_iteration multiplied by a hand-chosen BOOST factor to
compensate for the extra data. That factor is a guess, and the 21-day ES window is
never used for fitting.

The architecture every ensembling competition converges on is K-fold bagging: split
the training period into K temporal blocks, fit K models each early-stopping on its
own held-out block, average the K predictions. No heuristic, K-model variance
reduction, and honest out-of-fold predictions for free.

M27 tried bagging once but was compared against a full ensemble (319.09), never
against the same model family under the same protocol. This does that.

  A  3 seeds, single split, early stop on the fixed ES window   (today)
  B  5 temporal folds, 1 seed each, early stop on own fold
  C  5 folds x 2 seeds = 10 models
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
PROC = ROOT / "data" / "processed"
from dc26.validation import rmse
from dc26 import log_experiment

CAP, MIN_TRAIN = 86400, 20000
FOLDS = [(1, 2, 3), (4, 5), (6, 7), (8, 9), (10, 11, 12)]

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
cat = [c for c in X.columns if X[c].dtype.name == "category"]
apt = X["AIRPORT"].astype(str).to_numpy()
month = X["month"].to_numpy()
apt_va, y_va, clean = apt[va], y[va].to_numpy(), ~art[va]
vp = meta.loc[va, "val_part"].to_numpy()


def tuned(a, seed):
    from dc26.mixture import REG_PARAMS
    p = ROOT / "experiments" / ("optuna_" + a + ".json")
    b = dict(objective="regression", metric="rmse", verbosity=-1, num_threads=0,
             bagging_freq=1, seed=seed, bagging_seed=seed + 1, feature_fraction_seed=seed + 2)
    if not p.exists():
        return dict(REG_PARAMS, seed=seed)
    d = json.loads(p.read_text())
    b.update({k: v for k, v in d.items() if not k.startswith("_")})
    for k in ("num_leaves", "min_data_in_leaf", "max_bin"):
        b[k] = int(b[k])
    b["seed"] = seed
    return b


def train_one(a, fit_m, stop_m, seed, sv):
    m = lgb.train(tuned(a, seed),
                  lgb.Dataset(X[fit_m], y[fit_m].clip(upper=CAP), categorical_feature=cat),
                  num_boost_round=6000,
                  valid_sets=[lgb.Dataset(X[stop_m], y[stop_m].clip(upper=CAP))],
                  callbacks=[lgb.early_stopping(150, verbose=False)])
    return m.predict(X[va][sv], num_iteration=m.best_iteration), m.best_iteration


VARIANTS = {
    "A  3 graines, coupe fixe": ("split", [42, 1337, 2024]),
    "B  5 plis, 1 graine": ("kfold", [42]),
    "C  5 plis, 2 graines": ("kfold", [42, 1337]),
}

t0 = time.time()
res = {}
for tag, (mode, seeds) in VARIANTS.items():
    pred = np.full(va.sum(), np.nan)
    per, nmod = {}, {}
    for a in np.unique(apt):
        base = ok & tr & (apt == a)
        sv = apt_va == a
        if base.sum() < MIN_TRAIN or not sv.any():
            continue
        P, iters = [], []
        if mode == "split":
            for s in seeds:
                p, it = train_one(a, base & ~es, base & es, s, sv)
                P.append(p); iters.append(it)
        else:
            for months in FOLDS:
                hold = base & np.isin(month, months)
                fit = base & ~np.isin(month, months)
                if hold.sum() < 500 or fit.sum() < MIN_TRAIN // 2:
                    continue
                for s in seeds:
                    p, it = train_one(a, fit, hold, s, sv)
                    P.append(p); iters.append(it)
        pred[sv] = np.mean(P, 0)
        per[a] = rmse(y_va[sv & clean], np.clip(pred[sv & clean], 60, CAP))
        nmod[a] = len(P)
    pred = np.clip(pred, 60, CAP)
    g = np.isfinite(pred)
    res[tag] = (rmse(y_va[clean & g], pred[clean & g]),
                rmse(y_va[clean & g & (vp == "jul")], pred[clean & g & (vp == "jul")]), per, nmod)
    np.save(ROOT / "experiments" / ("m95_%s.npy" % tag.split()[0]), pred)
    print("%-26s nettoye=%7.2f  juillet=%7.2f  (%d modeles/aeroport, %.0f min)" % (
        tag, res[tag][0], res[tag][1], list(nmod.values())[0], (time.time() - t0) / 60), flush=True)

A = res["A  3 graines, coupe fixe"]
print("\n  %-8s %10s %10s %10s" % ("aero", "A", "B", "C"))
for a in sorted(A[2]):
    print("  %-8s %10.1f %+9.1f %+9.1f" % (
        a, A[2][a], res["B  5 plis, 1 graine"][2][a] - A[2][a],
        res["C  5 plis, 2 graines"][2][a] - A[2][a]))
for t in ("B  5 plis, 1 graine", "C  5 plis, 2 graines"):
    print("\n%s : %+.2f nettoye | %+.2f juillet" % (t, res[t][0] - A[0], res[t][1] - A[1]))
    log_experiment("M95 %s" % t, model="LGBM par aeroport", features="full",
                   val_rmse=res[t][0], notes="A=%.2f juillet %.2f vs %.2f" % (A[0], res[t][1], A[1]), keep="")
