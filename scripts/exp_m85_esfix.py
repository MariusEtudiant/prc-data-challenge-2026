"""M85: make early stopping robust to the tail.

At LIRF ten flights out of 8,652 carry 62.5% of the early-stopping window's squared
error, so `best_iteration` is chosen largely at random. Fix: winsorise the ES
window's labels at 7200s - the training objective is untouched, only the stopping
criterion becomes robust. Three seeds per variant, because the claim is about
variance and a single seed cannot show it.
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

CAP, MIN_TRAIN, ES_CAP = 86400, 20000, 7200
SEEDS = [42, 1337, 2024]

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
apt_va = apt[va]
y_va = y[va].to_numpy()
clean = ~art[va]


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


t0 = time.time()
res = {}
for tag, es_cap in (("brut", CAP), ("winsorise", ES_CAP)):
    per, iters = {}, {}
    pred = np.full(va.sum(), np.nan)
    for a in np.unique(apt):
        f, s = ok & tr & (apt == a) & ~es, ok & tr & (apt == a) & es
        sv = apt_va == a
        if f.sum() < MIN_TRAIN or not sv.any():
            continue
        P, it = [], []
        for seed in SEEDS:
            m = lgb.train(tuned(a, seed),
                          lgb.Dataset(X[f], y[f].clip(upper=CAP), categorical_feature=cat),
                          num_boost_round=6000,
                          valid_sets=[lgb.Dataset(X[s], y[s].clip(upper=es_cap))],
                          callbacks=[lgb.early_stopping(150, verbose=False)])
            P.append(m.predict(X[va][sv], num_iteration=m.best_iteration))
            it.append(m.best_iteration)
        pred[sv] = np.mean(P, 0)
        per[a] = rmse(y_va[sv & clean], np.clip(pred[sv & clean], 60, CAP))
        iters[a] = it
    pred = np.clip(pred, 60, CAP)
    g = np.isfinite(pred)
    res[tag] = (rmse(y_va[clean & g], pred[clean & g]), per, iters)
    np.save(ROOT / "experiments" / ("m85_%s.npy" % tag), pred)
    print("%-12s nettoye=%7.2f   (%.0f min)" % (tag, res[tag][0], (time.time() - t0) / 60), flush=True)

b, w = res["brut"], res["winsorise"]
print("\n  %-8s %10s %10s %9s %26s %26s" % ("aero", "brut", "winsor.", "gain",
                                            "iterations brut", "iterations winsor."))
for a in sorted(b[1]):
    print("  %-8s %10.1f %10.1f %+9.1f %26s %26s" % (
        a, b[1][a], w[1][a], w[1][a] - b[1][a], b[2][a], w[2][a]))
print("\nGLOBAL nettoye %+.2f" % (w[0] - b[0]))
