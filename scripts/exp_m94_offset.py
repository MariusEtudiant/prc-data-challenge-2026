"""M94: revive M16 (init_score residual modelling) in the per-airport architecture.

A gradient-boosted tree is piecewise constant, so making it rebuild y ~ d_aobt3 spends
splits approximating the identity function on a quantity we already hold. Handing
d_aobt3 as the init_score leaves the trees only the correction to learn.

M16 tested this on 2026-09-01 with a single global model (343.09 vs 344.8, marked
KEEP) and it was never carried over when the pipeline moved to per-airport tuned
models. Three seeds per variant, cleaned mirror plus July.
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
apt_va, y_va, clean = apt[va], y[va].to_numpy(), ~art[va]
vp = meta.loc[va, "val_part"].to_numpy()

d = X["d_aobt3"].to_numpy(dtype="float64")
ref = X["ref_txot_p10"].to_numpy(dtype="float64")
o = np.where(np.isfinite(d) & (d > 60) & (d < 7200), d, np.nan)
o = np.where(np.isnan(o), ref, o)
OFF = np.nan_to_num(o, nan=900.0)
print("offset: mediane=%.0fs | issu d AOBT_3 pour %.1f%% des lignes" % (
    np.median(OFF), 100 * np.mean(np.isfinite(d) & (d > 60) & (d < 7200))), flush=True)
print("RMSE de l offset seul sur le holdout nettoye: %.1f" % rmse(
    y_va[clean], np.clip(OFF[va][clean], 60, CAP)), flush=True)


def tuned(a, seed):
    from dc26.mixture import REG_PARAMS
    p = ROOT / "experiments" / ("optuna_" + a + ".json")
    b = dict(objective="regression", metric="rmse", verbosity=-1, num_threads=0,
             bagging_freq=1, seed=seed, bagging_seed=seed + 1, feature_fraction_seed=seed + 2)
    if not p.exists():
        return dict(REG_PARAMS, seed=seed)
    dd = json.loads(p.read_text())
    b.update({k: v for k, v in dd.items() if not k.startswith("_")})
    for k in ("num_leaves", "min_data_in_leaf", "max_bin"):
        b[k] = int(b[k])
    b["seed"] = seed
    return b


t0 = time.time()
res = {}
for tag, use in (("sans offset", False), ("avec offset", True)):
    pred = np.full(va.sum(), np.nan)
    per = {}
    for a in np.unique(apt):
        f, s = ok & tr & (apt == a) & ~es, ok & tr & (apt == a) & es
        sv = apt_va == a
        if f.sum() < MIN_TRAIN or not sv.any():
            continue
        P = []
        for seed in SEEDS:
            dtr = lgb.Dataset(X[f], y[f].clip(upper=CAP), categorical_feature=cat)
            des = lgb.Dataset(X[s], y[s].clip(upper=CAP), reference=dtr)
            if use:
                dtr.set_init_score(OFF[f])
                des.set_init_score(OFF[s])
            m = lgb.train(tuned(a, seed), dtr, num_boost_round=6000, valid_sets=[des],
                          callbacks=[lgb.early_stopping(150, verbose=False)])
            p = m.predict(X[va][sv], num_iteration=m.best_iteration)
            P.append(p + OFF[va][sv] if use else p)
        pred[sv] = np.mean(P, 0)
        per[a] = rmse(y_va[sv & clean], np.clip(pred[sv & clean], 60, CAP))
    pred = np.clip(pred, 60, CAP)
    g = np.isfinite(pred)
    res[tag] = (rmse(y_va[clean & g], pred[clean & g]),
                rmse(y_va[clean & g & (vp == "jul")], pred[clean & g & (vp == "jul")]), per)
    np.save(ROOT / "experiments" / ("m94_%s.npy" % tag.split()[1]), pred)
    print("%-14s nettoye=%7.2f  juillet=%7.2f   (%.0f min)" % (
        tag, res[tag][0], res[tag][1], (time.time() - t0) / 60), flush=True)

b, w = res["sans offset"], res["avec offset"]
print("\n  %-8s %10s %12s %9s" % ("aero", "sans", "avec", "gain"))
for a in sorted(b[2]):
    print("  %-8s %10.1f %12.1f %+9.1f" % (a, b[2][a], w[2][a], w[2][a] - b[2][a]))
print("\nGLOBAL nettoye %+.2f | juillet %+.2f" % (w[0] - b[0], w[1] - b[1]))
log_experiment("M94 offset init_score par aeroport", model="LGBM par aeroport + offset",
               features="full", val_rmse=w[0],
               notes="sans %.2f | juillet %.2f vs %.2f" % (b[0], w[1], b[1]), keep="")
