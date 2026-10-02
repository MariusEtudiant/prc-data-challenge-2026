"""N01: are we underfitting, or are the features saturated?

We are 13th on the reset leaderboard, 66 points behind, and yesterday every
refinement was worth under a point. That pattern says the gap is structural, not a
matter of tuning. The first structural question is the cheapest to answer: our
per-airport models early-stop between 300 and 600 rounds with 31-138 leaves on
150,000 rows. If ten times the capacity buys a large improvement, our ceiling is the
model. If it buys nothing, the ceiling is the feature set and we should be building
new features rather than polishing these.

Four configurations per airport, same data, same early stopping, on the mirror
rebuilt for the regenerated July:

  tuned    what we ship today (Optuna, fitted on the OLD composition)
  big      512 leaves, lr 0.02, min_data 40
  huge     2048 leaves, lr 0.01, min_data 20, up to 20000 rounds
  slow     current leaves, lr 0.005, up to 30000 rounds - depth of boosting rather
           than width of trees

Run on three airports chosen to span the difficulty range, not on all ten, so the
answer arrives in minutes rather than hours.
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

CAP = 86400
AIRPORTS = ["LIRF", "LTFM", "EDDF"]
FILES = ["X_v2_mirror_t.parquet", "flightctx_train2025.parquet", "depregime_train2025.parquet",
         "aobtq_train2025.parquet", "anomreg_train2025.parquet"]


def tuned(a):
    from dc26.mixture import REG_PARAMS
    p = ROOT / "experiments" / ("optuna_" + a + ".json")
    b = dict(objective="regression", metric="rmse", verbosity=-1, num_threads=0,
             bagging_freq=1, seed=42)
    if not p.exists():
        return dict(REG_PARAMS)
    d = json.loads(p.read_text())
    b.update({k: v for k, v in d.items() if not k.startswith("_")})
    for k in ("num_leaves", "min_data_in_leaf", "max_bin"):
        b[k] = int(b[k])
    return b


def variants(a):
    t = tuned(a)
    base = dict(objective="regression", metric="rmse", verbosity=-1, num_threads=0,
                bagging_freq=1, seed=42, feature_fraction=0.8, bagging_fraction=0.8)
    return {
        "tuned": (t, 6000),
        "big": (dict(base, num_leaves=512, learning_rate=0.02, min_data_in_leaf=40,
                     lambda_l2=1.0, max_bin=255), 10000),
        "huge": (dict(base, num_leaves=2048, learning_rate=0.01, min_data_in_leaf=20,
                      lambda_l2=1.0, max_bin=255), 20000),
        "slow": (dict(t, learning_rate=0.005), 30000),
    }


X = pd.concat([pd.read_parquet(PROC / f) for f in FILES], axis=1)
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
mvt = pd.read_parquet(PROC / "movements_2025.parquet", columns=["PHASE_mvt", "MVT_TIME_UTC_mvt"])
mon = mvt[mvt["PHASE_mvt"] == "DEP"]["MVT_TIME_UTC_mvt"].reset_index(drop=True).dt.month.to_numpy()[va]

print("%-7s %-7s %9s %9s %9s %9s %8s" % (
    "aero", "config", "nettoye", "janvier", "juillet", "iters", "minutes"))
for a in AIRPORTS:
    f, s = ok & tr & (apt == a) & ~es, ok & tr & (apt == a) & es
    sv = apt_va == a
    ref = None
    for tag, (params, rounds) in variants(a).items():
        t0 = time.time()
        m = lgb.train(params, lgb.Dataset(X[f], y[f].clip(upper=CAP), categorical_feature=cat),
                      num_boost_round=rounds,
                      valid_sets=[lgb.Dataset(X[s], y[s].clip(upper=CAP))],
                      callbacks=[lgb.early_stopping(200, verbose=False)])
        p = np.clip(m.predict(X[va][sv], num_iteration=m.best_iteration), 60, CAP)
        c = clean[sv]
        r = rmse(y_va[sv][c], p[c])
        rj = rmse(y_va[sv][c & (mon[sv] == 1)], p[c & (mon[sv] == 1)])
        ru = rmse(y_va[sv][c & (mon[sv] == 7)], p[c & (mon[sv] == 7)])
        if ref is None:
            ref = r
        print("%-7s %-7s %9.2f %9.2f %9.2f %9d %8.1f  %s" % (
            a, tag, r, rj, ru, m.best_iteration, (time.time() - t0) / 60,
            "" if tag == "tuned" else "(%+.2f)" % (r - ref)), flush=True)
    print(flush=True)
