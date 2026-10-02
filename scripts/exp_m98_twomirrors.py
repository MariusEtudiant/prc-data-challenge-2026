"""M98: bracket the truth between a pessimistic and an optimistic mirror.

With a single year of training data no mirror can reproduce the real situation -
predicting July 2026 while holding July 2025 in training. Two bounds instead:

  A  conservative  hold out all of January and all of July      (current)
     the model gets no same-month exposure at all, so it cannot learn that Rome's
     July carries six times January's rate of >2h flights -> overstates difficulty

  B  exposed       hold out only the 16th-31st of each month
     the model trains on the 1st-15th of the same month, so it has same-month
     exposure as production does -> adjacent-period leakage understates difficulty

A variant that improves on BOTH is safe; one that improves on only one is suspect.

Note: the reference times and the AOBT_3 spike model were fitted on mirror A's train
mask, which for mirror B is conservative (they saw less data than B's model does).
That biases B slightly pessimistic, i.e. against the point being made, so it does not
weaken the conclusion.
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

CAP, MIN_TRAIN = 86400, 20000
SEEDS = [42, 1337, 2024]

X = pd.concat([pd.read_parquet(PROC / f) for f in
               ["X_v2_mirror_t.parquet", "flightctx_train2025.parquet",
                "depregime_train2025.parquet", "aobtq_train2025.parquet",
                "anomreg_train2025.parquet"]], axis=1)
meta = pd.read_parquet(PROC / "meta_train2025.parquet")
y = meta["y"].astype("float64")
es = meta["es"].to_numpy()
ok = (y > 0).to_numpy() & (y < 200000).to_numpy()
art = np.zeros(len(y), bool)
for k in (1, 2, 3):
    art |= (y - k * 86400).abs().to_numpy() < 3600
ok &= ~art
cat = [c for c in X.columns if X[c].dtype.name == "category"]
apt = X["AIRPORT"].astype(str).to_numpy()
mvt = pd.read_parquet(PROC / "movements_2025.parquet", columns=["PHASE_mvt", "MVT_TIME_UTC_mvt"])
dep_t = mvt[mvt["PHASE_mvt"] == "DEP"]["MVT_TIME_UTC_mvt"].reset_index(drop=True)
month, day = dep_t.dt.month.to_numpy(), dep_t.dt.day.to_numpy()

MIRRORS = {
    "A conservateur": np.isin(month, (1, 7)),
    "B expose": np.isin(month, (1, 7)) & (day >= 16),
}


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
for tag, vmask in MIRRORS.items():
    tmask = ~vmask
    yv = y[vmask].to_numpy()
    av, cl = apt[vmask], ~art[vmask]
    jv = month[vmask] == 1
    pred = np.full(vmask.sum(), np.nan)
    per = {}
    for a in np.unique(apt):
        f = ok & tmask & (apt == a) & ~es
        s = ok & tmask & (apt == a) & es
        sv = av == a
        if f.sum() < MIN_TRAIN or s.sum() < 300 or not sv.any():
            continue
        P = [lgb.train(tuned(a, sd), lgb.Dataset(X[f], y[f].clip(upper=CAP), categorical_feature=cat),
                       num_boost_round=6000,
                       valid_sets=[lgb.Dataset(X[s], y[s].clip(upper=CAP))],
                       callbacks=[lgb.early_stopping(150, verbose=False)]) for sd in SEEDS]
        pred[sv] = np.mean([m.predict(X[vmask][sv], num_iteration=m.best_iteration) for m in P], 0)
        c = sv & cl
        per[a] = (rmse(yv[c], np.clip(pred[c], 60, CAP)),
                  rmse(yv[c & jv], np.clip(pred[c & jv], 60, CAP)) if (c & jv).sum() > 100 else np.nan,
                  rmse(yv[c & ~jv], np.clip(pred[c & ~jv], 60, CAP)) if (c & ~jv).sum() > 100 else np.nan)
    pred = np.clip(pred, 60, CAP)
    g = np.isfinite(pred) & cl
    print("\n=== miroir %s : n_val=%d, entrainement=%d ===" % (tag, vmask.sum(), (ok & tmask).sum()), flush=True)
    print("  nettoye=%7.2f | janvier=%7.2f | juillet=%7.2f   (%.0f min)" % (
        rmse(yv[g], pred[g]), rmse(yv[g & jv], pred[g & jv]), rmse(yv[g & ~jv], pred[g & ~jv]),
        (time.time() - t0) / 60), flush=True)
    print("  %-8s %10s %10s %10s" % ("aero", "global", "janvier", "juillet"))
    for a in sorted(per):
        print("  %-8s %10.1f %10.1f %10.1f" % (a, *per[a]))
    np.save(ROOT / "experiments" / ("m98_%s.npy" % tag.split()[0]), pred)
