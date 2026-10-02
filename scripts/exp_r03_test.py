"""R03: do the forward-looking arrival features earn their place?

Same protocol as every other feature test we have run: per-airport LightGBM, three
seeds, tuned parameters, mirror rebuilt for the regenerated July, cleaned metric,
reported separately for January and July and broken down per airport.

The bar is the one that caught the runway-configuration features out: a feature that
helps one airport and hurts elsewhere is rejected. LIRF July is where the deficit is,
but the test is both months at ten airports.
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
FILES = ["X_v2_mirror_t.parquet", "flightctx_train2025.parquet", "depregime_train2025.parquet",
         "aobtq_train2025.parquet", "anomreg_train2025.parquet"]

X = pd.concat([pd.read_parquet(PROC / f) for f in FILES], axis=1)
E = pd.read_parquet(PROC / "arrfwd_train2025.parquet")
meta = pd.read_parquet(PROC / "meta_train2025.parquet")
y = meta["y"].astype("float64")
tr, va, es = meta["tr"].to_numpy(), meta["va"].to_numpy(), meta["es"].to_numpy()
ok = (y > 0).to_numpy() & (y < 200000).to_numpy()
art = np.zeros(len(y), bool)
for k in (1, 2, 3):
    art |= (y - k * 86400).abs().to_numpy() < 3600
ok &= ~art
apt = X["AIRPORT"].astype(str).to_numpy()
apt_va, y_va, clean = apt[va], y[va].to_numpy(), ~art[va]
mvt = pd.read_parquet(PROC / "movements_2025.parquet", columns=["PHASE_mvt", "MVT_TIME_UTC_mvt"])
mon = mvt[mvt["PHASE_mvt"] == "DEP"]["MVT_TIME_UTC_mvt"].reset_index(drop=True).dt.month.to_numpy()[va]


def tuned(a, seed):
    from dc26.mixture import REG_PARAMS
    p = ROOT / "experiments" / ("optuna_" + a + ".json")
    b = dict(objective="regression", metric="rmse", verbosity=-1, num_threads=0,
             bagging_freq=1, seed=seed, bagging_seed=seed + 1, feature_fraction_seed=seed + 2,
             feature_pre_filter=False)
    if not p.exists():
        return dict(REG_PARAMS, seed=seed, feature_pre_filter=False)
    d = json.loads(p.read_text())
    b.update({k: v for k, v in d.items() if not k.startswith("_")})
    for k in ("num_leaves", "min_data_in_leaf", "max_bin"):
        b[k] = int(b[k])
    b["seed"] = seed
    return b


t0 = time.time()
res = {}
for tag, XX in (("sans", X), ("avec arr. avant", pd.concat([X, E], axis=1))):
    cat = [c for c in XX.columns if XX[c].dtype.name == "category"]
    pred = np.full(va.sum(), np.nan)
    per = {}
    for a in np.unique(apt):
        f, s = ok & tr & (apt == a) & ~es, ok & tr & (apt == a) & es
        sv = apt_va == a
        if f.sum() < MIN_TRAIN or not sv.any():
            continue
        P = []
        for seed in SEEDS:
            m = lgb.train(tuned(a, seed),
                          lgb.Dataset(XX[f], y[f].clip(upper=CAP), categorical_feature=cat,
                                      params={"feature_pre_filter": False}),
                          num_boost_round=6000,
                          valid_sets=[lgb.Dataset(XX[s], y[s].clip(upper=CAP))],
                          callbacks=[lgb.early_stopping(150, verbose=False)])
            P.append(m.predict(XX[va][sv], num_iteration=m.best_iteration))
        pred[sv] = np.mean(P, 0)
        c = sv & clean
        per[a] = (rmse(y_va[c], np.clip(pred[c], 60, CAP)),
                  rmse(y_va[c & (mon == 1)], np.clip(pred[c & (mon == 1)], 60, CAP)),
                  rmse(y_va[c & (mon == 7)], np.clip(pred[c & (mon == 7)], 60, CAP)))
    pred = np.clip(pred, 60, CAP)
    g = np.isfinite(pred) & clean
    res[tag] = (rmse(y_va[g], pred[g]), rmse(y_va[g & (mon == 1)], pred[g & (mon == 1)]),
                rmse(y_va[g & (mon == 7)], pred[g & (mon == 7)]), per)
    np.save(ROOT / "experiments" / ("r03_%s.npy" % tag.split()[0]), pred)
    print("%-16s nettoye=%7.2f | janvier=%7.2f | juillet=%7.2f  (%.0f min)" % (
        tag, res[tag][0], res[tag][1], res[tag][2], (time.time() - t0) / 60), flush=True)

b, w = res["sans"], res["avec arr. avant"]
print("\n  %-8s %10s %10s %10s %10s %10s %10s" % (
    "aero", "sans", "avec", "gain", "jul sans", "jul avec", "gain jul"))
for a in sorted(b[3]):
    print("  %-8s %10.1f %10.1f %+10.1f %10.1f %10.1f %+10.1f" % (
        a, b[3][a][0], w[3][a][0], w[3][a][0] - b[3][a][0],
        b[3][a][2], w[3][a][2], w[3][a][2] - b[3][a][2]))
print("\nGLOBAL %+.2f nettoye | %+.2f janvier | %+.2f juillet" % (
    w[0] - b[0], w[1] - b[1], w[2] - b[2]))
log_experiment("R03 arrivees prospectives (transductif)", model="LGBM par aeroport x3",
               features="full + 7 arr. avant/centrees", val_rmse=w[0],
               val_rmse_jan=w[1], val_rmse_jul=w[2],
               notes="sans: %.2f / %.2f / %.2f" % (b[0], b[1], b[2]), keep="")
