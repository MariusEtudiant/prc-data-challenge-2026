"""R05: is there anything in the departure QUEUE we are not already capturing?

Rome sends 90% of its departures down a single runway, so its taxi-out variance is
not geometry (a perfect distance feature explains 10% there against 42% at Madrid) -
it is queueing. And the day-level signal is already ours: three of the four worst July
days are the only ones in the month carrying a weather ATFM delay, and
atfm_wx_per_arr is in the matrix with no missing values. Knowing the day is bad does
not say which flight suffers.

Before building an iterative scheme that estimates each aircraft's place in the
departure sequence, measure the CEILING of the idea with an oracle: give the model
the TRUE taxi time of the neighbouring departures on the same runway. Those values are
blanked in the ranking file, so this is not a usable feature - it is an upper bound.
If even the oracle buys little, the whole sequential direction is dead and we save
days; if it buys a lot, the iterative version (substituting our own predictions for
the unknown neighbours) is worth building.
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
SEEDS = [42, 1337]
FILES = ["X_v2_mirror_t.parquet", "flightctx_train2025.parquet", "depregime_train2025.parquet",
         "aobtq_train2025.parquet", "anomreg_train2025.parquet"]
_EPOCH = pd.Timestamp(0, tz="UTC")

mv = pd.read_parquet(PROC / "movements_2025.parquet",
                     columns=["AIRPORT", "PHASE_mvt", "RUNWAY_mvt", "MVT_TIME_UTC_mvt",
                              "TAXITIME_SEC_mvt", "AOBT_3_flt"])
dep = mv[mv["PHASE_mvt"] == "DEP"].reset_index(drop=True)
t_all = ((dep["MVT_TIME_UTC_mvt"] - _EPOCH) / pd.Timedelta(seconds=1)).to_numpy(dtype=float)
y_all = dep["TAXITIME_SEC_mvt"].astype(float).to_numpy()
# REALISABLE substitute: d_aobt3 of the neighbours is observable in the ranking file,
# where their taxi time is not. Same sequence aggregation, noisier input.
d_all = ((dep["MVT_TIME_UTC_mvt"] - dep["AOBT_3_flt"]).dt.total_seconds()).to_numpy(dtype=float)
key = (dep["AIRPORT"].astype(str) + "|" + dep["RUNWAY_mvt"].astype(str)).to_numpy()

# ORACLE: mean true taxi of the k previous departures on the same runway
ORACLE = False   # False = realisable version built on the neighbours' d_aobt3
K = 5
nbr = np.full(len(dep), np.nan)
nbr_gap = np.full(len(dep), np.nan)
for k, idx in pd.Series(np.arange(len(dep))).groupby(key):
    i = idx.to_numpy()
    o = i[np.argsort(t_all[i])]
    yv = y_all[o] if ORACLE else d_all[o]
    ok = np.isfinite(yv) & (yv > 0) & (yv < 20000)
    run = np.full(len(o), np.nan)
    buf = []
    for j in range(len(o)):
        run[j] = np.mean(buf) if len(buf) >= 3 else np.nan
        if ok[j]:
            buf.append(yv[j])
            if len(buf) > K:
                buf.pop(0)
    nbr[o] = run
    g = np.diff(t_all[o], prepend=np.nan)
    nbr_gap[o] = pd.Series(g).rolling(K, min_periods=3).mean().to_numpy()
print("oracle voisins: couverture %.1f%% | corr avec la cible %.3f" % (
    100 * np.isfinite(nbr).mean(),
    np.corrcoef(nbr[np.isfinite(nbr) & np.isfinite(y_all) & (y_all < 7200)],
                y_all[np.isfinite(nbr) & np.isfinite(y_all) & (y_all < 7200)])[0, 1]), flush=True)

X = pd.concat([pd.read_parquet(PROC / f) for f in FILES], axis=1)
meta = pd.read_parquet(PROC / "meta_train2025.parquet")
y = meta["y"].astype("float64")
tr, va, es = meta["tr"].to_numpy(), meta["va"].to_numpy(), meta["es"].to_numpy()
ok = (y > 0).to_numpy() & (y < 200000).to_numpy()
art = np.zeros(len(y), bool)
for k in (1, 2, 3):
    art |= (y - k * 86400).abs().to_numpy() < 3600
ok &= ~art
E = pd.DataFrame({"nbr_dprox_rwy": nbr, "nbr_gap_rwy": nbr_gap})
cat = [c for c in X.columns if X[c].dtype.name == "category"]
apt = X["AIRPORT"].astype(str).to_numpy()
apt_va, y_va, clean = apt[va], y[va].to_numpy(), ~art[va]
mon = dep["MVT_TIME_UTC_mvt"].dt.month.to_numpy()[va]


def tuned(a, seed):
    from dc26.mixture import REG_PARAMS
    p = ROOT / "experiments" / ("optuna_" + a + ".json")
    b = dict(objective="regression", metric="rmse", verbosity=-1, num_threads=0,
             bagging_freq=1, seed=seed, bagging_seed=seed + 1, feature_fraction_seed=seed + 2,
             feature_pre_filter=False, max_bin=255)
    if p.exists():
        d = json.loads(p.read_text())
        b.update({k: v for k, v in d.items() if not k.startswith("_")})
        for k in ("num_leaves", "min_data_in_leaf"):
            b[k] = int(b[k])
        b["max_bin"] = 255
    else:
        b.update(REG_PARAMS)
    b["seed"] = seed
    return b


t0 = time.time()
res = {}
for tag, XX in (("sans", X), ("+ file realisable", pd.concat([X, E], axis=1))):
    pred = np.full(va.sum(), np.nan)
    per = {}
    for a in np.unique(apt):
        f, s = ok & tr & (apt == a) & ~es, ok & tr & (apt == a) & es
        sv = apt_va == a
        if f.sum() < MIN_TRAIN or not sv.any():
            continue
        P = [lgb.train(tuned(a, sd),
                       lgb.Dataset(XX[f], y[f].clip(upper=CAP), categorical_feature=cat,
                                   params={"feature_pre_filter": False, "max_bin": 255}),
                       num_boost_round=6000,
                       valid_sets=[lgb.Dataset(XX[s], y[s].clip(upper=CAP))],
                       callbacks=[lgb.early_stopping(150, verbose=False)]) for sd in SEEDS]
        pred[sv] = np.mean([m.predict(XX[va][sv], num_iteration=m.best_iteration) for m in P], 0)
        c = sv & clean
        per[a] = rmse(y_va[c], np.clip(pred[c], 60, CAP))
    pred = np.clip(pred, 60, CAP)
    g = np.isfinite(pred) & clean
    res[tag] = (rmse(y_va[g], pred[g]), rmse(y_va[g & (mon == 7)], pred[g & (mon == 7)]), per)
    print("%-16s nettoye=%7.2f | juillet=%7.2f  (%.0f min)" % (
        tag, res[tag][0], res[tag][1], (time.time() - t0) / 60), flush=True)

b, w = res["sans"], res["+ file realisable"]
print("\n  %-8s %10s %10s %10s" % ("aero", "sans", "realisable","gain"))
for a in sorted(b[2]):
    print("  %-8s %10.1f %10.1f %+10.1f" % (a, b[2][a], w[2][a], w[2][a] - b[2][a]))
print("\nVERSION REALISABLE (d_aobt3 des voisins) : %+.2f nettoye | %+.2f juillet" % (
    w[0] - b[0], w[1] - b[1]))
