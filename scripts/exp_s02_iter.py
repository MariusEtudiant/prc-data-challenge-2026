"""S02: substitute our own predictions for the neighbours we cannot observe.

The oracle (R05) showed that the true taxi of the five previous departures on the same
runway is worth -33.5 at LTFM, -12.4 at LEBL, -7.7 at LEMD. Those values are blanked
in the ranking file. Substituting the neighbours' d_aobt3 recovered almost nothing
(-1.4 at LTFM) because 14.3% of that timestamp's error is shared within an
airport-day, so averaging five of them does not clean it.

Our own predictions are the other candidate, and on paper a better one: at LTFM they
carry an error of 258s against d_aobt3's 432s, and they draw on geometry, turnaround
and queue rather than the Network Manager timestamp alone.

Leakage is the whole difficulty, so predictions feeding the feature are out-of-fold on
the training rows (five temporal folds) and come from a model that never saw the
holdout. Then the feature is rebuilt and the model retrained on top of it - one
iteration, measured honestly.
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
PROC, EXP = ROOT / "data" / "processed", ROOT / "experiments"
from dc26.validation import rmse

CAP, MIN_TRAIN, K = 86400, 20000, 5
SEEDS = [42, 1337]
FOLDS = [(2, 3), (4, 5), (6, 8), (9, 10), (11, 12)]
FILES = ["X_v2_mirror_t.parquet", "flightctx_train2025.parquet", "depregime_train2025.parquet",
         "aobtq_train2025.parquet", "anomreg_train2025.parquet"]
_EPOCH = pd.Timestamp(0, tz="UTC")

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

mv = pd.read_parquet(PROC / "movements_2025.parquet",
                     columns=["PHASE_mvt", "RUNWAY_mvt", "MVT_TIME_UTC_mvt"])
dep = mv[mv["PHASE_mvt"] == "DEP"].reset_index(drop=True)
t_all = ((dep["MVT_TIME_UTC_mvt"] - _EPOCH) / pd.Timedelta(seconds=1)).to_numpy(dtype=float)
mon_all = dep["MVT_TIME_UTC_mvt"].dt.month.to_numpy()
key = (X["AIRPORT"].astype(str) + "|" + dep["RUNWAY_mvt"].astype(str)).to_numpy()
mon = mon_all[va]


def params(a, seed):
    from dc26.mixture import REG_PARAMS
    p = EXP / ("optuna_" + a + ".json")
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


def fit_predict(XX, fit_m, stop_m, out_m, a):
    P = [lgb.train(params(a, sd),
                   lgb.Dataset(XX[fit_m], y[fit_m].clip(upper=CAP), categorical_feature=
                               [c for c in XX.columns if XX[c].dtype.name == "category"],
                               params={"feature_pre_filter": False, "max_bin": 255}),
                   num_boost_round=6000,
                   valid_sets=[lgb.Dataset(XX[stop_m], y[stop_m].clip(upper=CAP))],
                   callbacks=[lgb.early_stopping(150, verbose=False)]) for sd in SEEDS]
    return np.mean([m.predict(XX[out_m], num_iteration=m.best_iteration) for m in P], 0)


# ---- stage 1: out-of-fold predictions on training rows, full-model on the holdout
t0 = time.time()
src = np.full(len(y), np.nan)
base_va = np.full(va.sum(), np.nan)
for a in np.unique(apt):
    sel = ok & tr & (apt == a)
    sv = apt_va == a
    if sel.sum() < MIN_TRAIN or not sv.any():
        continue
    for months in FOLDS:
        hold = sel & np.isin(mon_all, months)
        fit = sel & ~np.isin(mon_all, months) & ~es
        stop = sel & ~np.isin(mon_all, months) & es
        if hold.sum() < 500 or fit.sum() < MIN_TRAIN // 2 or stop.sum() < 200:
            continue
        src[hold] = fit_predict(X, fit, stop, hold, a)
    out = np.zeros(len(y), bool)
    out[np.where(va)[0][sv]] = True          # boolean mask: pandas reads an integer array as columns
    base_va[sv] = fit_predict(X, sel & ~es, sel & es, out, a)
    print("  [%s] hors-pli %.0f%% (%.0f min)" % (
        a, 100 * np.isfinite(src[sel]).mean(), (time.time() - t0) / 60), flush=True)
src[va] = base_va
print("predictions sources pretes: %.1f%% couvertes" % (100 * np.isfinite(src).mean()), flush=True)

# ---- stage 2: neighbour feature built from those predictions
nbr = np.full(len(y), np.nan)
for k_, idx in pd.Series(np.arange(len(y))).groupby(key):
    i = idx.to_numpy()
    o = i[np.argsort(t_all[i])]
    v = src[o]
    good = np.isfinite(v)
    run, buf = np.full(len(o), np.nan), []
    for j in range(len(o)):
        run[j] = np.mean(buf) if len(buf) >= 3 else np.nan
        if good[j]:
            buf.append(v[j])
            if len(buf) > K:
                buf.pop(0)
    nbr[o] = run
np.save(EXP / "s02_nbr_iter1.npy", nbr)
E = pd.DataFrame({"nbr_pred_rwy": nbr})
print("variable voisins: %.1f%% | corr avec la cible %.3f" % (
    100 * np.isfinite(nbr).mean(),
    np.corrcoef(nbr[ok & np.isfinite(nbr) & (y.to_numpy() < 7200)],
                y.to_numpy()[ok & np.isfinite(nbr) & (y.to_numpy() < 7200)])[0, 1]), flush=True)

# ---- stage 3: retrain with it
res = {}
for tag, XX in (("sans", X), ("+ voisins predits", pd.concat([X, E], axis=1))):
    pred = np.full(va.sum(), np.nan)
    per = {}
    for a in np.unique(apt):
        sel = ok & tr & (apt == a)
        sv = apt_va == a
        if sel.sum() < MIN_TRAIN or not sv.any():
            continue
        out = np.zeros(len(y), bool)
        out[np.where(va)[0][sv]] = True
        pred[sv] = fit_predict(XX, sel & ~es, sel & es, out, a)
        c = sv & clean
        per[a] = rmse(y_va[c], np.clip(pred[c], 60, CAP))
    pred = np.clip(pred, 60, CAP)
    g = np.isfinite(pred) & clean
    res[tag] = (rmse(y_va[g], pred[g]), rmse(y_va[g & (mon == 7)], pred[g & (mon == 7)]), per)
    if tag != "sans":
        np.save(EXP / "s02_pred_iter1_va.npy", pred)
    print("%-20s nettoye=%7.2f | juillet=%7.2f (%.0f min)" % (
        tag, res[tag][0], res[tag][1], (time.time() - t0) / 60), flush=True)

b, w = res["sans"], res["+ voisins predits"]
print("\n  %-8s %10s %10s %10s" % ("aero", "sans", "avec", "gain"))
for a in sorted(b[2]):
    print("  %-8s %10.1f %10.1f %+10.1f" % (a, b[2][a], w[2][a], w[2][a] - b[2][a]))
print("\nGLOBAL %+.2f nettoye | %+.2f juillet" % (w[0] - b[0], w[1] - b[1]))
