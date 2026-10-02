"""M30: are the day-offset artifacts poisoning training?

The 2026 extract contains no ~24h artifacts in its labelled arrivals, while 2025
training does (15 departures). Those rows are data errors, not operations, yet RMSE
gives them enormous leverage on the trees.

  A  train as now (artifacts kept, capped)     evaluated on full and cleaned holdout
  B  train with near-24h rows removed          idem
Cleaned holdout = mirror holdout minus the same artifacts; it is the better proxy
for the 2026 scored set.
"""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd, lightgbm as lgb

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from dc26 import log_experiment
from dc26.validation import rmse, rmse_report
from dc26.mixture import REG_PARAMS

PROC = ROOT / "data" / "processed"
CAP, MIN_TRAIN = 86400, 20000
X = pd.concat([pd.read_parquet(PROC / "X_v2_mirror_t.parquet"),
               pd.read_parquet(PROC / "flightctx_train2025.parquet"),
               pd.read_parquet(PROC / "depregime_train2025.parquet")], axis=1)
meta = pd.read_parquet(PROC / "meta_train2025.parquet")
y = meta["y"].astype("float64")
tr, va, es = meta["tr"].to_numpy(), meta["va"].to_numpy(), meta["es"].to_numpy()
vp = meta.loc[va, "val_part"].to_numpy()
keep = tr & (y > 0).to_numpy() & (y < 200000).to_numpy()
apt = X["AIRPORT"].astype(str).to_numpy(); apt_va = apt[va]

# day-offset artifacts: within an hour of a whole number of days
artifact = np.zeros(len(y), bool)
for k in (1, 2, 3):
    artifact |= (y - k * 86400).abs().to_numpy() < 3600
print("artefacts detectes: %d au total, %d dans le train, %d dans le holdout"
      % (artifact.sum(), (artifact & tr).sum(), (artifact & va).sum()), flush=True)

y_va = y[va].to_numpy()
clean_va = ~artifact[va]
print("holdout: %d lignes, dont %d artefacts" % (len(y_va), (~clean_va).sum()), flush=True)

def tuned(a):
    p = ROOT / "experiments" / ("optuna_" + a + ".json")
    base = dict(objective="regression", metric="rmse", verbosity=-1, num_threads=0,
                bagging_freq=1, seed=42)
    if not p.exists():
        return dict(REG_PARAMS)
    d = json.loads(p.read_text())
    base.update({k: v for k, v in d.items() if not k.startswith("_")})
    for k in ("num_leaves", "min_data_in_leaf", "max_bin"):
        base[k] = int(base[k])
    return base

cat = [c for c in X.columns if X[c].dtype.name == "category"]

def run(mask_extra, tag):
    pred = np.full(va.sum(), np.nan)
    for a in np.unique(apt):
        fit = keep & (apt == a) & ~es & mask_extra
        ev = keep & (apt == a) & es & mask_extra
        sel = apt_va == a
        if fit.sum() < MIN_TRAIN or not sel.any():
            continue
        m = lgb.train(tuned(a), lgb.Dataset(X[fit], y[fit].clip(upper=CAP), categorical_feature=cat),
                      num_boost_round=6000, valid_sets=[lgb.Dataset(X[ev], y[ev].clip(upper=CAP))],
                      callbacks=[lgb.early_stopping(150, verbose=False)])
        pred[sel] = m.predict(X[va][sel], num_iteration=m.best_iteration)
    pred = np.clip(pred, 60, CAP)
    full = rmse_report(y_va, pred, vp)
    cl = rmse(y_va[clean_va], pred[clean_va])
    print("%s: holdout complet=%.1f | holdout NETTOYE=%.1f  (jan=%.1f jul=%.1f)"
          % (tag, full["overall"], cl, full["jan"], full["jul"]), flush=True)
    return full, cl, pred

fullA, clA, pA = run(np.ones(len(y), bool), "A artefacts gardes")
fullB, clB, pB = run(~artifact, "B artefacts retires du train")
print("\ndelta sur holdout nettoye: %+.2f s" % (clB - clA), flush=True)
for a in ["LIRF", "LFPG", "EHAM", "LTFM"]:
    s = apt_va == a
    sc = s & clean_va
    print("   [%s] nettoye: %.1f -> %.1f" % (a, rmse(y_va[sc], pA[sc]), rmse(y_va[sc], pB[sc])), flush=True)
log_experiment("M30 artefacts 24h retires du train", model="per-airport tuned LGBM",
               features="full", val_rmse=fullB["overall"], val_rmse_jan=fullB["jan"],
               val_rmse_jul=fullB["jul"],
               notes="holdout nettoye A=%.1f B=%.1f" % (clA, clB),
               keep="KEEP" if clB < clA else "DROP")
print("done", flush=True)
