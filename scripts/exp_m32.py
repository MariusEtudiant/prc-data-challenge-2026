"""M32: re-run the M21 extreme-segment decision on the CLEANED metric.

M21 concluded "do not shrink the extreme segment" from the raw mirror, which is
dominated by 2 day-offset artifacts - one of them inside the very segment tested.
The 2026 scored set appears to contain no such artifacts, so the decision must be
re-taken on a holdout with them removed.
"""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd, lightgbm as lgb

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from dc26 import log_experiment
from dc26.validation import rmse
from dc26.mixture import REG_PARAMS

PROC = ROOT / "data" / "processed"
CAP, MIN_TRAIN, THR = 86400, 20000, 7200
X = pd.concat([pd.read_parquet(PROC / "X_v2_mirror_t.parquet"),
               pd.read_parquet(PROC / "flightctx_train2025.parquet"),
               pd.read_parquet(PROC / "depregime_train2025.parquet")], axis=1)
meta = pd.read_parquet(PROC / "meta_train2025.parquet")
y = meta["y"].astype("float64")
tr, va, es = meta["tr"].to_numpy(), meta["va"].to_numpy(), meta["es"].to_numpy()
keep = tr & (y > 0).to_numpy() & (y < 200000).to_numpy()
apt = X["AIRPORT"].astype(str).to_numpy(); apt_va = apt[va]
unm = (X["has_fpl"].to_numpy() == 0)

artifact = np.zeros(len(y), bool)
for k in (1, 2, 3):
    artifact |= (y - k * 86400).abs().to_numpy() < 3600
clean_tr = keep & ~artifact          # M30 says train without artifacts is better
y_va, clean_va = y[va].to_numpy(), ~artifact[va]

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
pred = np.full(va.sum(), np.nan)
for a in np.unique(apt):
    fit, ev = clean_tr & (apt == a) & ~es, clean_tr & (apt == a) & es
    sel = apt_va == a
    if fit.sum() < MIN_TRAIN or not sel.any():
        continue
    m = lgb.train(tuned(a), lgb.Dataset(X[fit], y[fit].clip(upper=CAP), categorical_feature=cat),
                  num_boost_round=6000, valid_sets=[lgb.Dataset(X[ev], y[ev].clip(upper=CAP))],
                  callbacks=[lgb.early_stopping(150, verbose=False)])
    pred[sel] = m.predict(X[va][sel], num_iteration=m.best_iteration)
pred = np.clip(pred, 60, CAP)
base_clean = rmse(y_va[clean_va], pred[clean_va])
print("baseline (train nettoye) sur holdout NETTOYE: %.1f" % base_clean, flush=True)

print("\n=== segment non-apparie : modele vs constante, metrique NETTOYEE ===")
for a in ["LIRF", "LFPG", "EGLL", "LTFM", "LSZH", "EHAM"]:
    seg_tr = clean_tr & (apt == a) & unm
    seg_va = (apt_va == a) & unm[va] & clean_va
    if seg_va.sum() < 20 or seg_tr.sum() < 100:
        continue
    ys = y[seg_tr]
    P = float((ys > THR).mean())
    const = float(P * ys[ys > THR].mean() + (1 - P) * ys[ys <= THR].mean()) if P > 0 else float(ys.mean())
    yv, pv = y_va[seg_va], pred[seg_va]
    best = min(((al, rmse(yv, al * pv + (1 - al) * const)) for al in np.arange(0, 1.01, 0.25)),
               key=lambda t: t[1])
    print("  %s (n=%d, ext=%d, P_train=%.3f, const=%.0fs): modele=%.1f constante=%.1f -> alpha*=%.2f (%.1f)"
          % (a, seg_va.sum(), int((yv > THR).sum()), P, const,
             rmse(yv, pv), rmse(yv, np.full(len(yv), const)), best[0], best[1]), flush=True)

print("\n=== effet global d'un plafond sur les predictions ===")
for capv in [86400, 20000, 10000, 7200, 5400, 3600]:
    print("  plafond %6ds : holdout nettoye = %.1f" % (capv, rmse(y_va[clean_va], np.minimum(pred, capv)[clean_va])), flush=True)
log_experiment("M32 segment extreme sur metrique nettoyee", model="per-airport tuned LGBM",
               features="full, train sans artefacts", val_rmse=base_clean,
               notes="re-examen de la decision M21", keep="")
print("done", flush=True)
