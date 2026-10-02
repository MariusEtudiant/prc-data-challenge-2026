"""M31: importance weighting for a MEASURED covariate shift.

Unmatched flights (no NM FLIGHT_ID) are 3x more frequent in the January 2026 test
than in training at EHAM (4.96% vs 1.65%) and LSZH (3.00% vs 1.70%). Unlike the
seasonal weighting tested in M18 - which had no measured basis and gave nothing -
this shift is observed directly in the ranking file.

Weight each training row by the ratio test_rate/train_rate of its own segment
(airport x matched/unmatched), the textbook importance-weighting estimator.
"""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd, lightgbm as lgb

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from dc26 import load_ranking_2026, log_experiment
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
y_va = y[va].to_numpy()
apt = X["AIRPORT"].astype(str).to_numpy(); apt_va = apt[va]
unm = (X["has_fpl"].to_numpy() == 0)

rk = load_ranking_2026(); rkd = rk[rk["PHASE_mvt"] == "DEP"]
test_rate = rkd.groupby("AIRPORT")["FLIGHT_ID_mvt"].apply(lambda s: s.isna().mean())
train_rate = pd.Series(unm[keep]).groupby(apt[keep]).mean()

w = np.ones(len(y))
print("ratios test/train du segment non-apparie:", flush=True)
for a in np.unique(apt):
    ttest, ttrain = float(test_rate.get(a, np.nan)), float(train_rate.get(a, np.nan))
    if not np.isfinite(ttest) or not np.isfinite(ttrain) or ttrain <= 0:
        continue
    r_unm = min(ttest / ttrain, 5.0)
    r_m = (1 - ttest) / (1 - ttrain)
    w[(apt == a) & unm] = r_unm
    w[(apt == a) & ~unm] = r_m
    print("  %s: test=%.2f%% train=%.2f%% -> poids non-apparies x%.2f" % (a, 100*ttest, 100*ttrain, r_unm), flush=True)

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

def run(weights, tag):
    pred = np.full(va.sum(), np.nan)
    for a in np.unique(apt):
        fit, ev = keep & (apt == a) & ~es, keep & (apt == a) & es
        sel = apt_va == a
        if fit.sum() < MIN_TRAIN or not sel.any():
            continue
        d = lgb.Dataset(X[fit], y[fit].clip(upper=CAP), categorical_feature=cat,
                        weight=None if weights is None else weights[fit])
        m = lgb.train(tuned(a), d, num_boost_round=6000,
                      valid_sets=[lgb.Dataset(X[ev], y[ev].clip(upper=CAP))],
                      callbacks=[lgb.early_stopping(150, verbose=False)])
        pred[sel] = m.predict(X[va][sel], num_iteration=m.best_iteration)
    pred = np.clip(pred, 60, CAP)
    rep = rmse_report(y_va, pred, vp)
    print("%s: overall=%.1f jan=%.1f jul=%.1f" % (tag, rep["overall"], rep["jan"], rep["jul"]), flush=True)
    return rep, pred

repA, pA = run(None, "M31a sans ponderation")
repB, pB = run(w, "M31b ponderation covariate-shift")
for a in ["EHAM", "LSZH", "EDDF", "EDDM"]:
    s = apt_va == a
    su = s & unm[va]
    print("   [%s] tous: %.1f -> %.1f | non-apparies (n=%d): %.1f -> %.1f" % (
        a, rmse(y_va[s], pA[s]), rmse(y_va[s], pB[s]), su.sum(),
        rmse(y_va[su], pA[su]) if su.sum() else np.nan,
        rmse(y_va[su], pB[su]) if su.sum() else np.nan), flush=True)
log_experiment("M31 importance weighting (shift non-apparies)", model="per-airport tuned LGBM",
               features="full", val_rmse=repB["overall"], val_rmse_jan=repB["jan"],
               val_rmse_jul=repB["jul"], notes="vs %.1f" % repA["overall"],
               keep="KEEP" if repB["overall"] < repA["overall"] else "DROP")
print("done", flush=True)
