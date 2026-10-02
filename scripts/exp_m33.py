"""M33: does the ambient anomaly-regime signal help predict episodic extremes?

The benefit targets EHAM Jan 2026 (arrival anomaly rate 61x its 2025 level), which
the usual mirror cannot show. So we validate on the analogous episode inside 2025:
February, where 21.6% of unmatched departures were extreme against ~4% normally.

  test 1  hold out February 2025 (the regime episode) - the mechanism test
  test 2  usual mirror holdout - a do-no-harm check
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
base = [pd.read_parquet(PROC / f) for f in
        ["X_v2_mirror_t.parquet", "flightctx_train2025.parquet",
         "depregime_train2025.parquet", "aobtq_train2025.parquet"]]
X = pd.concat(base, axis=1)
A = pd.read_parquet(PROC / "anomreg_train2025.parquet")
XA = pd.concat([X, A], axis=1)
meta = pd.read_parquet(PROC / "meta_train2025.parquet")
y = meta["y"].astype("float64")
tr, va, es = meta["tr"].to_numpy(), meta["va"].to_numpy(), meta["es"].to_numpy()
vp = meta.loc[va, "val_part"].to_numpy()
ok = (y > 0).to_numpy() & (y < 200000).to_numpy()
art = np.zeros(len(y), bool)
for k in (1, 2, 3):
    art |= (y - k * 86400).abs().to_numpy() < 3600
ok &= ~art
apt = X["AIRPORT"].astype(str).to_numpy()
month = X["month"].to_numpy()

def tuned(a):
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

def fit_eval(XX, fit_mask, hold_mask, stop_mask, tag):
    cat = [c for c in XX.columns if XX[c].dtype.name == "category"]
    pred = np.full(hold_mask.sum(), np.nan)
    hold_apt = apt[hold_mask]
    for a in np.unique(apt):
        f, s = fit_mask & (apt == a), stop_mask & (apt == a)
        sel = hold_apt == a
        if f.sum() < MIN_TRAIN or not sel.any() or s.sum() < 200:
            continue
        m = lgb.train(tuned(a), lgb.Dataset(XX[f], y[f].clip(upper=CAP), categorical_feature=cat),
                      num_boost_round=6000, valid_sets=[lgb.Dataset(XX[s], y[s].clip(upper=CAP))],
                      callbacks=[lgb.early_stopping(150, verbose=False)])
        pred[sel] = m.predict(XX[hold_mask][sel], num_iteration=m.best_iteration)
    return np.clip(pred, 60, CAP)

# ---- test 1 : February 2025 as the regime episode
feb = ok & (month == 2)
fit1 = ok & ~np.isin(month, [1, 2]) & ~es
stop1 = ok & es
y_feb = y[feb].to_numpy()
print("=== TEST 1 : fevrier 2025 en holdout (%d vols, %d extremes >1h) ===" % (
    feb.sum(), int((y_feb > 3600).sum())), flush=True)
for XX, tag in [(X, "sans regime"), (XA, "AVEC regime")]:
    p = fit_eval(XX, fit1, feb, stop1, tag)
    m_ext = y_feb > 3600
    print("  %-12s RMSE fevrier=%.1f | sur les %d extremes=%.1f | sur le reste=%.1f" % (
        tag, rmse(y_feb, p), m_ext.sum(), rmse(y_feb[m_ext], p[m_ext]), rmse(y_feb[~m_ext], p[~m_ext])), flush=True)

# ---- test 2 : usual mirror, do-no-harm
print("\n=== TEST 2 : miroir habituel ===", flush=True)
fit2, stop2 = ok & tr & ~es, ok & tr & es
y_va = y[va].to_numpy()
for XX, tag in [(X, "sans regime"), (XA, "AVEC regime")]:
    p = fit_eval(XX, fit2, va, stop2, tag)
    r = rmse_report(y_va, p, vp)
    print("  %-12s miroir=%.1f (jan %.1f jul %.1f)" % (tag, r["overall"], r["jan"], r["jul"]), flush=True)
    if tag == "AVEC regime":
        log_experiment("M33 regime d'anomalie (arrivees)", model="per-airport tuned LGBM",
                       features="full + arr_anom_rate", val_rmse=r["overall"],
                       val_rmse_jan=r["jan"], val_rmse_jul=r["jul"], keep="")
print("done", flush=True)
