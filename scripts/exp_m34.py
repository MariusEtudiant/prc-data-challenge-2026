"""M34: target treatment / loss function.

The target is strongly right-skewed and heteroscedastic (variance grows with the
mean), while L2 assumes constant variance. GLM-style objectives (Poisson, Tweedie,
Gamma) still predict the conditional mean - the RMSE optimum - but weight errors
relative to the expected magnitude. log1p+L2 targets a different functional and
needs Duan's smearing correction to come back to the mean.

Evaluated on the CLEANED mirror (our corrected target metric) and the raw one.
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
X = pd.concat([pd.read_parquet(PROC / f) for f in
               ["X_v2_mirror_t.parquet", "flightctx_train2025.parquet",
                "depregime_train2025.parquet", "aobtq_train2025.parquet",
                "anomreg_train2025.parquet"]], axis=1)
meta = pd.read_parquet(PROC / "meta_train2025.parquet")
y = meta["y"].astype("float64")
tr, va, es = meta["tr"].to_numpy(), meta["va"].to_numpy(), meta["es"].to_numpy()
vp = meta.loc[va, "val_part"].to_numpy()
ok = (y > 0).to_numpy() & (y < 200000).to_numpy()
art = np.zeros(len(y), bool)
for k in (1, 2, 3):
    art |= (y - k * 86400).abs().to_numpy() < 3600
ok &= ~art
apt = X["AIRPORT"].astype(str).to_numpy(); apt_va = apt[va]
y_va = y[va].to_numpy()
clean = ~art[va]
cat = [c for c in X.columns if X[c].dtype.name == "category"]

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

VARIANTS = [
    ("L2 (reference)",   dict(objective="regression"),                       None),
    ("Poisson",          dict(objective="poisson"),                          None),
    ("Tweedie p=1.1",    dict(objective="tweedie", tweedie_variance_power=1.1), None),
    ("Tweedie p=1.3",    dict(objective="tweedie", tweedie_variance_power=1.3), None),
    ("Tweedie p=1.5",    dict(objective="tweedie", tweedie_variance_power=1.5), None),
    ("Gamma",            dict(objective="gamma"),                            None),
    ("log1p + smearing", dict(objective="regression"),                       "log1p"),
]

for tag, obj, transform in VARIANTS:
    pred = np.full(va.sum(), np.nan)
    for a in np.unique(apt):
        f, s = ok & tr & (apt == a) & ~es, ok & tr & (apt == a) & es
        sel = apt_va == a
        if f.sum() < MIN_TRAIN or not sel.any():
            continue
        p = dict(tuned(a)); p.update(obj); p["metric"] = "rmse"
        yt_f, yt_s = y[f].clip(upper=CAP), y[s].clip(upper=CAP)
        if transform == "log1p":
            yt_f, yt_s = np.log1p(yt_f), np.log1p(yt_s)
        m = lgb.train(p, lgb.Dataset(X[f], yt_f, categorical_feature=cat),
                      num_boost_round=6000, valid_sets=[lgb.Dataset(X[s], yt_s)],
                      callbacks=[lgb.early_stopping(150, verbose=False)])
        raw = m.predict(X[va][sel], num_iteration=m.best_iteration)
        if transform == "log1p":
            resid = np.log1p(y[f].clip(upper=CAP)) - m.predict(X[f], num_iteration=m.best_iteration)
            smear = float(np.mean(np.exp(resid)))       # Duan's smearing estimator
            raw = np.expm1(raw) * smear
        pred[sel] = raw
    pred = np.clip(pred, 60, CAP)
    rep = rmse_report(y_va, pred, vp)
    print("%-18s miroir NETTOYE=%.1f | brut=%.1f (jan %.1f jul %.1f)" % (
        tag, rmse(y_va[clean], pred[clean]), rep["overall"], rep["jan"], rep["jul"]), flush=True)
    log_experiment("M34 " + tag, model="per-airport LGBM", features="full",
                   val_rmse=rep["overall"], val_rmse_jan=rep["jan"], val_rmse_jul=rep["jul"],
                   notes="nettoye=%.1f" % rmse(y_va[clean], pred[clean]), keep="")
print("done", flush=True)
