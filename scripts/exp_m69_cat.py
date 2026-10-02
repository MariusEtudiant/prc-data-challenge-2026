"""M69: LightGBM's categorical hyperparameters were never tuned, and the residual
analysis (M68) names STAND as 38% of the signal we are leaving behind.

Defaults cap each categorical split at 32 categories (max_cat_threshold) while our
airports carry 142-417 stands, and regularise hard (cat_smooth=10,
min_data_per_group=100). Three settings, same protocol as every other experiment:
fit on training months, early-stop on the ES window, measure on the mirror holdout.
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
from dc26.validation import rmse, rmse_report
from dc26 import log_experiment

CAP, MIN_TRAIN = 86400, 20000

VARIANTS = {
    "V0 defauts actuels": {},
    "V1 catégoriel elargi": dict(max_cat_threshold=256, cat_smooth=5.0,
                                 min_data_per_group=30, cat_l2=5.0),
    "V2 catégoriel libre": dict(max_cat_threshold=512, cat_smooth=1.0,
                                min_data_per_group=10, cat_l2=1.0),
}

X = pd.concat([pd.read_parquet(PROC / f) for f in
               ["X_v2_mirror_t.parquet", "flightctx_train2025.parquet",
                "depregime_train2025.parquet", "aobtq_train2025.parquet",
                "anomreg_train2025.parquet"]], axis=1)
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
apt_va = apt[va]
y_va = y[va].to_numpy()
clean = ~art[va]
vp = meta.loc[va, "val_part"].to_numpy()


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


t0 = time.time()
res = {}
for tag, extra in VARIANTS.items():
    pred = np.full(va.sum(), np.nan)
    per_apt = {}
    for a in np.unique(apt):
        f = ok & tr & (apt == a) & ~es
        s = ok & tr & (apt == a) & es
        sv = apt_va == a
        if f.sum() < MIN_TRAIN or not sv.any():
            continue
        m = lgb.train(dict(tuned(a), **extra),
                      lgb.Dataset(X[f], y[f].clip(upper=CAP), categorical_feature=cat),
                      num_boost_round=6000,
                      valid_sets=[lgb.Dataset(X[s], y[s].clip(upper=CAP))],
                      callbacks=[lgb.early_stopping(150, verbose=False)])
        pred[sv] = m.predict(X[va][sv], num_iteration=m.best_iteration)
        per_apt[a] = rmse(y_va[sv & clean], np.clip(pred[sv & clean], 60, CAP))
    pred = np.clip(pred, 60, CAP)
    g = np.isfinite(pred)
    rc = rmse(y_va[clean & g], pred[clean & g])
    rj = rmse(y_va[clean & g & (vp == "jul")], pred[clean & g & (vp == "jul")])
    res[tag] = (rc, rj, per_apt)
    np.save(ROOT / "experiments" / ("m69_%s.npy" % tag.split()[0]), pred)
    print("%-24s nettoye=%7.2f  juillet=%7.2f   (%.0f min)" % (tag, rc, rj, (time.time() - t0) / 60), flush=True)

b = res["V0 defauts actuels"]
print("\n=== ecart par aeroport (nettoye) ===")
print("  %-8s %10s %10s %10s" % ("aero", "V0", "V1", "V2"))
for a in sorted(b[2]):
    print("  %-8s %10.1f %+9.1f %+9.1f" % (
        a, b[2][a], res["V1 catégoriel elargi"][2].get(a, np.nan) - b[2][a],
        res["V2 catégoriel libre"][2].get(a, np.nan) - b[2][a]))
for tag in ("V1 catégoriel elargi", "V2 catégoriel libre"):
    print("\n%s : %+.2f nettoye, %+.2f juillet" % (tag, res[tag][0] - b[0], res[tag][1] - b[1]))
    log_experiment("M69 %s" % tag, model="LGBM par aeroport", features="full",
                   val_rmse=res[tag][0], notes="juillet %.2f, base V0 %.2f" % (res[tag][1], b[0]), keep="")
