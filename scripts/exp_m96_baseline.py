"""M96: the new reference point, on the mirror rebuilt for the regenerated July.

Everything measured before 2026-09-04 was validated against a 70/30 January-dominant
test that no longer exists. The mirror now holds out January AND July 2025 for all ten
airports, 44.6/55.4, against the real test's 44.3/55.7.

Two variants in one run, so the reference is established and M86's runway-config
features are re-validated on the corrected mirror at the same time.

Caveat carried openly: the per-airport Optuna parameters were tuned when July sat in
the training period. They are stale here - usable as a starting point, worth redoing
once the reference exists.
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
BASE = ["X_v2_mirror_t.parquet", "flightctx_train2025.parquet", "depregime_train2025.parquet",
        "aobtq_train2025.parquet", "anomreg_train2025.parquet"]

X = pd.concat([pd.read_parquet(PROC / f) for f in BASE], axis=1)
E = pd.read_parquet(PROC / "rwycfg_train2025.parquet")
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
vp = meta.loc[va, "val_part"].to_numpy()
jan, jul = vp == "jan", vp == "jul"
print("miroir: train=%d val=%d (jan %.1f%% / jul %.1f%%) | artefacts ecartes=%d" % (
    (ok & tr).sum(), va.sum(), 100 * jan.mean(), 100 * jul.mean(), art[va].sum()), flush=True)


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
res = {}
for tag, XX in (("reference", X), ("+ config pistes", pd.concat([X, E], axis=1))):
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
                          lgb.Dataset(XX[f], y[f].clip(upper=CAP), categorical_feature=cat),
                          num_boost_round=6000,
                          valid_sets=[lgb.Dataset(XX[s], y[s].clip(upper=CAP))],
                          callbacks=[lgb.early_stopping(150, verbose=False)])
            P.append(m.predict(XX[va][sv], num_iteration=m.best_iteration))
        pred[sv] = np.mean(P, 0)
        c = sv & clean
        per[a] = (rmse(y_va[c], np.clip(pred[c], 60, CAP)),
                  rmse(y_va[c & jan], np.clip(pred[c & jan], 60, CAP)),
                  rmse(y_va[c & jul], np.clip(pred[c & jul], 60, CAP)))
    pred = np.clip(pred, 60, CAP)
    g = np.isfinite(pred)
    res[tag] = (rmse(y_va[clean & g], pred[clean & g]),
                rmse(y_va[clean & g & jan], pred[clean & g & jan]),
                rmse(y_va[clean & g & jul], pred[clean & g & jul]), per)
    np.save(ROOT / "experiments" / ("m96_%s.npy" % tag.split()[0].strip("+ ")), pred)
    print("%-16s nettoye=%7.2f | janvier=%7.2f | juillet=%7.2f  (%.0f min)" % (
        tag, res[tag][0], res[tag][1], res[tag][2], (time.time() - t0) / 60), flush=True)

b, w = res["reference"], res["+ config pistes"]
print("\n=== par aeroport (nettoye | janvier | juillet), variante de reference ===")
print("  %-8s %10s %10s %10s %12s" % ("aero", "global", "janvier", "juillet", "gain config"))
for a in sorted(b[3]):
    print("  %-8s %10.1f %10.1f %10.1f %+12.1f" % (
        a, b[3][a][0], b[3][a][1], b[3][a][2], w[3][a][0] - b[3][a][0]))
print("\nconfig pistes : %+.2f nettoye | %+.2f janvier | %+.2f juillet" % (
    w[0] - b[0], w[1] - b[1], w[2] - b[2]))
for tag in res:
    log_experiment("M96 %s (nouveau miroir juillet complet)" % tag, model="LGBM par aeroport x3",
                   features="104 var" if "config" in tag else "100 var",
                   val_rmse=res[tag][0], val_rmse_jan=res[tag][1], val_rmse_jul=res[tag][2],
                   notes="miroir 44.6/55.4 apres regeneration de juillet", keep="")
