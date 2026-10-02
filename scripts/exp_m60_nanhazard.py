"""M60: does the unseen-NaN state of atxot_month_add_min damage the July prediction?

In the ranking set that feature is missing on exactly the July rows of EDDF/EGLL/
EHAM - 29% of the test - while it is never missing during training. The model's
NaN routing for it was therefore fitted on zero examples.

Our mirror has a July part on the same three airports, so the condition can be
reproduced exactly:
  A  feature present            (what our validation has always measured)
  B  feature NaN on the holdout (what actually happens at test time)
  C  model retrained without it (the candidate fix)
  D  model trained with the feature randomly NaN'd 30% of the time (learned fallback)
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

CAP, MIN_TRAIN, COL = 86400, 20000, "atxot_month_add_min"
rng = np.random.default_rng(42)

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
vp = meta.loc[va, "val_part"].to_numpy()
y_va = y[va].to_numpy()
clean = ~art[va]
jul = vp == "jul"


def tuned(a, seed=42):
    from dc26.mixture import REG_PARAMS
    p = ROOT / "experiments" / ("optuna_" + a + ".json")
    b = dict(objective="regression", metric="rmse", verbosity=-1, num_threads=0,
             bagging_freq=1, seed=seed)
    if not p.exists():
        return dict(REG_PARAMS, seed=seed)
    d = json.loads(p.read_text())
    b.update({k: v for k, v in d.items() if not k.startswith("_")})
    for k in ("num_leaves", "min_data_in_leaf", "max_bin"):
        b[k] = int(b[k])
    return b


def run(tag, drop_col, nan_train_frac, nan_holdout):
    Xw = X.drop(columns=[COL]) if drop_col else X.copy()
    if nan_train_frac:
        m = rng.random(len(Xw)) < nan_train_frac
        Xw.loc[m, COL] = np.nan
    Xh = Xw.copy()
    if nan_holdout and not drop_col:
        h = np.zeros(len(Xw), bool)
        h[np.where(va)[0][jul]] = True
        Xh.loc[h, COL] = np.nan
    cats = [c for c in Xw.columns if Xw[c].dtype.name == "category"]
    pred = np.full(va.sum(), np.nan)
    for a in np.unique(apt):
        f = ok & tr & (apt == a) & ~es
        s = ok & tr & (apt == a) & es
        sv = apt[va] == a
        if f.sum() < MIN_TRAIN or not sv.any():
            continue
        m = lgb.train(tuned(a), lgb.Dataset(Xw[f], y[f].clip(upper=CAP), categorical_feature=cats),
                      num_boost_round=6000,
                      valid_sets=[lgb.Dataset(Xw[s], y[s].clip(upper=CAP))],
                      callbacks=[lgb.early_stopping(150, verbose=False)])
        pred[sv] = m.predict(Xh[va][sv], num_iteration=m.best_iteration)
    pred = np.clip(pred, 60, CAP)
    g = np.isfinite(pred)
    r_all = rmse(y_va[clean & g], pred[clean & g])
    r_jul = rmse(y_va[clean & g & jul], pred[clean & g & jul])
    print("%-52s nettoye=%7.2f  JUILLET=%7.2f" % (tag, r_all, r_jul), flush=True)
    return r_all, r_jul


t0 = time.time()
a = run("A  variable presente (validation historique)", False, 0.0, False)
b = run("B  variable absente sur juillet (condition reelle)", False, 0.0, True)
c = run("C  variable retiree du modele", True, 0.0, False)
d = run("D  entrainee avec 30% d absences simulees, testee absente", False, 0.30, True)
print("\n=== degats et remedes, sur la partie JUILLET ===")
print("  A -> B  degat de la condition reelle : %+.2f" % (b[1] - a[1]))
print("  B -> C  gain en retirant la variable : %+.2f" % (c[1] - b[1]))
print("  B -> D  gain en simulant l absence   : %+.2f" % (d[1] - b[1]))
print("termine en %.0f min" % ((time.time() - t0) / 60))
log_experiment("M60 danger NaN atxot", model="LGBM par aeroport", features="full",
               val_rmse=b[1], notes="A=%.2f B=%.2f C=%.2f D=%.2f (juillet)" % (a[1], b[1], c[1], d[1]),
               keep="")
