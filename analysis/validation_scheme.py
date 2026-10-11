"""T01: the validation scheme the top teams use, and what it changes for us.

Our mirror withholds January AND July, so the validating model has no exposure to
either scored month - while the production model has both. GREKI's scheme, shared on
the Discord, validates on the scored months themselves: "train on January, test on
July, and vice versa; 12-month CV and random or day-based folds misled us". That
matches our own history: four consecutively validated improvements failed to transfer.

Implemented as the honest analogue of production: the model predicting July is trained
on all of 2025 except July (so it has January), and the model predicting January is
trained on all of 2025 except January (so it has July). Each month is then predicted by
a model with eleven months of data, like production's twelve, instead of ten with
neither scored month.

Compares, on the same rows, the number our old mirror reports against the number this
scheme reports - for the plain per-airport LGBM. If they differ materially, every
feature decision taken on the old scheme is suspect.
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

CAP, MIN_TRAIN = 86400, 20000
SEEDS = [42, 1337, 2024]
FILES = ["X_v2_mirror_t.parquet", "flightctx_train2025.parquet", "depregime_train2025.parquet",
         "aobtq_train2025.parquet", "anomreg_train2025.parquet"]

X = pd.concat([pd.read_parquet(PROC / f) for f in FILES], axis=1)
meta = pd.read_parquet(PROC / "meta_train2025.parquet")
y = meta["y"].astype("float64")
es = meta["es"].to_numpy()
valid = (y > 0).to_numpy() & (y < 200000).to_numpy()
art = np.zeros(len(y), bool)
for k in (1, 2, 3):
    art |= (y - k * 86400).abs().to_numpy() < 3600
cat = [c for c in X.columns if X[c].dtype.name == "category"]
apt = X["AIRPORT"].astype(str).to_numpy()
mvt = pd.read_parquet(PROC / "movements_2025.parquet", columns=["PHASE_mvt", "MVT_TIME_UTC_mvt"])
t = mvt[mvt["PHASE_mvt"] == "DEP"]["MVT_TIME_UTC_mvt"].reset_index(drop=True)
mon = t.dt.month.to_numpy()
yv_all = y.to_numpy()


def params(a, seed):
    from dc26.mixture import REG_PARAMS
    p = EXP / ("optuna_" + a + ".json")
    b = dict(objective="regression", metric="rmse", verbosity=-1, num_threads=0, bagging_freq=1,
             seed=seed, bagging_seed=seed + 1, feature_fraction_seed=seed + 2,
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


def run(train_mask, out_mask, tag):
    pred = np.full(len(y), np.nan)
    t0 = time.time()
    for a in np.unique(apt):
        f = train_mask & (apt == a) & ~es
        s = train_mask & (apt == a) & es
        o = out_mask & (apt == a)
        if f.sum() < MIN_TRAIN or s.sum() < 300 or not o.any():
            continue
        P = [lgb.train(params(a, sd),
                       lgb.Dataset(X[f], y[f].clip(upper=CAP), categorical_feature=cat,
                                   params={"feature_pre_filter": False, "max_bin": 255}),
                       num_boost_round=6000,
                       valid_sets=[lgb.Dataset(X[s], y[s].clip(upper=CAP))],
                       callbacks=[lgb.early_stopping(150, verbose=False)]) for sd in SEEDS]
        pred[o] = np.mean([m.predict(X[o], num_iteration=m.best_iteration) for m in P], 0)
    print("  %s fait (%.0f min)" % (tag, (time.time() - t0) / 60), flush=True)
    return np.clip(pred, 60, CAP)


base = valid & ~art           # artefacts out of training, as in our shipped model
JAN, JUL = mon == 1, mon == 7

# --- scheme A: our old mirror. Train on the ten other months, predict both.
old = run(base & ~JAN & ~JUL, base & (JAN | JUL), "ancien miroir (ni jan ni jul au train)")
# --- scheme B: cross-month. Each scored month predicted by a model that saw the other.
pj = run(base & ~JUL, base & JUL, "croise: train sans juillet -> predit juillet")
pi = run(base & ~JAN, base & JAN, "croise: train sans janvier -> predit janvier")
new = np.where(JUL, pj, np.where(JAN, pi, np.nan))

np.save(EXP / "t01_old.npy", old)
np.save(EXP / "t01_new.npy", new)

g = base & (JAN | JUL) & np.isfinite(old) & np.isfinite(new)
print("\n=== memes %d lignes, deux schemas de validation ===" % g.sum())
print("  %-26s %10s %10s %10s" % ("", "global", "janvier", "juillet"))
for lab, p in (("ancien miroir", old), ("croise jan<->jul", new)):
    print("  %-26s %10.2f %10.2f %10.2f" % (
        lab, rmse(yv_all[g], p[g]),
        rmse(yv_all[g & JAN], p[g & JAN]), rmse(yv_all[g & JUL], p[g & JUL])))
print("\n  (metrique BRUTE, artefacts compris, celle que le classement applique)")
cl = g & ~art
print("  nettoye : ancien=%.2f | croise=%.2f" % (rmse(yv_all[cl], old[cl]), rmse(yv_all[cl], new[cl])))

print("\n=== par aeroport (nettoye) ===")
print("  %-8s %10s %10s %9s" % ("aero", "ancien", "croise", "ecart"))
for a in sorted(set(apt)):
    m = cl & (apt == a)
    if m.sum() < 500:
        continue
    ro, rn = rmse(yv_all[m], old[m]), rmse(yv_all[m], new[m])
    print("  %-8s %10.1f %10.1f %+9.1f" % (a, ro, rn, rn - ro))
