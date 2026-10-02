"""R04: we have been optimising the wrong metric since day one.

Every decision of the last four days - features, models, blend weights, architecture -
was taken on a "cleaned" mirror that removes the day-offset artefacts. The leaderboard
removes nothing. On the same predictions our mirror reads 294.38 cleaned and 391.47
raw: seven flights out of 344,412 account for 97 points. So a third of the real error
has been invisible to every comparison we ran.

The same exclusion is applied to TRAINING, which means the model learns that the
probability of a 24-hour taxi is zero on a population where it is around 15%. Under a
squared-error metric that is not conservative, it is simply wrong: missing an 86,400s
flight costs 7.3e9 while overshooting a normal one by 17,000s costs 2.9e8.

Four variants, per airport, three seeds, judged on BOTH metrics:
  A  train without artefacts   (what we ship)
  B  train WITH artefacts, target clipped at 86400 as usual
  C  as B, but artefact rows given weight 1/3 (a compromise if B is too violent)
  D  as A, plus an explicit hedge: candidates (no FPL and d_sched > 54000s) are
     pushed to the conditional mean learned on the training rows

The honest question is not which wins the cleaned metric - it is which wins the raw
one, because that is the one the competition scores.
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

CAP, MIN_TRAIN, THR = 86400, 20000, 54000.0
SEEDS = [42, 1337, 2024]
FILES = ["X_v2_mirror_t.parquet", "flightctx_train2025.parquet", "depregime_train2025.parquet",
         "aobtq_train2025.parquet", "anomreg_train2025.parquet"]

X = pd.concat([pd.read_parquet(PROC / f) for f in FILES], axis=1)
meta = pd.read_parquet(PROC / "meta_train2025.parquet")
y = meta["y"].astype("float64")
tr, va, es = meta["tr"].to_numpy(), meta["va"].to_numpy(), meta["es"].to_numpy()
valid = (y > 0).to_numpy() & (y < 200000).to_numpy()
art = np.zeros(len(y), bool)
for k in (1, 2, 3):
    art |= (y - k * 86400).abs().to_numpy() < 3600
cat = [c for c in X.columns if X[c].dtype.name == "category"]
apt = X["AIRPORT"].astype(str).to_numpy()
apt_va, y_va = apt[va], y[va].to_numpy()
clean_va = ~art[va]
ds = X["d_sched"].to_numpy(dtype="float64")
fp = X["has_fpl"].to_numpy()
cand = np.isfinite(ds) & (ds > THR) & (fp == 0)
mu_cand = y[valid & tr & cand].mean() if (valid & tr & cand).sum() > 10 else 18000.0
print("candidats (sans FPL, d_sched>%ds): %d a l entrainement, moyenne conditionnelle %.0f s | %d dans le holdout" % (
    THR, (valid & tr & cand).sum(), mu_cand, (cand & va).sum()), flush=True)
print("artefacts: %d a l entrainement, %d dans le holdout\n" % ((valid & tr & art).sum(), (art & va).sum()), flush=True)


def tuned(a, seed):
    from dc26.mixture import REG_PARAMS
    p = ROOT / "experiments" / ("optuna_" + a + ".json")
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


VAR = {"A sans artefacts": ("excl", None), "B avec artefacts": ("incl", None),
       "C artefacts ponderes 1/3": ("incl", 1 / 3), "D couverture explicite": ("excl", "hedge")}
t0 = time.time()
res = {}
for tag, (mode, opt) in VAR.items():
    pred = np.full(va.sum(), np.nan)
    for a in np.unique(apt):
        base = valid & tr & (apt == a)
        keep = base if mode == "incl" else base & ~art
        f, s = keep & ~es, keep & es
        sv = apt_va == a
        if f.sum() < MIN_TRAIN or s.sum() < 300 or not sv.any():
            continue
        w = None
        if opt == 1 / 3:
            w = np.where(art[f], 1 / 3, 1.0)
        P = []
        for seed in SEEDS:
            dtr = lgb.Dataset(X[f], y[f].clip(upper=CAP), categorical_feature=cat, weight=w,
                              params={"feature_pre_filter": False, "max_bin": 255})
            m = lgb.train(tuned(a, seed), dtr, num_boost_round=6000,
                          valid_sets=[lgb.Dataset(X[s], y[s].clip(upper=CAP))],
                          callbacks=[lgb.early_stopping(150, verbose=False)])
            P.append(m.predict(X[va][sv], num_iteration=m.best_iteration))
        pred[sv] = np.mean(P, 0)
    pred = np.clip(pred, 60, CAP)
    if opt == "hedge":
        h = cand[va] & np.isfinite(pred)
        pred[h] = mu_cand
    g = np.isfinite(pred)
    raw = rmse(y_va[g], pred[g])
    cln = rmse(y_va[g & clean_va], pred[g & clean_va])
    res[tag] = (raw, cln)
    np.save(ROOT / "experiments" / ("r04_%s.npy" % tag.split()[0]), pred)
    print("%-26s BRUT=%8.2f   nettoye=%8.2f   (%.0f min)" % (tag, raw, cln, (time.time() - t0) / 60), flush=True)

b = res["A sans artefacts"]
print("\n%-26s %10s %10s" % ("variante", "d BRUT", "d nettoye"))
for tag, (raw, cln) in res.items():
    print("%-26s %+10.2f %+10.2f" % (tag, raw - b[0], cln - b[1]))
for tag, (raw, cln) in res.items():
    log_experiment("R04 %s" % tag, model="LGBM par aeroport x3", features="full",
                   val_rmse=raw, notes="nettoye %.2f | metrique BRUTE privilegiee" % cln, keep="")
