"""N04: what signal is left in our residual, on the composition that now counts?

The capacity test settled that the models are not the bottleneck: widening the trees
degrades, and the current tuned parameters already beat the defaults everywhere. So
the ceiling sits in the FEATURES, and the residual is where a missing feature shows
itself - if a model fed our own inputs can predict what we got wrong, that is signal
we left on the table, and the gains say where it lives.

M68 ran this on the old mirror (70% January, July at three hubs) and found R2 0.0135
with STAND carrying 38%. The mirror has since been rebuilt for the regenerated July
(44.6/55.4, all ten airports) and we lose sixty points there, so the question is
worth asking again - and this time separately for each month, because January and
July behave differently enough that a pooled answer would hide the interesting half.

Honest split: fitted on odd days, measured on even days, so nothing leaks through the
neighbouring-flight features.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
PROC, EXP = ROOT / "data" / "processed", ROOT / "experiments"
from dc26.validation import rmse

CAP = 86400
FILES = ["X_v2_mirror_t.parquet", "flightctx_train2025.parquet", "depregime_train2025.parquet",
         "aobtq_train2025.parquet", "anomreg_train2025.parquet", "rwycfg_train2025.parquet"]

X = pd.concat([pd.read_parquet(PROC / f) for f in FILES], axis=1)
meta = pd.read_parquet(PROC / "meta_train2025.parquet")
y = meta["y"].astype("float64").to_numpy()
va = meta["va"].to_numpy()
art = np.zeros(len(y), bool)
for k in (1, 2, 3):
    art |= np.abs(y - k * 86400) < 3600

# our submitted blend, rebuilt on the mirror with the v15 weights
n39 = np.load(EXP / "m39_preds.npz")
ng = np.clip(n39["nn_global"], 60, CAP)
na = np.clip(np.where(np.isfinite(n39["nn_airport"]), n39["nn_airport"], n39["nn_global"]), 60, CAP)
V = np.vstack([np.load(EXP / "m96_reference.npy"), np.load(EXP / "v14_cb_mirror.npy"), ng, na])
W = json.loads((EXP / "v15_weights.json").read_text())

Xv = X[va].reset_index(drop=True)
yv = y[va]
apt = Xv["AIRPORT"].astype(str).to_numpy()
pred = np.full(len(yv), np.nan)
for a, w in W.items():
    m = apt == a
    if m.any():
        pred[m] = np.clip(np.array(w) @ np.nan_to_num(V[:, m]), 60, CAP)

mvt = pd.read_parquet(PROC / "movements_2025.parquet", columns=["PHASE_mvt", "MVT_TIME_UTC_mvt"])
t = mvt[mvt["PHASE_mvt"] == "DEP"]["MVT_TIME_UTC_mvt"].reset_index(drop=True)
day, mon = t.dt.day.to_numpy()[va], t.dt.month.to_numpy()[va]

res = yv - pred
cat = [c for c in Xv.columns if Xv[c].dtype.name == "category"]
# ordinary traffic only: the tail is a separate, already-settled question
base = np.isfinite(pred) & ~art[va] & (yv < 7200)
print("holdout exploitable: %d vols | RMSE du melange = %.2f" % (base.sum(), rmse(yv[base], pred[base])))

for lab, sel in (("GLOBAL", base), ("janvier", base & (mon == 1)), ("juillet", base & (mon == 7))):
    A, B = sel & (day % 2 == 0), sel & (day % 2 == 1)
    if B.sum() < 5000:
        continue
    m = lgb.train(dict(objective="regression", metric="rmse", learning_rate=0.03,
                       num_leaves=63, min_data_in_leaf=200, feature_fraction=0.7,
                       bagging_fraction=0.7, bagging_freq=1, lambda_l2=5.0,
                       verbosity=-1, num_threads=0, seed=42, feature_pre_filter=False),
                  lgb.Dataset(Xv[A], res[A], categorical_feature=cat,
                              params={"feature_pre_filter": False}),
                  num_boost_round=3000, valid_sets=[lgb.Dataset(Xv[B], res[B])],
                  callbacks=[lgb.early_stopping(100, verbose=False)])
    q = m.predict(Xv[B], num_iteration=m.best_iteration)
    r2 = 1 - ((res[B] - q) ** 2).sum() / ((res[B] - res[B].mean()) ** 2).sum()
    r0, r1 = rmse(yv[B], pred[B]), rmse(yv[B], pred[B] + q)
    print("\n=== %s : n=%d, ecart-type du residu=%.1f ===" % (lab, sel.sum(), res[sel].std()))
    print("  iterations=%d | R2 hors echantillon sur le residu = %.4f" % (m.best_iteration, r2))
    print("  RMSE %.2f -> %.2f en appliquant la correction (%+.2f)" % (r0, r1, r1 - r0))
    if r2 > 0.002:
        imp = pd.Series(m.feature_importance("gain"), index=Xv.columns).sort_values(ascending=False)
        print("  ou vit le signal restant :")
        for c in imp.head(12).index:
            print("    %-34s %5.1f%%" % (c, 100 * imp[c] / imp.sum()))
    else:
        print("  aucun signal exploitable : le modele a pris ce qu il y avait")
