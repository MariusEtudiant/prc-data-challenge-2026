"""M68: can a model predict OUR residual? That is the gap, made visible.

If the blend has extracted everything, y - pred is noise with respect to every
observable and a GBDT fitted on it generalises to nothing (R2 <= 0). Any real
out-of-sample R2 is signal we left on the table, and the gain importances say
where it lives.

Honest split: fitted on half the holdout, measured on the other half, with the
two halves separated in TIME (odd/even days) so nothing leaks through the
neighbouring-flight features.
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import lightgbm as lgb

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
PROC, EXP = ROOT / "data" / "processed", ROOT / "experiments"
from dc26.validation import rmse

X = pd.concat([pd.read_parquet(PROC / f) for f in
               ["X_v2_mirror_t.parquet", "flightctx_train2025.parquet",
                "depregime_train2025.parquet", "aobtq_train2025.parquet",
                "anomreg_train2025.parquet"]], axis=1)
meta = pd.read_parquet(PROC / "meta_train2025.parquet")
y = meta["y"].astype("float64").to_numpy()
va = meta["va"].to_numpy()
p = np.load(EXP / "m49_blend_va.npy")

Xv = X[va].reset_index(drop=True)
yv = y[va]
art = np.zeros(len(y), bool)
for k in (1, 2, 3):
    art |= np.abs(y - k * 86400) < 3600
# ordinary traffic only: the extremes are a separate, already-closed question
keep = np.isfinite(p) & ~art[va] & (yv < 3600)
res = yv - p
doy = Xv["doy"].to_numpy()
A = keep & (doy % 2 == 0)
B = keep & (doy % 2 == 1)
print("trafic ordinaire: %d vols | ajustement sur %d jours pairs, mesure sur %d jours impairs" % (
    keep.sum(), A.sum(), B.sum()))
print("residu: ecart-type=%.1f  moyenne=%.1f" % (res[keep].std(), res[keep].mean()))

cat = [c for c in Xv.columns if Xv[c].dtype.name == "category"]
m = lgb.train(dict(objective="regression", metric="rmse", learning_rate=0.03,
                   num_leaves=63, min_data_in_leaf=200, feature_fraction=0.7,
                   bagging_fraction=0.7, bagging_freq=1, lambda_l2=5.0,
                   verbosity=-1, num_threads=0, seed=42),
              lgb.Dataset(Xv[A], res[A], categorical_feature=cat),
              num_boost_round=3000,
              valid_sets=[lgb.Dataset(Xv[B], res[B])],
              callbacks=[lgb.early_stopping(100, verbose=False)])
q = m.predict(Xv[B], num_iteration=m.best_iteration)
sst = ((res[B] - res[B].mean()) ** 2).sum()
r2 = 1 - ((res[B] - q) ** 2).sum() / sst
print("\niterations retenues=%d" % m.best_iteration)
print("R2 hors echantillon sur le residu = %.4f" % r2)
print("RMSE trafic ordinaire : %.2f -> %.2f en appliquant la correction (%+.2f)" % (
    rmse(yv[B], p[B]), rmse(yv[B], p[B] + q), rmse(yv[B], p[B] + q) - rmse(yv[B], p[B])))

if r2 > 0.002:
    imp = pd.Series(m.feature_importance("gain"), index=Xv.columns).sort_values(ascending=False)
    print("\n=== ou vit le signal restant ===")
    for c in imp.head(15).index:
        print("  %-34s %6.1f%%" % (c, 100 * imp[c] / imp.sum()))
else:
    print("\naucun signal exploitable dans le residu : le modele a tout pris.")
