"""M57: adversarial validation - can a classifier tell 2025 training rows from
2026 ranking rows? We sit at the noise floor on the 2025 mirror yet 10-15 RMSE
points behind the leader on the 2026 test. Distribution shift is the obvious
suspect and we have never measured it.

AUC ~0.5 means no shift. High AUC means the test is a different population, and
the driving features tell us which.
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
PROC = ROOT / "data" / "processed"
rng = np.random.default_rng(42)

FILES_T = ["X_v2_full_t.parquet", "flightctx_train2025.parquet", "depregime_train2025.parquet",
           "aobtq_full2025.parquet", "anomreg_train2025.parquet"]
FILES_R = ["X_v2_ranking_t.parquet", "flightctx_ranking2026.parquet",
           "depregime_ranking2026.parquet", "aobtq_ranking2026.parquet",
           "anomreg_ranking2026.parquet"]
from dc26.pipeline import align_categories
X = pd.concat([pd.read_parquet(PROC / f) for f in FILES_T], axis=1)
Xr = align_categories(pd.concat([pd.read_parquet(PROC / f) for f in FILES_R], axis=1), X)
# drop calendar columns: they trivially separate 2025 from 2026 and say nothing
drop = [c for c in ("doy", "month") if c in X.columns]
X, Xr = X.drop(columns=drop), Xr.drop(columns=drop)
print("colonnes retirees (calendaires): %s | %d variables restantes" % (drop, X.shape[1]))

n = 120000
i1 = rng.choice(len(X), min(n, len(X)), replace=False)
i2 = rng.choice(len(Xr), min(n, len(Xr)), replace=False)
A = pd.concat([X.iloc[i1], Xr.iloc[i2]], ignore_index=True)
lab = np.r_[np.zeros(len(i1)), np.ones(len(i2))]
cat = [c for c in A.columns if A[c].dtype.name == "category"]
perm = rng.permutation(len(A))
A, lab = A.iloc[perm].reset_index(drop=True), lab[perm]
cut = int(0.7 * len(A))

m = lgb.train(dict(objective="binary", metric="auc", learning_rate=0.05, num_leaves=63,
                   feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1,
                   verbosity=-1, num_threads=0, seed=42),
              lgb.Dataset(A[:cut], lab[:cut], categorical_feature=cat),
              num_boost_round=600,
              valid_sets=[lgb.Dataset(A[cut:], lab[cut:])],
              callbacks=[lgb.early_stopping(50, verbose=False)])
auc = roc_auc_score(lab[cut:], m.predict(A[cut:], num_iteration=m.best_iteration))
print("\nAUC 2025 vs 2026 = %.4f  (0.50 = distributions identiques)" % auc)

imp = pd.Series(m.feature_importance("gain"), index=A.columns).sort_values(ascending=False)
print("\n=== variables qui separent le plus les deux annees ===")
print("  %-30s %10s %12s %12s" % ("variable", "gain %", "moy. 2025", "moy. 2026"))
for c in imp.head(12).index:
    if A[c].dtype.name == "category":
        a, b = "categorielle", ""
    else:
        a = "%.1f" % np.nanmean(X[c].to_numpy(dtype="float64"))
        b = "%.1f" % np.nanmean(Xr[c].to_numpy(dtype="float64"))
    print("  %-30s %9.1f%% %12s %12s" % (c, 100 * imp[c] / imp.sum(), a, b))
