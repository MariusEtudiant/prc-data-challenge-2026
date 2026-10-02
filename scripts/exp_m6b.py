"""M6b: fix the mixture's probability calibration with out-of-fold (OOF) predictions
over the full training year (541 positives instead of ~50 in the Dec window).
Reuses saved mA/mB/CatBoost predictions from the staged M6 run (m6_preds.npz)."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.isotonic import IsotonicRegression

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from dc26 import log_experiment
from dc26.validation import rmse_report, rmse

PROC = ROOT / "data" / "processed"
X = pd.read_parquet(PROC / "X_train2025.parquet")
meta = pd.read_parquet(PROC / "meta_train2025.parquet")
d = np.load(ROOT / "experiments" / "m6_preds.npz")
cat_cols = [c for c in X.columns if X[c].dtype.name == "category"]

y = meta["y"].astype("float64")
tr = meta["tr"].to_numpy()
va = meta["va"].to_numpy()
vp = meta.loc[va, "val_part"].to_numpy()
y_va = y[va].to_numpy()
THR = 7200
lab = ((y > THR) & (y < 200000)).astype(int).to_numpy()
print(f"positives in train: {lab[tr].sum()}")

CLF_PARAMS = dict(objective="binary", metric="binary_logloss", learning_rate=0.05,
                  num_leaves=63, min_data_in_leaf=50, feature_fraction=0.9,
                  num_threads=0, verbosity=-1, seed=42)

# --- OOF scores over the training rows (6 folds by month blocks)
# NOTE: no early stopping — with 541 positives / 1.8M rows, logloss-ES stops at 1-8
# iterations and yields near-constant scores. Fixed budget instead.
N_ROUNDS = 150
month = X["month"].to_numpy()
folds = [(1, 2), (3, 4), (5, 6), (7, 8), (9, 10), (11, 12)]
oof = np.full(len(X), np.nan)
for months in folds:
    te_m = tr & np.isin(month, months)
    fit_m = tr & ~np.isin(month, months)
    m = lgb.train(dict(CLF_PARAMS), lgb.Dataset(X[fit_m], lab[fit_m], categorical_feature=cat_cols),
                  num_boost_round=N_ROUNDS)
    oof[te_m] = m.predict(X[te_m])
    print(f"fold {months}: done", flush=True)

iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
iso.fit(oof[tr], lab[tr])

# --- final classifier on all training rows (same fixed budget)
n_final = N_ROUNDS
clf = lgb.train(dict(CLF_PARAMS), lgb.Dataset(X[tr], lab[tr], categorical_feature=cat_cols),
                num_boost_round=n_final)
p_va = iso.predict(clf.predict(X[va]))
true_rate = (y_va > THR).mean()
print(f"final clf rounds={n_final} | p_va mean={p_va.mean():.5f} (true {true_rate:.5f}) "
      f"max={p_va.max():.3f} | jan mean={p_va[vp=='jan'].mean():.5f} jul mean={p_va[vp=='jul'].mean():.5f}")

# --- mixtures with saved regressor predictions
pA_l, pB = d["pA_l_va"], d["pB_va"]
mix_l = ((1 - p_va) * pA_l + p_va * pB).clip(60, 86400)
rep_l = rmse_report(y_va, mix_l, vp)
print(f"M6b mixture LGBM (OOF cal): overall={rep_l['overall']:.1f} jan={rep_l['jan']:.1f} jul={rep_l['jul']:.1f}")

# blend weight re-optimized on ES rows (reuse saved ES preds)
normal = y.between(30, THR).to_numpy() & tr
esn_rows = normal & meta["es"].to_numpy()
y_es = y[esn_rows].to_numpy()
ws = np.linspace(0, 1, 21)
scores = [rmse(y_es, w * d["pA_c_es"] + (1 - w) * d["pA_l_es"]) for w in ws]
w = float(ws[int(np.argmin(scores))])
pA_blend = w * d["pA_c_va"] + (1 - w) * pA_l
mix_b = ((1 - p_va) * pA_blend + p_va * pB).clip(60, 86400)
rep_b = rmse_report(y_va, mix_b, vp)
print(f"M6b mixture blend (w_cb={w:.2f}): overall={rep_b['overall']:.1f} jan={rep_b['jan']:.1f} jul={rep_b['jul']:.1f}")

best = min([("LGBM", rep_l), ("blend", rep_b)], key=lambda kv: kv[1]["overall"])
log_experiment(f"M6b OOF-calibrated mixture ({best[0]})", model="mixture OOF-cal",
               features="M5 features; OOF isotonic (6 folds)",
               val_rmse=best[1]["overall"], val_rmse_jan=best[1]["jan"], val_rmse_jul=best[1]["jul"],
               notes=f"541 positives for calibration; w_cb={w:.2f}", keep="KEEP")
print("done", flush=True)
