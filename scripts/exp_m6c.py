"""M6c: mixture with Platt-scaled (logistic) calibration on the ES window.
Smooth parametric calibration -> no isotonic plateaus (the source of July false
positives), no cross-fold score incomparability (single classifier).
Reuses cached regressor predictions (m6_preds.npz)."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.linear_model import LogisticRegression

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
tr, va, es = meta["tr"].to_numpy(), meta["va"].to_numpy(), meta["es"].to_numpy()
vp = meta.loc[va, "val_part"].to_numpy()
y_va = y[va].to_numpy()
THR = 7200
lab = ((y > THR) & (y < 200000)).astype(int).to_numpy()

CLF_PARAMS = dict(objective="binary", metric="binary_logloss", learning_rate=0.05,
                  num_leaves=63, min_data_in_leaf=50, feature_fraction=0.9,
                  num_threads=0, verbosity=-1, seed=42)

fit_m = tr & ~es
clf = lgb.train(CLF_PARAMS, lgb.Dataset(X[fit_m], lab[fit_m], categorical_feature=cat_cols),
                num_boost_round=2000,
                valid_sets=[lgb.Dataset(X[tr & es], lab[tr & es])],
                callbacks=[lgb.early_stopping(100, verbose=False)])
print(f"clf iters={clf.best_iteration} | ES positives={lab[tr & es].sum()}")

def logit(p):
    p = np.clip(p, 1e-7, 1 - 1e-7)
    return np.log(p / (1 - p))

s_es = logit(clf.predict(X[tr & es], num_iteration=clf.best_iteration))
platt = LogisticRegression(C=1e6)
platt.fit(s_es.reshape(-1, 1), lab[tr & es])
s_va = logit(clf.predict(X[va], num_iteration=clf.best_iteration))
p_va = platt.predict_proba(s_va.reshape(-1, 1))[:, 1]
print(f"p_va mean={p_va.mean():.5f} (true {(y_va > THR).mean():.5f}) max={p_va.max():.3f} "
      f"| jan mean={p_va[vp == 'jan'].mean():.5f} jul mean={p_va[vp == 'jul'].mean():.5f}")

pA_l, pB = d["pA_l_va"], d["pB_va"]
mix_l = ((1 - p_va) * pA_l + p_va * pB).clip(60, 86400)
rep_l = rmse_report(y_va, mix_l, vp)
print(f"M6c mixture LGBM (Platt): overall={rep_l['overall']:.1f} jan={rep_l['jan']:.1f} jul={rep_l['jul']:.1f}")

normal = y.between(30, THR).to_numpy() & tr
esn_rows = normal & es
y_es = y[esn_rows].to_numpy()
ws = np.linspace(0, 1, 21)
scores = [rmse(y_es, w * d["pA_c_es"] + (1 - w) * d["pA_l_es"]) for w in ws]
w = float(ws[int(np.argmin(scores))])
pA_blend = w * d["pA_c_va"] + (1 - w) * pA_l
mix_b = ((1 - p_va) * pA_blend + p_va * pB).clip(60, 86400)
rep_b = rmse_report(y_va, mix_b, vp)
print(f"M6c mixture blend (w_cb={w:.2f}): overall={rep_b['overall']:.1f} jan={rep_b['jan']:.1f} jul={rep_b['jul']:.1f}")

best = min([("LGBM", rep_l), ("blend", rep_b)], key=lambda kv: kv[1]["overall"])
log_experiment(f"M6c Platt mixture ({best[0]})", model="mixture Platt-cal",
               features=f"M5 features; Platt on ES window; w_cb={w:.2f}",
               val_rmse=best[1]["overall"], val_rmse_jan=best[1]["jan"], val_rmse_jul=best[1]["jul"],
               notes=f"p_max={p_va.max():.3f}", keep="")
print("done", flush=True)
