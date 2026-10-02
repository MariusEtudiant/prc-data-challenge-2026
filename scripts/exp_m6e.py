"""M6e: classifier gets an explicit airport x no-FPL interaction feature
(extreme signature is airport-dependent: LIRF no-fpl ~30% extreme, hubs ~0.5%),
calibration window extended to Nov 15 (3x positives), Platt scaling.
Regressors reused from cache."""
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

# explicit interaction for the classifier
X["AIRPORT_nofpl"] = np.where(X["has_fpl"].to_numpy() == 0,
                              X["AIRPORT"].astype(str), "MATCHED")
X["AIRPORT_nofpl"] = X["AIRPORT_nofpl"].astype("category")
cat_cols = [c for c in X.columns if X[c].dtype.name == "category"]

y = meta["y"].astype("float64")
tr, va = meta["tr"].to_numpy(), meta["va"].to_numpy()
vp = meta.loc[va, "val_part"].to_numpy()
y_va = y[va].to_numpy()
THR = 7200
lab = ((y > THR) & (y < 200000)).astype(int).to_numpy()

# calibration window: Nov 15 - Dec 31 (still inside mirror-train months)
cal = meta["es"].to_numpy() & tr
# cal = Dec 10-31 window (as M6c)
fit_m = tr & ~cal
print(f"cal window rows={cal.sum():,} positives={lab[cal].sum()}")

CLF_PARAMS = dict(objective="binary", metric="binary_logloss", learning_rate=0.05,
                  num_leaves=63, min_data_in_leaf=50, feature_fraction=0.9,
                  num_threads=0, verbosity=-1, seed=42)
clf = lgb.train(CLF_PARAMS, lgb.Dataset(X[fit_m], lab[fit_m], categorical_feature=cat_cols),
                num_boost_round=250)
print("clf fixed rounds=250")

def logit(p):
    p = np.clip(p, 1e-7, 1 - 1e-7)
    return np.log(p / (1 - p))

platt = LogisticRegression(C=1e6)
platt.fit(logit(clf.predict(X[cal], )).reshape(-1, 1), lab[cal])
p_va = platt.predict_proba(logit(clf.predict(X[va], )).reshape(-1, 1))[:, 1]
print(f"p_va mean={p_va.mean():.5f} (true {(y_va > THR).mean():.5f}) max={p_va.max():.3f} "
      f"| jan mean={p_va[vp == 'jan'].mean():.5f} jul mean={p_va[vp == 'jul'].mean():.5f}")

pA_l, pB = d["pA_l_va"], d["pB_va"]
normal = y.between(30, THR).to_numpy() & tr
esn_rows = normal & meta["es"].to_numpy()
y_es = y[esn_rows].to_numpy()
ws = np.linspace(0, 1, 21)
scores = [rmse(y_es, w * d["pA_c_es"] + (1 - w) * d["pA_l_es"]) for w in ws]
w = float(ws[int(np.argmin(scores))])
pA_blend = w * d["pA_c_va"] + (1 - w) * pA_l

for tag, pAx in [("LGBM", pA_l), (f"blend w_cb={w:.2f}", pA_blend)]:
    mix = ((1 - p_va) * pAx + p_va * pB).clip(60, 86400)
    rep = rmse_report(y_va, mix, vp)
    print(f"M6e {tag}: overall={rep['overall']:.1f} jan={rep['jan']:.1f} jul={rep['jul']:.1f}")

log_experiment("M6e fixed-rounds Platt mixture", model="mixture Platt-cal v2",
               features="M5 + AIRPORT_nofpl in clf; cal Nov15-Dec31",
               val_rmse=rep["overall"], val_rmse_jan=rep["jan"], val_rmse_jul=rep["jul"],
               notes=f"p_max={p_va.max():.3f}", keep="")
print("done", flush=True)
