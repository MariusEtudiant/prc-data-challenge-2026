"""M12: + PRU ATFM daily delay features (by cause) and monthly ATXOT.
Base = M11b (M5 features + ref P10 + implied_add). Single m3bw LGBM eval."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from dc26 import load_movements_2025, log_experiment
from dc26.validation import rmse_report
from dc26.mixture import REG_PARAMS
from dc26.pru_features import build_pru_features

PROC = ROOT / "data" / "processed"
X = pd.read_parquet(PROC / "X_train2025.parquet")
meta = pd.read_parquet(PROC / "meta_train2025.parquet")
cat_cols = [c for c in X.columns if X[c].dtype.name == "category"]

mv = load_movements_2025()
dep = mv[mv["PHASE_mvt"] == "DEP"].reset_index(drop=True)
y = meta["y"].astype("float64")
tr, va, es = meta["tr"].to_numpy(), meta["va"].to_numpy(), meta["es"].to_numpy()
vp = meta.loc[va, "val_part"].to_numpy()

# M11 features (ref P10 + implied_add), train-only sample for the reference
combo = dep["AIRPORT"].astype(str) + "_" + dep["STAND_mvt"].astype(str) + "_" + dep["RUNWAY_mvt"].astype(str)
rwy_key = dep["AIRPORT"].astype(str) + "_" + dep["RUNWAY_mvt"].astype(str)
samp = tr & y.between(60, 7200).to_numpy()
def p10_ref(keys):
    g = y[samp].groupby(keys[samp])
    ref = g.quantile(0.10)
    return ref[g.size() >= 100]
ref = combo.map(p10_ref(combo)).fillna(rwy_key.map(p10_ref(rwy_key))).fillna(
    dep["AIRPORT"].astype(str).map(p10_ref(dep["AIRPORT"].astype(str))))
X["ref_txot_p10"] = ref.to_numpy(dtype="float32")
X["implied_add"] = (X["d_aobt3"].to_numpy() - ref.to_numpy()).astype("float32")

Xpru = build_pru_features(dep)
print("PRU features notna:", Xpru.notna().mean().round(3).to_dict(), flush=True)
X2 = pd.concat([X, Xpru], axis=1)

CAP = 86400
keep = tr & (y > 0).to_numpy() & (y < 200000).to_numpy()
yk = y[keep].clip(upper=CAP)
es_k = es[keep]

def run(tag, XX):
    m = lgb.train(REG_PARAMS, lgb.Dataset(XX[keep][~es_k], yk[~es_k], categorical_feature=cat_cols),
                  num_boost_round=6000,
                  valid_sets=[lgb.Dataset(XX[keep][es_k], yk[es_k])],
                  callbacks=[lgb.early_stopping(150, verbose=False)])
    pred = m.predict(XX[va], num_iteration=m.best_iteration).clip(60, CAP)
    rep = rmse_report(y[va].to_numpy(), pred, vp)
    print(f"{tag}: overall={rep['overall']:.1f} jan={rep['jan']:.1f} jul={rep['jul']:.1f} (iters={m.best_iteration})", flush=True)
    return rep

rep = run("M12 m3bw + refP10 + ATFM/ATXOT", X2)
log_experiment("M12 + PRU ATFM daily & ATXOT", model="m3bw LGBM",
               features="M11b + atfm daily by cause + atxot monthly",
               val_rmse=rep["overall"], val_rmse_jan=rep["jan"], val_rmse_jul=rep["jul"],
               notes="vs M11b 357.9", keep="")

# if good, persist the enriched matrix for the ensemble
X2.to_parquet(PROC / "X_train2025_v2.parquet", index=False)
print("X_train2025_v2 saved:", X2.shape, flush=True)
print("done", flush=True)
