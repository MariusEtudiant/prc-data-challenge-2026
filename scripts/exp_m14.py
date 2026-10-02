"""M14: v2 features + turnaround / stand-chaining features (free, from our own data).
Single m3bw LGBM on the mirror split, compared against the v2 baseline."""
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
from dc26.turnaround import build_turnaround_features

PROC = ROOT / "data" / "processed"
X = pd.read_parquet(PROC / "X_v2_mirror.parquet")
meta = pd.read_parquet(PROC / "meta_train2025.parquet")
cat = [c for c in X.columns if X[c].dtype.name == "category"]
y = meta["y"].astype("float64")
tr, va, es = meta["tr"].to_numpy(), meta["va"].to_numpy(), meta["es"].to_numpy()
vp = meta.loc[va, "val_part"].to_numpy()

mv = load_movements_2025()
dep = mv[mv["PHASE_mvt"] == "DEP"].reset_index(drop=True)
Xt = build_turnaround_features(dep, mv)
Xt.to_parquet(PROC / "turnaround_train2025.parquet", index=False)
print(f"turnaround: coverage {Xt['inbound_gap_ok'].mean():.3f}", flush=True)
X2 = pd.concat([X, Xt], axis=1)

CAP = 86400
keep = tr & (y > 0).to_numpy() & (y < 200000).to_numpy()
yk, esk = y[keep].clip(upper=CAP), es[keep]


def run(tag, XX):
    m = lgb.train(REG_PARAMS, lgb.Dataset(XX[keep][~esk], yk[~esk], categorical_feature=cat),
                  num_boost_round=6000, valid_sets=[lgb.Dataset(XX[keep][esk], yk[esk])],
                  callbacks=[lgb.early_stopping(150, verbose=False)])
    rep = rmse_report(y[va].to_numpy(), m.predict(XX[va], num_iteration=m.best_iteration).clip(60, CAP), vp)
    print(f"{tag}: overall={rep['overall']:.1f} jan={rep['jan']:.1f} jul={rep['jul']:.1f} "
          f"(iters={m.best_iteration})", flush=True)
    return rep, m


rep_base, _ = run("M14a v2 baseline", X)
rep14, m14 = run("M14b v2 + turnaround", X2)
imp = pd.Series(m14.feature_importance("gain"), index=m14.feature_name()).sort_values(ascending=False)
print("turnaround feature ranks:", {c: int((imp.index == c).argmax()) + 1 for c in Xt.columns}, flush=True)
log_experiment("M14 + turnaround/stand-chaining", model="m3bw LGBM",
               features="v2 + turnaround_avail, inbound_taxiin, inbound_late, taxi_upper_bound",
               val_rmse=rep14["overall"], val_rmse_jan=rep14["jan"], val_rmse_jul=rep14["jul"],
               notes=f"vs v2 baseline {rep_base['overall']:.1f}",
               keep="KEEP" if rep14["overall"] < rep_base["overall"] else "DROP")
print("done", flush=True)
