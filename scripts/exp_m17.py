"""M17: physical queue reconstruction + arrival-based surface probe.

  A  baseline (v2+turnaround)
  B  + queue model  (t_join = AOBT_3 + ref_txot, exact backlog, service rate, overtaking)
  C  + arrival probe (taxi-in anomalies measured in the SAME period — for the test
     set these are real Jan/Jul 2026 measurements)
  D  + both
"""
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
from dc26.queue_model import build_queue_features
from dc26.arrival_probe import fit_arrival_reference, build_arrival_probe

PROC = ROOT / "data" / "processed"
CAP = 86400
X = pd.read_parquet(PROC / "X_v2_mirror_t.parquet")
meta = pd.read_parquet(PROC / "meta_train2025.parquet")
cat = [c for c in X.columns if X[c].dtype.name == "category"]
y = meta["y"].astype("float64")
tr, va, es = meta["tr"].to_numpy(), meta["va"].to_numpy(), meta["es"].to_numpy()
vp = meta.loc[va, "val_part"].to_numpy()
keep = tr & (y > 0).to_numpy() & (y < 200000).to_numpy()
yk, esk = y[keep].clip(upper=CAP), es[keep]
y_va = y[va].to_numpy()

mv = load_movements_2025()
dep = mv[mv["PHASE_mvt"] == "DEP"].reset_index(drop=True)
ref = pd.Series(X["ref_txot_p10"].to_numpy(dtype=float), index=dep.index)

Xq = build_queue_features(dep, mv, ref)
Xq.to_parquet(PROC / "queue_train2025.parquet", index=False)
print("queue features:", {c: round(float(Xq[c].corr(y)), 3) for c in Xq.columns}, flush=True)

arr_refs = fit_arrival_reference(mv)
Xa = build_arrival_probe(dep, mv, arr_refs)
Xa.to_parquet(PROC / "arrprobe_train2025.parquet", index=False)
print("arrival probe:", {c: round(float(Xa[c].corr(y)), 3) for c in Xa.columns}, flush=True)


def run(tag, XX):
    m = lgb.train(REG_PARAMS, lgb.Dataset(XX[keep][~esk], yk[~esk], categorical_feature=cat),
                  num_boost_round=6000, valid_sets=[lgb.Dataset(XX[keep][esk], yk[esk])],
                  callbacks=[lgb.early_stopping(150, verbose=False)])
    rep = rmse_report(y_va, m.predict(XX[va], num_iteration=m.best_iteration).clip(60, CAP), vp)
    print(f"{tag}: overall={rep['overall']:.1f} jan={rep['jan']:.1f} jul={rep['jul']:.1f} "
          f"(iters={m.best_iteration})", flush=True)
    return rep, m


repA, _ = run("M17a baseline", X)
repB, _ = run("M17b + queue model", pd.concat([X, Xq], axis=1))
repC, _ = run("M17c + arrival probe", pd.concat([X, Xa], axis=1))
repD, mD = run("M17d + both", pd.concat([X, Xq, Xa], axis=1))

imp = pd.Series(mD.feature_importance("gain"), index=mD.feature_name()).sort_values(ascending=False)
print("\nnew feature ranks:", {c: int((imp.index == c).argmax()) + 1
                               for c in list(Xq.columns) + list(Xa.columns)}, flush=True)
log_experiment("M17 queue model + arrival probe", model="m3bw LGBM",
               features="v2+turnaround +queue +arrprobe",
               val_rmse=repD["overall"], val_rmse_jan=repD["jan"], val_rmse_jul=repD["jul"],
               notes=f"base {repA['overall']:.1f} | queue {repB['overall']:.1f} | probe {repC['overall']:.1f}",
               keep="KEEP" if repD["overall"] < repA["overall"] else "DROP")
print("done", flush=True)
