"""M16: residual modelling via init_score.

Trees approximate the identity function poorly: forcing LightGBM to rebuild
y ~ d_aobt3 quantises a quantity we already know to ~5s median accuracy.
Instead we hand it an offset (init_score) and let it learn only the correction.

offset = d_aobt3 when it is present and plausible, else the PRU unimpeded
reference (ref_txot_p10). Prediction = offset + model output.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from dc26 import log_experiment
from dc26.validation import rmse_report
from dc26.mixture import REG_PARAMS

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


def offset(df):
    d = df["d_aobt3"].to_numpy(dtype="float64")
    ref = df["ref_txot_p10"].to_numpy(dtype="float64")
    o = np.where(np.isfinite(d) & (d > 60) & (d < 7200), d, np.nan)
    o = np.where(np.isnan(o), ref, o)
    return np.nan_to_num(o, nan=900.0)


off_all = offset(X)
print(f"offset: median={np.median(off_all):.0f}s | from AOBT_3 for "
      f"{np.mean(np.isfinite(X['d_aobt3']) & (X['d_aobt3'] > 60) & (X['d_aobt3'] < 7200)):.1%} of rows",
      flush=True)
print(f"offset alone RMSE on holdout: {rmse_report(y_va, off_all[va].clip(60, CAP), vp)['overall']:.1f}", flush=True)


def run(tag, use_offset):
    dtr = lgb.Dataset(X[keep][~esk], yk[~esk], categorical_feature=cat)
    des = lgb.Dataset(X[keep][esk], yk[esk], reference=dtr)
    if use_offset:
        o = off_all[keep]
        dtr.set_init_score(o[~esk])
        des.set_init_score(o[esk])
    m = lgb.train(REG_PARAMS, dtr, num_boost_round=6000, valid_sets=[des],
                  callbacks=[lgb.early_stopping(150, verbose=False)])
    p = m.predict(X[va], num_iteration=m.best_iteration)
    if use_offset:
        p = p + off_all[va]
    rep = rmse_report(y_va, np.clip(p, 60, CAP), vp)
    print(f"{tag}: overall={rep['overall']:.1f} jan={rep['jan']:.1f} jul={rep['jul']:.1f} "
          f"(iters={m.best_iteration})", flush=True)
    return rep


rep_base = run("M16a plain (reference)", False)
rep_off = run("M16b init_score = AOBT_3 offset", True)
log_experiment("M16 residual modelling (init_score)", model="LGBM + offset",
               features="v2+turnaround; init_score=d_aobt3 or ref_txot",
               val_rmse=rep_off["overall"], val_rmse_jan=rep_off["jan"], val_rmse_jul=rep_off["jul"],
               notes=f"vs plain {rep_base['overall']:.1f}",
               keep="KEEP" if rep_off["overall"] < rep_base["overall"] else "DROP")
print("done", flush=True)
