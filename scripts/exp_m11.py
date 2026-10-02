"""M11: official PRU-style unimpeded reference times as features.
- ref_txot = P10 of taxi-out per STAND x DRWY combo (min 100 flights; fallback runway,
  then airport level), computed on mirror-train rows only (PRU: rolling year, filters:
  taxi <= 120min).
- implied_add = d_aobt3 - ref_txot  (negative => AOBT_3 physically inconsistent
  => principled distrust signal).
Evaluated as m3bw single model (stable config) vs M9-A reference 359.7."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from dc26 import load_movements_2025, mirror_split, log_experiment
from dc26.validation import rmse_report
from dc26.mixture import REG_PARAMS

PROC = ROOT / "data" / "processed"
X = pd.read_parquet(PROC / "X_train2025.parquet")
meta = pd.read_parquet(PROC / "meta_train2025.parquet")
cat_cols = [c for c in X.columns if X[c].dtype.name == "category"]

# need STAND/RUNWAY/taxi from the source table (X's STAND is airport_stand category)
mv = load_movements_2025()
dep = mv[mv["PHASE_mvt"] == "DEP"].reset_index(drop=True)
assert len(dep) == len(X)
y = meta["y"].astype("float64")
tr, va, es = meta["tr"].to_numpy(), meta["va"].to_numpy(), meta["es"].to_numpy()
vp = meta.loc[va, "val_part"].to_numpy()

combo = dep["AIRPORT"].astype(str) + "_" + dep["STAND_mvt"].astype(str) + "_" + dep["RUNWAY_mvt"].astype(str)
rwy_key = dep["AIRPORT"].astype(str) + "_" + dep["RUNWAY_mvt"].astype(str)

# PRU-style sample: train rows, taxi <= 120 min, valid
samp = tr & y.between(60, 7200).to_numpy()
def p10_ref(keys):
    g = y[samp].groupby(keys[samp])
    ref = g.quantile(0.10)
    n = g.size()
    return ref[n >= 100]

ref_combo = p10_ref(combo)
ref_rwy = p10_ref(rwy_key)
ref_apt = p10_ref(dep["AIRPORT"].astype(str))

ref = combo.map(ref_combo)
ref = ref.fillna(rwy_key.map(ref_rwy))
ref = ref.fillna(dep["AIRPORT"].astype(str).map(ref_apt))
print(f"ref coverage: combo {combo.map(ref_combo).notna().mean():.3f}, "
      f"after fallbacks {ref.notna().mean():.3f} | median ref={ref.median():.0f}s")

X2 = X.copy()
X2["ref_txot_p10"] = ref.to_numpy(dtype="float32")
X2["implied_add"] = (X["d_aobt3"].to_numpy() - ref.to_numpy()).astype("float32")

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
    return rep, m


rep_ref, _ = run("M11a m3bw (re-run reference)", X)
rep11, m11 = run("M11b m3bw + PRU ref P10 + implied_add", X2)
imp = pd.Series(m11.feature_importance("gain"), index=m11.feature_name()).sort_values(ascending=False)
print("top new-feature ranks:", {c: int((imp.index == c).argmax()) + 1 for c in ["ref_txot_p10", "implied_add"]})
log_experiment("M11 PRU unimpeded ref P10", model="m3bw LGBM",
               features="M5 + ref_txot_p10 (stand x rwy) + implied_add",
               val_rmse=rep11["overall"], val_rmse_jan=rep11["jan"], val_rmse_jul=rep11["jul"],
               notes=f"vs rerun ref {rep_ref['overall']:.1f}", keep="")
print("done", flush=True)
