"""M28: explicit AOBT_3-quality features (reconstructed vs measured off-block)."""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd, lightgbm as lgb

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from dc26 import load_movements_2025, log_experiment
from dc26.validation import rmse, rmse_report
from dc26.mixture import REG_PARAMS
from dc26.aobt_quality import fit_aobt_spikes, build_aobt_quality

PROC = ROOT / "data" / "processed"
CAP, MIN_TRAIN = 86400, 20000
X = pd.concat([pd.read_parquet(PROC / "X_v2_mirror_t.parquet"),
               pd.read_parquet(PROC / "flightctx_train2025.parquet"),
               pd.read_parquet(PROC / "depregime_train2025.parquet")], axis=1)
meta = pd.read_parquet(PROC / "meta_train2025.parquet")
y = meta["y"].astype("float64")
tr, va, es = meta["tr"].to_numpy(), meta["va"].to_numpy(), meta["es"].to_numpy()
vp = meta.loc[va, "val_part"].to_numpy()
keep = tr & (y > 0).to_numpy() & (y < 200000).to_numpy()
y_va = y[va].to_numpy()

mv = load_movements_2025()
dep = mv[mv["PHASE_mvt"] == "DEP"].reset_index(drop=True)
model = fit_aobt_spikes(dep[tr])          # spikes learned on mirror-train only
(ROOT / "experiments" / "aobt_spikes.json").write_text(json.dumps(model))
Xq = build_aobt_quality(dep, model)
Xq.to_parquet(PROC / "aobtq_train2025.parquet", index=False)
print("aobt-quality corr:", {c: round(float(Xq[c].corr(y)), 3) for c in Xq.columns}, flush=True)

apt = X["AIRPORT"].astype(str).to_numpy(); apt_va = apt[va]

def tuned(a):
    p = ROOT / "experiments" / ("optuna_" + a + ".json")
    base = dict(objective="regression", metric="rmse", verbosity=-1, num_threads=0,
                bagging_freq=1, seed=42)
    if not p.exists():
        return dict(REG_PARAMS)
    d = json.loads(p.read_text())
    base.update({k: v for k, v in d.items() if not k.startswith("_")})
    for k in ("num_leaves", "min_data_in_leaf", "max_bin"):
        base[k] = int(base[k])
    return base

def run(XX, tag):
    cat = [c for c in XX.columns if XX[c].dtype.name == "category"]
    pred = np.full(va.sum(), np.nan)
    for a in np.unique(apt):
        fit, ev = keep & (apt == a) & ~es, keep & (apt == a) & es
        sel = apt_va == a
        if fit.sum() < MIN_TRAIN or not sel.any():
            continue
        m = lgb.train(tuned(a), lgb.Dataset(XX[fit], y[fit].clip(upper=CAP), categorical_feature=cat),
                      num_boost_round=6000, valid_sets=[lgb.Dataset(XX[ev], y[ev].clip(upper=CAP))],
                      callbacks=[lgb.early_stopping(150, verbose=False)])
        pred[sel] = m.predict(XX[va][sel], num_iteration=m.best_iteration)
    pred = np.clip(pred, 60, CAP)
    rep = rmse_report(y_va, pred, vp)
    print("%s: overall=%.1f jan=%.1f jul=%.1f" % (tag, rep["overall"], rep["jan"], rep["jul"]), flush=True)
    return rep, pred

repA, pA = run(X, "M28a baseline (tuned)")
repB, pB = run(pd.concat([X, Xq], axis=1), "M28b + AOBT quality")
for a in ["LTFM", "LEBL", "EDDM", "LIRF", "LFPG"]:
    s = apt_va == a
    if s.any():
        print("   [%s] %.1f -> %.1f" % (a, rmse(y_va[s], pA[s]), rmse(y_va[s], pB[s])), flush=True)
log_experiment("M28 AOBT_3 quality (reconstructed off-block)", model="per-airport tuned LGBM",
               features="full + aobt_spike_flag/dev/bucket_share",
               val_rmse=repB["overall"], val_rmse_jan=repB["jan"], val_rmse_jul=repB["jul"],
               notes="vs %.1f" % repA["overall"],
               keep="KEEP" if repB["overall"] < repA["overall"] else "DROP")
print("done", flush=True)
