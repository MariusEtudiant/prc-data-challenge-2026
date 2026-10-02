"""M43: monotonic constraints - a physical prior never imposed.

Some relations here are known a priori and cannot reverse: more time elapsed since
the NM off-block estimate means a longer taxi; a longer unimpeded reference means a
longer taxi; more aircraft ahead in the queue means a longer taxi. Letting the trees
discover non-monotone shapes in those directions only fits noise.

On a problem where we are fighting variance rather than bias, constraining the sign
is free regularisation.
"""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd, lightgbm as lgb

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from dc26 import log_experiment
from dc26.validation import rmse, rmse_report
from dc26.mixture import REG_PARAMS

PROC = ROOT / "data" / "processed"
CAP, MIN_TRAIN = 86400, 20000
X = pd.concat([pd.read_parquet(PROC / f) for f in
               ["X_v2_mirror_t.parquet", "flightctx_train2025.parquet",
                "depregime_train2025.parquet", "aobtq_train2025.parquet",
                "anomreg_train2025.parquet"]], axis=1)
meta = pd.read_parquet(PROC / "meta_train2025.parquet")
y = meta["y"].astype("float64")
tr, va, es = meta["tr"].to_numpy(), meta["va"].to_numpy(), meta["es"].to_numpy()
vp = meta.loc[va, "val_part"].to_numpy()
ok = (y > 0).to_numpy() & (y < 200000).to_numpy()
art = np.zeros(len(y), bool)
for k in (1, 2, 3):
    art |= (y - k * 86400).abs().to_numpy() < 3600
ok &= ~art
apt = X["AIRPORT"].astype(str).to_numpy(); apt_va = apt[va]
y_va = y[va].to_numpy(); clean = ~art[va]
cat = [c for c in X.columns if X[c].dtype.name == "category"]

INCREASING = ["d_aobt3", "ref_txot_p10", "q_dep_aobt_to_takeoff_runway",
              "q_dep_aobt_to_takeoff_airport", "q_backlog_at_takeoff",
              "deps_last_1800s", "deps_last_3600s", "push_last_1800s",
              "dep_proxy_mean_1800s", "dep_proxy_mean_3600s", "dep_proxy_rwy_1800s",
              "pred_queue_wait", "arr_taxi_mean_3600s"]
mono = [1 if c in INCREASING else 0 for c in X.columns]
print("contraintes croissantes sur %d variables presentes: %s" % (
    sum(mono), [c for c in X.columns if c in INCREASING]), flush=True)

def tuned(a):
    p = ROOT / "experiments" / ("optuna_" + a + ".json")
    b = dict(objective="regression", metric="rmse", verbosity=-1, num_threads=0,
             bagging_freq=1, seed=42)
    if not p.exists():
        return dict(REG_PARAMS)
    d = json.loads(p.read_text())
    b.update({k: v for k, v in d.items() if not k.startswith("_")})
    for k in ("num_leaves", "min_data_in_leaf", "max_bin"):
        b[k] = int(b[k])
    return b

def run(extra, tag):
    pred = np.full(va.sum(), np.nan)
    for a in np.unique(apt):
        f, s = ok & tr & (apt == a) & ~es, ok & tr & (apt == a) & es
        sel = apt_va == a
        if f.sum() < MIN_TRAIN or not sel.any():
            continue
        p = dict(tuned(a)); p.update(extra)
        m = lgb.train(p, lgb.Dataset(X[f], y[f].clip(upper=CAP), categorical_feature=cat),
                      num_boost_round=6000, valid_sets=[lgb.Dataset(X[s], y[s].clip(upper=CAP))],
                      callbacks=[lgb.early_stopping(150, verbose=False)])
        pred[sel] = m.predict(X[va][sel], num_iteration=m.best_iteration)
    pred = np.clip(pred, 60, CAP)
    rep = rmse_report(y_va, pred, vp)
    c = rmse(y_va[clean], pred[clean])
    print("%-28s NETTOYE=%.1f | brut=%.1f" % (tag, c, rep["overall"]), flush=True)
    return rep, c, pred

repA, cA, _ = run({}, "sans contrainte")
repB, cB, _ = run({"monotone_constraints": mono}, "contraintes monotones")
repC, cC, _ = run({"monotone_constraints": mono, "monotone_constraints_method": "advanced"},
                  "contraintes (methode avancee)")
best = min([("aucune", cA), ("basique", cB), ("avancee", cC)], key=lambda t: t[1])
print("\nmeilleur: %s (%.1f)" % best, flush=True)
log_experiment("M43 contraintes monotones", model="per-airport tuned LGBM", features="full",
               val_rmse=repB["overall"], val_rmse_jan=repB["jan"], val_rmse_jul=repB["jul"],
               notes="nettoye sans=%.1f basique=%.1f avancee=%.1f" % (cA, cB, cC),
               keep="KEEP" if best[1] < cA else "DROP")
