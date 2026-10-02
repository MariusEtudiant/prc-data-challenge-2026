"""M35: coarse entity groupings (Wu et al.) - stand pier/zone + taxi-time clusters."""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd, lightgbm as lgb

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from dc26 import load_movements_2025, load_ranking_2026, log_experiment
from dc26.validation import rmse, rmse_report
from dc26.mixture import REG_PARAMS
from dc26.entity_groups import fit_entity_groups, build_entity_features

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

mv = load_movements_2025()
dep = mv[mv["PHASE_mvt"] == "DEP"].reset_index(drop=True)
groups = fit_entity_groups(dep, y, mask=tr)          # fitted on mirror-train only
E = build_entity_features(dep, groups)
print("couverture des clusters:", {c: round(float(E[c].notna().mean()), 3)
                                   for c in E.columns if c.endswith("_cluster")}, flush=True)
rk = load_ranking_2026(); rkd = rk[rk["PHASE_mvt"] == "DEP"]
Erk = build_entity_features(rkd, groups)
print("couverture sur le RANKING:", {c: round(float(Erk[c].notna().mean()), 3)
                                     for c in Erk.columns if c.endswith("_cluster")}, flush=True)
E.to_parquet(PROC / "entity_train2025.parquet", index=False)

XE = pd.concat([X, E], axis=1)
cat_of = lambda XX: [c for c in XX.columns if XX[c].dtype.name == "category"]

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

def run(XX, tag):
    cat = cat_of(XX)
    pred = np.full(va.sum(), np.nan)
    for a in np.unique(apt):
        f, s = ok & tr & (apt == a) & ~es, ok & tr & (apt == a) & es
        sel = apt_va == a
        if f.sum() < MIN_TRAIN or not sel.any():
            continue
        m = lgb.train(tuned(a), lgb.Dataset(XX[f], y[f].clip(upper=CAP), categorical_feature=cat),
                      num_boost_round=6000, valid_sets=[lgb.Dataset(XX[s], y[s].clip(upper=CAP))],
                      callbacks=[lgb.early_stopping(150, verbose=False)])
        pred[sel] = m.predict(XX[va][sel], num_iteration=m.best_iteration)
    pred = np.clip(pred, 60, CAP)
    rep = rmse_report(y_va, pred, vp)
    print("%-22s NETTOYE=%.1f | brut=%.1f (jan %.1f jul %.1f)" % (
        tag, rmse(y_va[clean], pred[clean]), rep["overall"], rep["jan"], rep["jul"]), flush=True)
    return rep, rmse(y_va[clean], pred[clean])

repA, clA = run(X, "sans groupements")
repB, clB = run(XE, "AVEC groupements")
print("\ndelta nettoye: %+.2f s" % (clB - clA), flush=True)
log_experiment("M35 groupements d'entites (Wu et al.)", model="per-airport tuned LGBM",
               features="full + stand_pier/zone + clusters stand/op/actype",
               val_rmse=repB["overall"], val_rmse_jan=repB["jan"], val_rmse_jul=repB["jul"],
               notes="nettoye %.1f vs %.1f" % (clB, clA), keep="KEEP" if clB < clA else "DROP")
print("done", flush=True)
