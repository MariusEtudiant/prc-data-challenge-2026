"""M77: the reference-time cascade skips the stand level.

Today: stand x runway (if >=100 flights) -> RUNWAY -> airport. When a pair is thin
the fallback discards the stand entirely, although the stand is what sets the taxi
distance and each stand has hundreds of flights. LTFM has 79.1% thin pairs, LFPG
74.9% - our two most fragmented airports, and two of the three weakest R2.

Adds the missing levels as explicit features and lets the model choose:
  ref_stand      P10 for the stand, all runways
  ref_rwy_only   P10 for the runway, all stands
  ref_combo_raw  P10 for the pair, NaN when thin (so the model sees the gap)
  n_combo/n_stand  cell sizes = how much to trust each level
  ref_delta      how far this pair sits from its stand's own reference

References fitted on TRAINING rows only; holdout untouched.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
PROC = ROOT / "data" / "processed"
from dc26.validation import rmse
from dc26.data import load_movements_2025

CAP, MIN_TRAIN, REF_Q = 86400, 20000, 0.10

X = pd.concat([pd.read_parquet(PROC / f) for f in
               ["X_v2_mirror_t.parquet", "flightctx_train2025.parquet",
                "depregime_train2025.parquet", "aobtq_train2025.parquet",
                "anomreg_train2025.parquet"]], axis=1)
meta = pd.read_parquet(PROC / "meta_train2025.parquet")
y = meta["y"].astype("float64")
tr, va, es = meta["tr"].to_numpy(), meta["va"].to_numpy(), meta["es"].to_numpy()
ok = (y > 0).to_numpy() & (y < 200000).to_numpy()
art = np.zeros(len(y), bool)
for k in (1, 2, 3):
    art |= (y - k * 86400).abs().to_numpy() < 3600
ok &= ~art

mv = load_movements_2025()
dep = mv[mv["PHASE_mvt"] == "DEP"].reset_index(drop=True)
assert len(dep) == len(meta)
apt_s = dep["AIRPORT"].astype(str)
k_stand = (apt_s + "|" + dep["STAND_mvt"].astype(str)).to_numpy()
k_rwy = (apt_s + "|" + dep["RUNWAY_mvt"].astype(str)).to_numpy()
k_combo = (apt_s + "|" + dep["STAND_mvt"].astype(str) + "|" + dep["RUNWAY_mvt"].astype(str)).to_numpy()

fit = ok & tr & y.between(60, 7200).to_numpy()
yv_fit = y[fit].to_numpy()


def level(keys, min_n):
    s = pd.Series(yv_fit).groupby(keys[fit])
    q, n = s.quantile(REF_Q), s.size()
    return q[n >= min_n], n


ref_st, n_st = level(k_stand, 50)
ref_rw, n_rw = level(k_rwy, 50)
ref_cb, n_cb = level(k_combo, 100)
E = pd.DataFrame(index=X.index)
E["ref_stand"] = pd.Series(k_stand).map(ref_st).to_numpy()
E["ref_rwy_only"] = pd.Series(k_rwy).map(ref_rw).to_numpy()
E["ref_combo_raw"] = pd.Series(k_combo).map(ref_cb).to_numpy()
E["n_combo"] = pd.Series(k_combo).map(n_cb).fillna(0).to_numpy()
E["n_stand"] = pd.Series(k_stand).map(n_st).fillna(0).to_numpy()
E["ref_delta"] = E["ref_combo_raw"] - E["ref_stand"]
print("couverture ajoutee: ref_stand %.1f%%, ref_combo_raw %.1f%%" % (
    100 * E["ref_stand"].notna().mean(), 100 * E["ref_combo_raw"].notna().mean()), flush=True)

cat = [c for c in X.columns if X[c].dtype.name == "category"]
apt = X["AIRPORT"].astype(str).to_numpy()
apt_va = apt[va]
y_va = y[va].to_numpy()
clean = ~art[va]
vp = meta.loc[va, "val_part"].to_numpy()


def tuned(a):
    from dc26.mixture import REG_PARAMS
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


t0 = time.time()
out = {}
for tag, XX in (("base", X), ("hierarchique", pd.concat([X, E], axis=1))):
    pred = np.full(va.sum(), np.nan)
    per = {}
    for a in np.unique(apt):
        f, s = ok & tr & (apt == a) & ~es, ok & tr & (apt == a) & es
        sv = apt_va == a
        if f.sum() < MIN_TRAIN or not sv.any():
            continue
        m = lgb.train(tuned(a), lgb.Dataset(XX[f], y[f].clip(upper=CAP), categorical_feature=cat),
                      num_boost_round=6000,
                      valid_sets=[lgb.Dataset(XX[s], y[s].clip(upper=CAP))],
                      callbacks=[lgb.early_stopping(150, verbose=False)])
        pred[sv] = m.predict(XX[va][sv], num_iteration=m.best_iteration)
        per[a] = rmse(y_va[sv & clean], np.clip(pred[sv & clean], 60, CAP))
    pred = np.clip(pred, 60, CAP)
    g = np.isfinite(pred)
    out[tag] = (rmse(y_va[clean & g], pred[clean & g]),
                rmse(y_va[clean & g & (vp == "jul")], pred[clean & g & (vp == "jul")]), per)
    np.save(ROOT / "experiments" / ("m77_%s.npy" % tag), pred)
    print("%-14s nettoye=%7.2f  juillet=%7.2f   (%.0f min)" % (
        tag, out[tag][0], out[tag][1], (time.time() - t0) / 60), flush=True)

b, h = out["base"], out["hierarchique"]
print("\n=== ecart par aeroport (nettoye) ===")
print("  %-8s %10s %10s %9s" % ("aero", "base", "hierarch.", "gain"))
for a in sorted(b[2]):
    print("  %-8s %10.1f %10.1f %+9.1f" % (a, b[2][a], h[2][a], h[2][a] - b[2][a]))
print("\nGLOBAL nettoye %+.2f | juillet %+.2f" % (h[0] - b[0], h[1] - b[1]))
from dc26 import log_experiment
log_experiment("M77 reference hierarchique (niveau poste)", model="LGBM par aeroport",
               features="full + 6 references", val_rmse=h[0],
               notes="base %.2f, juillet %.2f vs %.2f" % (b[0], h[1], b[1]), keep="")
