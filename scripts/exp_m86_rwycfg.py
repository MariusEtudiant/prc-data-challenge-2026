"""M86: runway-configuration features, retested in the current architecture.

Marius' point: an airport's layout and congestion pattern are too far from the
average for one model to generalise. We already answer that with per-airport
models, but the configuration itself - which runways the airport is actually
working at this moment - is not in the matrix. build_runway_config_features()
exists in the library and was tried on 2026-09-01 (M7a 357.65 against a 359.67
baseline) inside the old global mixture architecture, then never rewired when the
pipeline moved to per-airport tuned models, v2 features and the cleaned metric at
~256. A verdict at the 357 baseline says nothing at the 256 one.

  rwy_dep_share_30m      share of the airport's departures using OUR runway
  arrs_own_rwy_30m       arrivals landing on our runway (mixed-mode interference)
  n_dep_rwys_30m         how many runways are active for departures
  dep_rank_on_rwy_today  our position in the day's sequence on that runway

stages: feat | test
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
PROC = ROOT / "data" / "processed"
TRAIN_F = PROC / "rwycfg_train2025.parquet"
RANK_F = PROC / "rwycfg_ranking2026.parquet"
CAP, MIN_TRAIN = 86400, 20000
COLS = ["rwy_dep_share_30m", "arrs_own_rwy_30m", "n_dep_rwys_30m", "dep_rank_on_rwy_today"]


def stage_feat():
    import gc
    from dc26.state_features import build_runway_config_features
    from dc26.data import load_movements_2025, load_ranking_2026
    for tag, loader, out in (("2025", load_movements_2025, TRAIN_F),
                             ("2026", load_ranking_2026, RANK_F)):
        mv = loader()
        dep = mv[mv["PHASE_mvt"] == "DEP"]
        t0 = time.time()
        f = build_runway_config_features(dep, mv).reset_index(drop=True)
        f.to_parquet(out, index=False)
        print("%s: %d lignes, %.0f s, manquant %s" % (
            tag, len(f), time.time() - t0,
            " ".join("%s=%.1f%%" % (c, 100 * f[c].isna().mean()) for c in COLS)), flush=True)
        del mv, dep, f
        gc.collect()


def stage_test():
    import lightgbm as lgb
    from dc26.validation import rmse
    from dc26 import log_experiment
    X = pd.concat([pd.read_parquet(PROC / f) for f in
                   ["X_v2_mirror_t.parquet", "flightctx_train2025.parquet",
                    "depregime_train2025.parquet", "aobtq_train2025.parquet",
                    "anomreg_train2025.parquet"]], axis=1)
    E = pd.read_parquet(TRAIN_F)
    meta = pd.read_parquet(PROC / "meta_train2025.parquet")
    y = meta["y"].astype("float64")
    tr, va, es = meta["tr"].to_numpy(), meta["va"].to_numpy(), meta["es"].to_numpy()
    ok = (y > 0).to_numpy() & (y < 200000).to_numpy()
    art = np.zeros(len(y), bool)
    for k in (1, 2, 3):
        art |= (y - k * 86400).abs().to_numpy() < 3600
    ok &= ~art
    cat = [c for c in X.columns if X[c].dtype.name == "category"]
    apt = X["AIRPORT"].astype(str).to_numpy()
    apt_va, y_va, clean = apt[va], y[va].to_numpy(), ~art[va]
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
    res = {}
    for tag, XX in (("base", X), ("config pistes", pd.concat([X, E], axis=1))):
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
        res[tag] = (rmse(y_va[clean & g], pred[clean & g]),
                    rmse(y_va[clean & g & (vp == "jul")], pred[clean & g & (vp == "jul")]), per)
        np.save(ROOT / "experiments" / ("m86_%s.npy" % tag.split()[0]), pred)
        print("%-14s nettoye=%7.2f  juillet=%7.2f   (%.0f min)" % (
            tag, res[tag][0], res[tag][1], (time.time() - t0) / 60), flush=True)

    b, h = res["base"], res["config pistes"]
    print("\n  %-8s %10s %12s %9s" % ("aero", "base", "config", "gain"))
    for a in sorted(b[2]):
        print("  %-8s %10.1f %12.1f %+9.1f" % (a, b[2][a], h[2][a], h[2][a] - b[2][a]))
    print("\nGLOBAL nettoye %+.2f | juillet %+.2f" % (h[0] - b[0], h[1] - b[1]))
    log_experiment("M86 config pistes (rearchitecture)", model="LGBM par aeroport",
                   features="full + 4 config pistes", val_rmse=h[0],
                   notes="base %.2f | juillet %.2f vs %.2f" % (b[0], h[1], b[1]), keep="")


if __name__ == "__main__":
    {"feat": stage_feat, "test": stage_test}[sys.argv[1]]()
