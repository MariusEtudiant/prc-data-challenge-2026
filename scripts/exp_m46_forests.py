"""M46: bring bagging into our own pipeline - the family AutoGluon just showed us.

On LIRF, AutoGluon ranked ExtraTrees and RandomForest at level 2 ABOVE CatBoost and
LightGBM, and beat our whole trees+networks pipeline by 11s. We had never tried a
single bagging ensemble. Boosting chases residuals and is pulled by the heavy tail;
bagging averages decorrelated trees and absorbs it.

Here the same idea is implemented in the main environment (sklearn) so it can go into
a submission without depending on AutoGluon at scoring time.

stages: fit | blend
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
NPZ = ROOT / "experiments" / "m46_preds.npz"
CAP, MIN_TRAIN = 86400, 20000


def _load():
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
    return X, meta, y, tr, va, es, ok, art


def _numeric(X):
    """Forests need numbers: ordinal-code the categoricals, median-fill the rest."""
    Z = X.copy()
    for c in Z.columns:
        if Z[c].dtype.name == "category":
            Z[c] = Z[c].cat.codes.astype("float32")
    Z = Z.astype("float32")
    return Z.fillna(Z.median(numeric_only=True)).fillna(0.0)


def stage_fit():
    from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
    X, meta, y, tr, va, es, ok, art = _load()
    Z = _numeric(X)
    apt = X["AIRPORT"].astype(str).to_numpy()
    apt_va, apt_es = apt[va], apt[ok & tr & es]
    out = {k: np.full(va.sum(), np.nan) for k in ("et", "rf")}
    oute = {k: np.full((ok & tr & es).sum(), np.nan) for k in ("et", "rf")}
    t0 = time.time()
    for a in np.unique(apt):
        f = ok & tr & (apt == a) & ~es
        sv, se = apt_va == a, apt_es == a
        if f.sum() < MIN_TRAIN or not sv.any():
            continue
        yf = y[f].clip(upper=CAP).to_numpy()
        for key, Model in (("et", ExtraTreesRegressor), ("rf", RandomForestRegressor)):
            m = Model(n_estimators=300, min_samples_leaf=5, max_features=0.5,
                      n_jobs=-1, random_state=42)
            m.fit(Z[f], yf)
            out[key][sv] = m.predict(Z[va][sv])
            if se.any():
                oute[key][se] = m.predict(Z[ok & tr & es][se])
        print("  [%s] n=%d fait (%.0f min)" % (a, f.sum(), (time.time() - t0) / 60), flush=True)
    np.savez(NPZ, **{k + "_va": v for k, v in out.items()},
             **{k + "_es": v for k, v in oute.items()})
    print("[fit] termine en %.0f min" % ((time.time() - t0) / 60), flush=True)


def stage_blend():
    from dc26.validation import rmse, rmse_report
    from dc26 import log_experiment
    X, meta, y, tr, va, es, ok, art = _load()
    apt_va = X["AIRPORT"].astype(str).to_numpy()[va]
    y_va = y[va].to_numpy()
    clean = ~art[va]
    vp = meta.loc[va, "val_part"].to_numpy()
    g = np.clip(np.load(ROOT / "experiments" / "m36_gbdt_va.npy"), 60, CAP)
    n39 = np.load(ROOT / "experiments" / "m39_preds.npz")
    ng = np.clip(n39["nn_global"], 60, CAP)
    na = np.clip(np.where(np.isfinite(n39["nn_airport"]), n39["nn_airport"], n39["nn_global"]), 60, CAP)
    d = np.load(NPZ)
    et = np.clip(d["et_va"], 60, CAP)
    rf = np.clip(d["rf_va"], 60, CAP)

    src = {"GBDT": g, "NN global": ng, "NN aeroport": na, "ExtraTrees": et, "RandomForest": rf}
    for k, p in src.items():
        okp = np.isfinite(p) & clean
        print("%-14s nettoye=%.1f | corr/GBDT=%.3f" % (
            k, rmse(y_va[okp], p[okp]),
            np.corrcoef(g[okp] - y_va[okp], p[okp] - y_va[okp])[0, 1]), flush=True)

    from scipy.optimize import nnls
    names = list(src)
    V = np.vstack([np.where(np.isfinite(src[k]), src[k], g) for k in names])
    jan = vp == "jan"
    pred = np.zeros(len(y_va))
    weights = {}
    for a in np.unique(apt_va):
        m_fit = (apt_va == a) & clean & jan
        m_all = apt_va == a
        w, _ = nnls(V[:, m_fit].T, y_va[m_fit])
        w = w / w.sum() if w.sum() > 0 else np.array([1.0] + [0.0] * (len(names) - 1))
        weights[a] = [float(x) for x in w]
        pred[m_all] = w @ V[:, m_all]
        print("  [%s] %s | %.1f -> %.1f" % (
            a, " ".join("%s=%.2f" % (n, x) for n, x in zip(names, w) if x > 0.02),
            rmse(y_va[m_fit], g[m_fit]), rmse(y_va[m_fit], pred[m_fit])), flush=True)
    (ROOT / "experiments" / "m46_weights.json").write_text(json.dumps(weights))
    pred = np.clip(pred, 60, CAP)
    jul = vp == "jul"
    rep = rmse_report(y_va, pred, vp)
    print("\nM46: nettoye=%.1f | JUILLET (hors ajustement)=%.1f vs GBDT %.1f" % (
        rmse(y_va[clean], pred[clean]), rmse(y_va[clean & jul], pred[clean & jul]),
        rmse(y_va[clean & jul], g[clean & jul])), flush=True)
    log_experiment("M46 forets (ExtraTrees/RF) dans le melange", model="GBDT+NN+ET+RF",
                   features="full", val_rmse=rep["overall"], val_rmse_jan=rep["jan"],
                   val_rmse_jul=rep["jul"],
                   notes="nettoye %.1f" % rmse(y_va[clean], pred[clean]), keep="KEEP")


if __name__ == "__main__":
    {"fit": stage_fit, "blend": stage_blend}[sys.argv[1]]()
