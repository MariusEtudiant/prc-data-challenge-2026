"""M29: true OOF stacking (Wu et al. / jubilant-vase recipe).

M27 produced LGBM out-of-fold predictions over the whole training year. Here we add
CatBoost OOF, then fit a meta-model on ~1.9M OOF rows instead of optimising weights
on a 110k early-stopping window.

stages: cb_oof | stack
"""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
PROC = ROOT / "data" / "processed"
CAP, MIN_TRAIN = 86400, 20000
FOLDS = [(1, 2, 3), (4, 5), (6, 7), (8, 9), (10, 11, 12)]


def _load():
    X = pd.concat([pd.read_parquet(PROC / "X_v2_mirror_t.parquet"),
                   pd.read_parquet(PROC / "flightctx_train2025.parquet"),
                   pd.read_parquet(PROC / "depregime_train2025.parquet"),
                   pd.read_parquet(PROC / "aobtq_train2025.parquet")], axis=1)
    meta = pd.read_parquet(PROC / "meta_train2025.parquet")
    y = meta["y"].astype("float64")
    tr, va = meta["tr"].to_numpy(), meta["va"].to_numpy()
    keep = tr & (y > 0).to_numpy() & (y < 200000).to_numpy()
    return X, meta, y, tr, va, keep


def stage_cb_oof():
    from catboost import CatBoostRegressor, Pool
    X, meta, y, tr, va, keep = _load()
    cat = [c for c in X.columns if X[c].dtype.name == "category"]
    for c in cat:
        X[c] = X[c].cat.add_categories("__NA__").fillna("__NA__")
    apt = X["AIRPORT"].astype(str).to_numpy()
    month = X["month"].to_numpy()
    oof = np.full(len(X), np.nan)
    pv = np.full(va.sum(), np.nan)
    for a in np.unique(apt):
        sel_tr, sel_va = keep & (apt == a), apt[va] == a
        if sel_tr.sum() < MIN_TRAIN or not sel_va.any():
            continue
        fp = []
        for months in FOLDS:
            hold = sel_tr & np.isin(month, months)
            fit = sel_tr & ~np.isin(month, months)
            if hold.sum() < 500 or fit.sum() < 5000:
                continue
            m = CatBoostRegressor(loss_function="RMSE", iterations=1200, learning_rate=0.1,
                                  depth=7, l2_leaf_reg=3.0, random_seed=42, od_type="Iter",
                                  od_wait=80, verbose=False, thread_count=-1,
                                  max_ctr_complexity=1, cat_features=cat)
            m.fit(Pool(X[fit], y[fit].clip(upper=CAP), cat_features=cat),
                  eval_set=Pool(X[hold], y[hold].clip(upper=CAP), cat_features=cat))
            oof[hold] = m.predict(X[hold])
            fp.append(m.predict(X[va][sel_va]))
        pv[sel_va] = np.mean(fp, axis=0)
        print("[cb_oof] %s done (%d folds)" % (a, len(fp)), flush=True)
    np.save(ROOT / "experiments" / "m29_cb_oof.npy", oof)
    np.save(ROOT / "experiments" / "m29_cb_va.npy", pv)


def stage_stack():
    from sklearn.linear_model import RidgeCV
    from dc26.validation import rmse, rmse_report
    from dc26 import log_experiment
    X, meta, y, tr, va, keep = _load()
    apt = X["AIRPORT"].astype(str).to_numpy()
    vp = meta.loc[va, "val_part"].to_numpy()
    y_va = y[va].to_numpy()

    l_oof = np.load(ROOT / "experiments" / "m27_oof.npy")
    l_va = np.load(ROOT / "experiments" / "m27_pred_va.npy")
    c_oof = np.load(ROOT / "experiments" / "m29_cb_oof.npy")
    c_va = np.load(ROOT / "experiments" / "m29_cb_va.npy")
    print("OOF rows usable: %d" % int((np.isfinite(l_oof) & np.isfinite(c_oof) & keep).sum()), flush=True)

    pred = np.full(va.sum(), np.nan)
    for a in np.unique(apt[va]):
        m_oof = keep & (apt == a) & np.isfinite(l_oof) & np.isfinite(c_oof)
        m_va = apt[va] == a
        if m_oof.sum() < 5000:
            pred[m_va] = 0.5 * l_va[m_va] + 0.5 * c_va[m_va]
            continue
        A = np.column_stack([l_oof[m_oof], c_oof[m_oof]])
        r = RidgeCV(alphas=[0.1, 1.0, 10.0, 100.0]).fit(A, y[m_oof].clip(upper=CAP))
        pred[m_va] = r.predict(np.column_stack([l_va[m_va], c_va[m_va]]))
        print("  [%s] coef=%s intercept=%.0f | mirror=%.1f" % (
            a, np.round(r.coef_, 3), r.intercept_,
            rmse(y_va[m_va], np.clip(pred[m_va], 60, CAP))), flush=True)
    rep = rmse_report(y_va, np.clip(pred, 60, CAP), vp)
    print("M29 OOF stacking: overall=%.1f jan=%.1f jul=%.1f | LB ~%.0f" % (
        rep["overall"], rep["jan"], rep["jul"], rep["overall"] * 0.837), flush=True)
    log_experiment("M29 OOF stacking (Ridge meta)", model="LGBM+CB OOF -> Ridge",
                   features="full", val_rmse=rep["overall"], val_rmse_jan=rep["jan"],
                   val_rmse_jul=rep["jul"], notes="meta fitted on 1.9M OOF rows", keep="")


if __name__ == "__main__":
    {"cb_oof": stage_cb_oof, "stack": stage_stack}[sys.argv[1]]()
