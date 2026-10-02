"""Champion architecture: calibrated regime mixture (see journal §8).
Calibration = Platt scaling on the ES window (M6c recipe): smooth parametric fit,
no isotonic plateaus (they caused July false positives), robust with ~30-50 positives.
fit_mixture / predict_mixture are reused by experiments and the submission pipeline."""
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.linear_model import LogisticRegression

THR = 7200
CAP = 86400

REG_PARAMS = dict(objective="regression", metric="rmse", learning_rate=0.06,
                  num_leaves=255, min_data_in_leaf=100, feature_fraction=0.9,
                  bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0,
                  num_threads=0, verbosity=-1, seed=42)
CLF_PARAMS = dict(objective="binary", metric="binary_logloss", learning_rate=0.05,
                  num_leaves=63, min_data_in_leaf=50, feature_fraction=0.9,
                  num_threads=0, verbosity=-1, seed=42)
EXT_PARAMS = dict(objective="regression", learning_rate=0.05, num_leaves=15,
                  min_data_in_leaf=20, feature_fraction=0.8, num_threads=0,
                  verbosity=-1, seed=42)


def _logit(p):
    p = np.clip(p, 1e-7, 1 - 1e-7)
    return np.log(p / (1 - p))


class _PlattCal:
    def __init__(self, clf):
        self.clf = clf
        self.lr = LogisticRegression(C=1e6)

    def fit(self, X, lab):
        s = _logit(self.clf.predict(X, num_iteration=self.clf.best_iteration))
        self.lr.fit(s.reshape(-1, 1), lab)
        return self

    def predict(self, X):
        s = _logit(self.clf.predict(X, num_iteration=self.clf.best_iteration))
        return self.lr.predict_proba(s.reshape(-1, 1))[:, 1]


class _SeedEnsembleCal:
    """Average of Platt-calibrated classifiers over several seeds — the single-seed
    classifier's early stopping is a large variance source (±40s jan RMSE)."""

    def __init__(self, cals):
        self.cals = cals

    def predict(self, X):
        return np.mean([c.predict(X) for c in self.cals], axis=0)


def fit_mixture(X, y, es_mask, cat_cols, reg_params=None, n_seeds=3):
    """X, y: training rows. es_mask: boolean array marking the ES/calibration window."""
    rp = dict(reg_params or REG_PARAMS)
    lab = ((y > THR) & (y < 200000)).astype(int).to_numpy()

    cals = []
    for s in range(n_seeds):
        cp = dict(CLF_PARAMS, seed=42 + s, bagging_fraction=0.8, bagging_freq=1)
        c = lgb.train(cp, lgb.Dataset(X[~es_mask], lab[~es_mask], categorical_feature=cat_cols),
                      num_boost_round=2000,
                      valid_sets=[lgb.Dataset(X[es_mask], lab[es_mask])],
                      callbacks=[lgb.early_stopping(100, verbose=False)])
        cals.append(_PlattCal(c).fit(X[es_mask], lab[es_mask]))
    iso = _SeedEnsembleCal(cals)
    clf = cals[0].clf  # kept for introspection

    normal = y.between(30, THR).to_numpy()
    mA = lgb.train(rp, lgb.Dataset(X[normal & ~es_mask], y[normal & ~es_mask], categorical_feature=cat_cols),
                   num_boost_round=6000,
                   valid_sets=[lgb.Dataset(X[normal & es_mask], y[normal & es_mask])],
                   callbacks=[lgb.early_stopping(150, verbose=False)])

    ext = ((y > THR) & (y < 200000)).to_numpy()
    mB = lgb.train(EXT_PARAMS, lgb.Dataset(X[ext], y[ext].clip(upper=CAP), categorical_feature=cat_cols),
                   num_boost_round=300)
    return {"clf": clf, "iso": iso, "mA": mA, "mB": mB}


def predict_mixture(mdl, X):
    p = mdl["iso"].predict(X)
    pA = mdl["mA"].predict(X, num_iteration=mdl["mA"].best_iteration)
    pB = mdl["mB"].predict(X)
    return ((1 - p) * pA + p * pB).clip(60, CAP), p
