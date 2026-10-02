"""Build submission files from full-2025 retrains:
  v3 = m3bw  : single LGBM, extremes kept (cap 86400), M5 features  [stable]
  v4 = mixture: 3-seed Platt-calibrated regime mixture              [high var, high upside]
Leaderboard keeps the BEST submission, so both are submitted."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from dc26 import load_movements_2025, load_ranking_2026
from dc26.state_features import get_state_features
from dc26.features import build_features_v1
from dc26.weather import build_weather_features, load_metar
from dc26.mixture import fit_mixture, predict_mixture, REG_PARAMS

mv = load_movements_2025()
dep = mv[mv["PHASE_mvt"] == "DEP"].copy()
y = dep["TAXITIME_SEC_mvt"]

met = load_metar()
X = pd.concat([build_features_v1(dep), get_state_features("train2025", dep, mv),
               build_weather_features(dep, met)], axis=1)
cat_cols = [c for c in X.columns if X[c].dtype.name == "category"]

rk = load_ranking_2026()
rk_dep = rk[rk["PHASE_mvt"] == "DEP"].copy()
X_rk = pd.concat([build_features_v1(rk_dep), get_state_features("ranking2026", rk_dep, rk),
                  build_weather_features(rk_dep, met)], axis=1)
for c in cat_cols:
    X_rk[c] = pd.Categorical(X_rk[c], categories=X[c].cat.categories)
print(f"train X: {X.shape} | ranking X: {X_rk.shape}", flush=True)

es = (dep["MVT_TIME_UTC_mvt"] >= "2025-12-10").to_numpy()
sub_tpl = pd.read_parquet(ROOT / "data" / "submitting.parquet").drop(columns=["TAXITIME_SEC_mvt"])


def write_sub(version, pred):
    out = sub_tpl.merge(pd.DataFrame({"MVT_ID_mvt": rk_dep["MVT_ID_mvt"].values,
                                      "TAXITIME_SEC_mvt": pred}), on="MVT_ID_mvt", how="left")
    assert out["TAXITIME_SEC_mvt"].notna().all() and len(out) == 215876
    path = ROOT / "submissions" / f"reliable-hamburger_v{version}.parquet"
    out.to_parquet(path, index=False)
    print(f"[v{version}] written: mean={out['TAXITIME_SEC_mvt'].mean():.0f}s "
          f"median={out['TAXITIME_SEC_mvt'].median():.0f}s "
          f"q99={out['TAXITIME_SEC_mvt'].quantile(0.99):.0f}s "
          f">2h: {(out['TAXITIME_SEC_mvt'] > 7200).sum()}", flush=True)


# ---- v3: m3bw
keep = (y > 0) & (y < 200000)
yk = y[keep].clip(upper=86400)
es_k = es[keep.to_numpy()]
m3 = lgb.train(REG_PARAMS, lgb.Dataset(X[keep][~es_k], yk[~es_k], categorical_feature=cat_cols),
               num_boost_round=6000,
               valid_sets=[lgb.Dataset(X[keep][es_k], yk[es_k])],
               callbacks=[lgb.early_stopping(150, verbose=False)])
print(f"[v3] m3bw iters={m3.best_iteration}", flush=True)
m3.save_model(str(ROOT / "models" / "sub_v3_m3bw.txt"))
write_sub(3, m3.predict(X_rk, num_iteration=m3.best_iteration).clip(60, 86400))

# ---- v4: mixture
mdl = fit_mixture(X, y, es, cat_cols)
pred4, p4 = predict_mixture(mdl, X_rk)
print(f"[v4] mixture p: mean={p4.mean():.5f} max={p4.max():.3f} n(p>0.5)={(p4 > 0.5).sum()}", flush=True)
mdl["mA"].save_model(str(ROOT / "models" / "sub_v4_mA.txt"))
mdl["mB"].save_model(str(ROOT / "models" / "sub_v4_mB.txt"))
write_sub(4, pred4)
print("done", flush=True)
