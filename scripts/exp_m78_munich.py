"""M78: Munich and Istanbul share the same handicap - AOBT_3 barely correlates with
the target (0.128 and 0.145). Munich reaches R2 0.856, the best of all ten airports;
Istanbul 0.609, the second worst. Whatever Munich substitutes for AOBT_3 has to be
visible in what its model actually uses.

Memory-lean: each feature file is subset to the two airports before concatenation,
so peak usage stays near one file rather than the whole 2M x 98 matrix.
"""
import gc
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
PROC = ROOT / "data" / "processed"

CAP = 86400
PAIR = ("EDDM", "LTFM")
FILES = ["X_v2_mirror_t.parquet", "flightctx_train2025.parquet", "depregime_train2025.parquet",
         "aobtq_train2025.parquet", "anomreg_train2025.parquet"]

apt_all = pd.read_parquet(PROC / "X_v2_mirror_t.parquet", columns=["AIRPORT"])["AIRPORT"].astype(str).to_numpy()
sel = np.isin(apt_all, PAIR)
print("lignes retenues: %d sur %d" % (sel.sum(), len(sel)), flush=True)

parts = []
for f in FILES:
    d = pd.read_parquet(PROC / f)
    parts.append(d[sel].reset_index(drop=True))
    del d
    gc.collect()
X = pd.concat(parts, axis=1)
del parts
gc.collect()

meta = pd.read_parquet(PROC / "meta_train2025.parquet")[sel].reset_index(drop=True)
y = meta["y"].astype("float64")
tr, es = meta["tr"].to_numpy(), meta["es"].to_numpy()
ok = (y > 0).to_numpy() & (y < 200000).to_numpy()
art = np.zeros(len(y), bool)
for k in (1, 2, 3):
    art |= (y - k * 86400).abs().to_numpy() < 3600
ok &= ~art
cat = [c for c in X.columns if X[c].dtype.name == "category"]
apt = X["AIRPORT"].astype(str).to_numpy()


def tuned(a):
    b = dict(objective="regression", metric="rmse", verbosity=-1, num_threads=0,
             bagging_freq=1, seed=42)
    d = json.loads((ROOT / "experiments" / ("optuna_" + a + ".json")).read_text())
    b.update({k: v for k, v in d.items() if not k.startswith("_")})
    for k in ("num_leaves", "min_data_in_leaf", "max_bin"):
        b[k] = int(b[k])
    return b


imps = {}
for a in PAIR:
    f, s = ok & tr & (apt == a) & ~es, ok & tr & (apt == a) & es
    m = lgb.train(tuned(a), lgb.Dataset(X[f], y[f].clip(upper=CAP), categorical_feature=cat),
                  num_boost_round=6000,
                  valid_sets=[lgb.Dataset(X[s], y[s].clip(upper=CAP))],
                  callbacks=[lgb.early_stopping(150, verbose=False)])
    v = pd.Series(m.feature_importance("gain"), index=X.columns)
    imps[a] = v / v.sum()
    print("%s: %d iterations, %d vols d entrainement" % (a, m.best_iteration, f.sum()), flush=True)
    del m
    gc.collect()

d = pd.DataFrame(imps)
d["ecart"] = d[PAIR[0]] - d[PAIR[1]]
print("\n=== ce sur quoi Munich s appuie bien plus qu Istanbul ===")
print("  %-34s %9s %9s" % ("variable", "EDDM", "LTFM"))
for c in d.sort_values("ecart", ascending=False).head(10).index:
    print("  %-34s %8.1f%% %8.1f%%" % (c, 100 * d.loc[c, "EDDM"], 100 * d.loc[c, "LTFM"]))
print("\n=== ce sur quoi Istanbul s appuie bien plus que Munich ===")
for c in d.sort_values("ecart").head(8).index:
    print("  %-34s %8.1f%% %8.1f%%" % (c, 100 * d.loc[c, "EDDM"], 100 * d.loc[c, "LTFM"]))

print("\n=== distribution de la cible (trafic ordinaire) ===")
for a in PAIR:
    v = y[ok & (apt == a)].to_numpy()
    v = v[v < 3600]
    print("  %-6s mediane=%5.0f  ecart-type=%5.0f  p10=%4.0f  p90=%5.0f  etendue=%5.0f" % (
        a, np.median(v), v.std(), np.percentile(v, 10), np.percentile(v, 90),
        np.percentile(v, 90) - np.percentile(v, 10)))
