"""M59: train/test missingness audit. A feature that is never missing during
training but often missing at test time is a silent hazard: the model's NaN
routing was fitted on zero examples, so it is arbitrary.
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
PROC = ROOT / "data" / "processed"
from dc26.pipeline import align_categories

FT = ["X_v2_full_t.parquet", "flightctx_train2025.parquet", "depregime_train2025.parquet",
      "aobtq_full2025.parquet", "anomreg_train2025.parquet"]
FR = ["X_v2_ranking_t.parquet", "flightctx_ranking2026.parquet",
      "depregime_ranking2026.parquet", "aobtq_ranking2026.parquet", "anomreg_ranking2026.parquet"]
X = pd.concat([pd.read_parquet(PROC / f) for f in FT], axis=1)
Xr = align_categories(pd.concat([pd.read_parquet(PROC / f) for f in FR], axis=1), X)
doy = pd.read_parquet(PROC / "X_v2_full_t.parquet", columns=["doy"])["doy"].to_numpy()
jan = doy < 32

rows = []
for c in X.columns:
    a = float(X[c].isna().mean())
    aj = float(X.loc[jan, c].isna().mean())
    b = float(Xr[c].isna().mean())
    rows.append((c, 100 * a, 100 * aj, 100 * b, 100 * (b - aj)))
d = pd.DataFrame(rows, columns=["var", "train_all", "train_jan", "ranking", "ecart"])
d = d.reindex(d["ecart"].abs().sort_values(ascending=False).index)

print("=== variables dont le taux de valeurs absentes differe le plus (janvier 2025 -> test 2026) ===")
print("  %-30s %10s %10s %10s %9s" % ("variable", "train tot", "train jan", "test", "ecart"))
for _, r in d.head(14).iterrows():
    print("  %-30s %9.1f%% %9.1f%% %9.1f%% %+8.1f" % (
        r["var"], r["train_all"], r["train_jan"], r["ranking"], r["ecart"]))

haz = d[(d["train_jan"] < 1.0) & (d["ranking"] > 5.0)]
print("\n=== DANGEREUSES : jamais absentes a l entrainement, souvent absentes au test ===")
if len(haz):
    for _, r in haz.iterrows():
        print("  %-30s entrainement %.2f%% -> test %.1f%%" % (r["var"], r["train_jan"], r["ranking"]))
else:
    print("  aucune")
d.to_csv(ROOT / "reports" / "m59_missingness.csv", index=False)
print("\n(rapport complet: reports/m59_missingness.csv)")
