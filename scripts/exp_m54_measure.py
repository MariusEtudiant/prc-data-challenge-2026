"""M54: honest measurement on the mirror holdout - what does the blend actually
predict on the detected flights, and what would predicting the conditional mean win?

Conditional mean is fitted on the TRAINING part only, applied to the holdout.
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
PROC, EXP = ROOT / "data" / "processed", ROOT / "experiments"
from dc26.validation import rmse

X = pd.read_parquet(PROC / "X_v2_mirror_t.parquet", columns=["d_sched", "has_fpl"])
meta = pd.read_parquet(PROC / "meta_train2025.parquet")
y = meta["y"].astype("float64").to_numpy()
tr, va = meta["tr"].to_numpy(), meta["va"].to_numpy()
art = np.zeros(len(y), bool)
for k in (1, 2, 3):
    art |= np.abs(y - k * 86400) < 3600
ok = (y > 0) & (y < 200000)
ds, fp = X["d_sched"].to_numpy(), X["has_fpl"].to_numpy()
pred = np.load(EXP / "m49_blend_va.npy")
y_va = y[va]
g = np.isfinite(pred)

for thr in (43200, 54000, 64800):
    det = np.isfinite(ds) & (ds > thr) & (fp == 0)
    mu = y[ok & tr & det].mean()          # conditional mean, TRAINING rows only
    n_tr = int((ok & tr & det).sum())
    d_va = det[va] & g
    if not d_va.any():
        print("seuil %d: aucun candidat dans le holdout" % thr)
        continue
    base = pred.copy()
    corr = pred.copy()
    corr[d_va] = mu
    print("\n=== seuil %ds, sans plan de vol ===" % thr)
    print("  moyenne conditionnelle apprise sur %d vols d entrainement = %.0fs" % (n_tr, mu))
    print("  candidats dans le holdout: %d (dont %d artefacts, %d >5h)" % (
        d_va.sum(), (d_va & art[va]).sum(), (d_va & (y_va > 18000)).sum()))
    print("  prediction actuelle du modele sur eux: mediane=%.0fs moyenne=%.0fs" % (
        np.median(pred[d_va]), pred[d_va].mean()))
    print("  verite sur eux: mediane=%.0fs moyenne=%.0fs" % (np.median(y_va[d_va]), y_va[d_va].mean()))
    print("  RMSE holdout  actuel = %.2f" % rmse(y_va[g], base[g]))
    print("  RMSE holdout corrige = %.2f  (%+.2f)" % (
        rmse(y_va[g], corr[g]), rmse(y_va[g], corr[g]) - rmse(y_va[g], base[g])))
