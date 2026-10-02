"""M52: day-offset artifacts are detectable by d_sched - and we excluded them from training.

Each such flight costs (86400-1000)^2 = 7.3e9, half our entire test error budget.
They were filtered out of training as data errors, which is right for fitting but
wrong for prediction: under RMSE the optimal prediction on a detectable population
is its conditional mean, artifacts included.
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
PROC = ROOT / "data" / "processed"

X = pd.read_parquet(PROC / "X_v2_full_t.parquet", columns=["d_sched", "has_fpl", "AIRPORT"])
Xr = pd.read_parquet(PROC / "X_v2_ranking_t.parquet", columns=["d_sched", "has_fpl", "AIRPORT"])
meta = pd.read_parquet(PROC / "meta_train2025.parquet")
y = meta["y"].astype("float64").to_numpy()
ds, dsr = X["d_sched"].to_numpy(), Xr["d_sched"].to_numpy()

art = np.zeros(len(y), bool)
for k in (1, 2, 3):
    art |= np.abs(y - k * 86400) < 3600
ok = (y > 0) & (y < 200000)
print("artefacts=%d sur %d vols valides | d_sched dispo: train %.1f%% / ranking %.1f%%" % (
    (art & ok).sum(), ok.sum(), 100 * np.isfinite(ds).mean(), 100 * np.isfinite(dsr).mean()))

print("\n=== detecteur d_sched > seuil (sur 2025 complet, artefacts INCLUS) ===")
print("  %-9s %8s %6s %9s %8s %11s %12s" % ("seuil", "cand.", "art.", "precis.", "rappel",
                                            "moy. y cand", "cand. rank."))
best = None
for thr in (21600, 32400, 43200, 54000, 64800, 75600):
    c = ok & np.isfinite(ds) & (ds > thr)
    a = c & art
    if not c.sum():
        continue
    cr = int((np.isfinite(dsr) & (dsr > thr)).sum())
    print("  %-9d %8d %6d %8.1f%% %7.1f%% %11.0f %12d" % (
        thr, c.sum(), a.sum(), 100 * a.sum() / c.sum(),
        100 * a.sum() / (art & ok).sum(), y[c].mean(), cr))

# expected value of acting, at the best operating point
print("\n=== ce que coute / rapporte la correction (seuil 43200s = 12h) ===")
thr = 43200
c = ok & np.isfinite(ds) & (ds > thr)
cr = np.isfinite(dsr) & (dsr > thr)
p_art = (c & art).sum() / c.sum()
mu = y[c].mean()
mu_clean = y[c & ~art].mean()
print("  parmi les candidats: P(artefact)=%.1f%% | moyenne y=%.0fs | moyenne y hors artefacts=%.0fs" % (
    100 * p_art, mu, mu_clean))
print("  prediction actuelle du modele sur ce type de vol ~ %.0fs" % mu_clean)
print("  candidats dans le jeu de classement: %d" % cr.sum())
n = cr.sum()
gain_sq = n * ((mu_clean - mu) ** 2)
tot = 215876 * 260.13 ** 2
print("  en predisant %.0fs au lieu de %.0fs sur ces %d vols:" % (mu, mu_clean, n))
print("    variation d erreur quadratique = %.3e sur un total de %.3e" % (gain_sq, tot))
print("    RMSE si le pari est BON  = %.2f" % np.sqrt(max(tot - gain_sq, 0) / 215876))
print("    RMSE si le pari est NUL (aucun artefact reel) = %.2f" % np.sqrt((tot + gain_sq) / 215876))
