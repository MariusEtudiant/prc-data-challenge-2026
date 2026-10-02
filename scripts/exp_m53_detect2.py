"""M53: refine the artifact detector. 14 of 15 have no flight-plan match.
Does combining that with d_sched raise precision enough to make the bet positive?
Also: is the detector stable, or fitted to 15 rows? Check by splitting the year.
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
PROC = ROOT / "data" / "processed"

cols = ["d_sched", "has_fpl", "AIRPORT", "doy"]
X = pd.read_parquet(PROC / "X_v2_full_t.parquet", columns=cols)
Xr = pd.read_parquet(PROC / "X_v2_ranking_t.parquet", columns=cols)
meta = pd.read_parquet(PROC / "meta_train2025.parquet")
y = meta["y"].astype("float64").to_numpy()
art = np.zeros(len(y), bool)
for k in (1, 2, 3):
    art |= np.abs(y - k * 86400) < 3600
ok = (y > 0) & (y < 200000)

ds, fp = X["d_sched"].to_numpy(), X["has_fpl"].to_numpy()
dsr, fpr = Xr["d_sched"].to_numpy(), Xr["has_fpl"].to_numpy()

print("=== detecteur combine : pas de plan de vol ET d_sched > seuil ===")
print("  %-9s %8s %6s %9s %8s %11s %12s" % ("seuil", "cand.", "art.", "precis.", "rappel",
                                            "moy. y cand", "cand. rank."))
for thr in (21600, 43200, 54000, 64800, 75600, 82800):
    c = ok & (fp == 0) & np.isfinite(ds) & (ds > thr)
    a = c & art
    cr = int(((fpr == 0) & np.isfinite(dsr) & (dsr > thr)).sum())
    if not c.sum():
        continue
    print("  %-9d %8d %6d %8.1f%% %7.1f%% %11.0f %12d" % (
        thr, c.sum(), a.sum(), 100 * a.sum() / c.sum(),
        100 * a.sum() / (art & ok).sum(), y[c].mean(), cr))

print("\n=== stabilite : le detecteur tient-il hors echantillon ? ===")
h1, h2 = X["doy"].to_numpy() < 183, X["doy"].to_numpy() >= 183
for thr in (43200, 64800):
    print("  seuil %d:" % thr)
    for lab, h in [("1er semestre", h1), ("2e semestre", h2)]:
        c = ok & h & (fp == 0) & np.isfinite(ds) & (ds > thr)
        a = c & art
        print("    %-14s %5d candidats, %2d artefacts (%.1f%%), moyenne y=%.0fs" % (
            lab, c.sum(), a.sum(), 100 * a.sum() / max(c.sum(), 1), y[c].mean() if c.sum() else 0))

print("\n=== esperance du pari, seuil 64800s sans plan de vol ===")
thr = 64800
c = ok & (fp == 0) & np.isfinite(ds) & (ds > thr)
cr = (fpr == 0) & np.isfinite(dsr) & (dsr > thr)
n, p_art = int(cr.sum()), (c & art).sum() / max(c.sum(), 1)
mu, mu_clean = y[c].mean(), y[c & ~art].mean()
tot = 215876 * 260.13 ** 2
print("  %d candidats dans le classement | P(artefact)=%.1f%% -> %.1f artefacts attendus" % (
    n, 100 * p_art, n * p_art))
print("  moyenne conditionnelle=%.0fs (hors artefacts %.0fs)" % (mu, mu_clean))
for k in (0, 1, 2, 3):
    gain = k * ((86400 - mu_clean) ** 2 - (86400 - mu) ** 2) - (n - k) * (mu - mu_clean) ** 2
    print("    si %d artefact(s) reellement present(s) -> RMSE %.2f (%+.2f)" % (
        k, np.sqrt((tot - gain) / 215876), np.sqrt((tot - gain) / 215876) - 260.13))
