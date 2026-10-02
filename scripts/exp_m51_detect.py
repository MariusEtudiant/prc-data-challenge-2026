"""M51: are the catastrophic flights visible in d_aobt3 = MVT - AOBT_3?

The target is MVT - BLOCK. AOBT_3 is Network Manager's independent record of the
same off-block event. If BLOCK_TIME carries a date-entry error, d_aobt3 stays
normal and the flight is invisible. If instead the whole record is shifted,
d_aobt3 shows it - and the flight is detectable in the ranking set, where
d_aobt3 is observable but the target is not.
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
PROC = ROOT / "data" / "processed"

X = pd.read_parquet(PROC / "X_v2_full_t.parquet", columns=["d_aobt3", "d_sched", "d_lobt", "has_fpl", "AIRPORT"])
meta = pd.read_parquet(PROC / "meta_train2025.parquet")
y = meta["y"].astype("float64").to_numpy()
d = X["d_aobt3"].to_numpy()

art = np.zeros(len(y), bool)
for k in (1, 2, 3):
    art |= np.abs(y - k * 86400) < 3600
ok = (y > 0) & (y < 200000)
ghost = ok & ~art & (y > 18000)
normal = ok & ~art & (y < 3600)

print("=== d_aobt3 (MVT - AOBT_3) par population ===")
print("  %-24s %5s %8s %10s %10s %10s" % ("population", "n", "dispo", "median", "moyenne", "max"))
for lab, m in [("normal (<1h)", normal), ("artefacts jour", art & ok), ("fantomes >5h", ghost)]:
    v = d[m]
    fin = np.isfinite(v)
    print("  %-24s %5d %7.0f%% %10.0f %10.0f %10.0f" % (
        lab, m.sum(), 100 * fin.mean(), np.median(v[fin]) if fin.any() else -1,
        v[fin].mean() if fin.any() else -1, v[fin].max() if fin.any() else -1))

print("\n=== les artefacts jour, un par un ===")
idx = np.where(art & ok)[0]
print("  %-10s %10s %10s %10s %6s" % ("cible y", "d_aobt3", "d_sched", "d_lobt", "fpl"))
for i in idx:
    print("  %-10.0f %10s %10s %10s %6d" % (
        y[i], "%.0f" % d[i] if np.isfinite(d[i]) else "absent",
        "%.0f" % X["d_sched"].iloc[i] if np.isfinite(X["d_sched"].iloc[i]) else "absent",
        "%.0f" % X["d_lobt"].iloc[i] if np.isfinite(X["d_lobt"].iloc[i]) else "absent",
        X["has_fpl"].iloc[i]))

# how well does d_aobt3 alone separate? treat it as a detector for y > 18000
print("\n=== d_aobt3 comme detecteur (parmi les vols ou il est disponible) ===")
fin = np.isfinite(d) & ok & ~art
for thr in (7200, 10800, 18000):
    cand = fin & (d > thr)
    hit = cand & (y > 18000)
    print("  d_aobt3 > %5ds : %6d candidats, %3d vraiment >5h -> precision %5.2f%% | rappel %5.1f%%" % (
        thr, cand.sum(), hit.sum(), 100 * hit.sum() / max(cand.sum(), 1),
        100 * hit.sum() / max((fin & (y > 18000)).sum(), 1)))
    if cand.sum():
        print("        taxi moyen des candidats = %.0fs (mediane %.0fs)" % (y[cand].mean(), np.median(y[cand])))
