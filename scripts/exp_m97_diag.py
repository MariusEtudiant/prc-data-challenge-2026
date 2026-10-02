"""M97: is the new mirror's difficulty real, or an artefact of holding out all of July?

The mirror now removes every July row from training, so the model has zero summer
exposure - whereas the real fit WILL contain July 2025 when predicting July 2026. If
July's error is concentrated in seasonal effects the model was simply never shown,
the mirror overstates the difficulty and would mislead every variant comparison.

Checks: where the error sits by month and airport, and whether LIRF's July is a
regime episode (as February 2025 was) rather than ordinary traffic.
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
PROC, EXP = ROOT / "data" / "processed", ROOT / "experiments"
from dc26.validation import rmse

X = pd.read_parquet(PROC / "X_v2_mirror_t.parquet", columns=["AIRPORT"])
meta = pd.read_parquet(PROC / "meta_train2025.parquet")
y = meta["y"].astype("float64").to_numpy()
va = meta["va"].to_numpy()
p = np.load(EXP / "m96_reference.npy")
art = np.zeros(len(y), bool)
for k in (1, 2, 3):
    art |= np.abs(y - k * 86400) < 3600
yv, apt = y[va], X["AIRPORT"].astype(str).to_numpy()[va]
vp = meta.loc[va, "val_part"].to_numpy()
g = np.isfinite(p) & ~art[va]
sq = (yv - p) ** 2
tot = sq[g].sum()

print("=== part du budget d erreur, par aeroport et saison ===")
print("  %-8s %10s %10s %12s %12s" % ("aero", "n jan", "n jul", "part jan", "part jul"))
for a in sorted(set(apt)):
    mj, ml = g & (apt == a) & (vp == "jan"), g & (apt == a) & (vp == "jul")
    print("  %-8s %10d %10d %11.1f%% %11.1f%%" % (
        a, mj.sum(), ml.sum(), 100 * sq[mj].sum() / tot, 100 * sq[ml].sum() / tot))

print("\n=== LIRF juillet : trafic ordinaire ou episode de regime ? ===")
m = g & (apt == "LIRF") & (vp == "jul")
v = yv[m]
print("  n=%d | RMSE=%.1f" % (m.sum(), rmse(v, p[m])))
for lo, hi, lab in [(0, 3600, "< 1h"), (3600, 7200, "1-2h"), (7200, 18000, "2-5h"), (18000, 1e9, "> 5h")]:
    s = (v >= lo) & (v < hi)
    if s.sum():
        print("    %-8s n=%6d (%5.2f%%)  part de l erreur = %5.1f%%  RMSE segment=%8.1f" % (
            lab, s.sum(), 100 * s.mean(), 100 * ((v[s] - p[m][s]) ** 2).sum() / ((v - p[m]) ** 2).sum(),
            rmse(v[s], p[m][s])))
print("  RMSE LIRF juillet sur le trafic ordinaire seul (<1h) = %.1f" % rmse(
    v[v < 3600], p[m][v < 3600]))

print("\n=== LIRF : taux d extremes par mois sur 2025 (le modele voit-il un juillet comparable ?) ===")
mvt = pd.read_parquet(PROC / "movements_2025.parquet", columns=["AIRPORT", "PHASE_mvt", "MVT_TIME_UTC_mvt", "TAXITIME_SEC_mvt"])
d = mvt[(mvt["PHASE_mvt"] == "DEP") & (mvt["AIRPORT"] == "LIRF")].copy()
d["m"] = d["MVT_TIME_UTC_mvt"].dt.month
t = d["TAXITIME_SEC_mvt"].astype(float)
d = d[np.isfinite(t) & (t > 0) & (t < 200000)]
r = d.groupby("m")["TAXITIME_SEC_mvt"].agg(
    n="size", mediane="median", part_2h=lambda s: 100 * (s.astype(float) > 7200).mean(),
    part_5h=lambda s: 100 * (s.astype(float) > 18000).mean())
print(r.to_string(float_format=lambda v: "%.2f" % v))
