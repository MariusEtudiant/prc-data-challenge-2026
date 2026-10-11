"""U06: are the flights without AOBT_3 predicted at their conditional mean, or far above it?

Rome's worst errors are gross over-predictions on flights with no `AOBT_3` - 30,964 s
predicted where the taxi was 1,077 s. Under RMSE the right prediction for such a group is
its conditional mean, and if that group contains a few genuine multi-hour values among many
ordinary ones, its mean is a few thousand seconds, not thirty thousand. So either the model
is estimating the mean and the mean really is enormous, or it is keying on the handful of
extreme training rows and the group is simply miscalibrated.

The test is direct: compare the mean of the predictions against the mean of the truth, per
airport and per group, on held-out rows. A model estimating a conditional mean correctly has
those two equal whatever the spread. Any gap is bias, and bias is removable.

Also counts the implausible targets that the day-offset filter misses. It flags
|y - k*86400| < 3600, which catches a clean 24-hour shift but not 58,206 s or 64,437 s -
and those two flights alone dominate Paris and Rome.

Usage: python scripts/exp_u06_noaobt.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
PROC, EXP = ROOT / "data" / "processed", ROOT / "experiments"
from dc26.validation import rmse

pred = np.load(EXP / "t01_new.npy")
meta = pd.read_parquet(PROC / "meta_train2025.parquet")
y = meta["y"].astype("float64").to_numpy()
art = np.zeros(len(y), bool)
for k in (1, 2, 3):
    art |= np.abs(y - k * 86400) < 3600
apt = pd.read_parquet(PROC / "X_v2_mirror_t.parquet",
                      columns=["AIRPORT"])["AIRPORT"].astype(str).to_numpy()
mv = pd.read_parquet(PROC / "movements_2025.parquet",
                     columns=["PHASE_mvt", "AOBT_3_flt", "FLIGHT_ID_mvt"])
d = mv[mv["PHASE_mvt"] == "DEP"].reset_index(drop=True)
no_aobt = d["AOBT_3_flt"].isna().to_numpy()
g = np.isfinite(pred) & (y > 0) & (y < 200000) & ~art

print("=== part des vols sans AOBT_3, et ce qu'ils coutent ===")
print("  %-7s %9s %9s %10s %12s %12s" % ("aero", "n", "sans aobt", "part", "RMSE avec",
                                         "RMSE sans"))
for a in sorted(set(apt)):
    m = g & (apt == a)
    if m.sum() < 500:
        continue
    k = m & no_aobt
    e2 = (y[m] - pred[m]) ** 2
    share = ((y[k] - pred[k]) ** 2).sum() / e2.sum() if k.any() else 0
    print("  %-7s %9d %9d %9.2f%% %12.1f %12.1f" % (
        a, m.sum(), k.sum(), 100 * k.mean() / max(m.mean(), 1e-9) * m.mean() / m.mean()
        if False else 100 * k.sum() / m.sum(),
        rmse(y[m & no_aobt], pred[m & no_aobt]) if k.sum() > 5 else np.nan,
        rmse(y[m & ~no_aobt], pred[m & ~no_aobt])))
    if k.any():
        print("          ces %d vols portent %.1f%% de l'erreur carree de l'aeroport"
              % (k.sum(), 100 * share))

print("\n=== biais: moyenne predite contre moyenne vraie, sur les vols sans AOBT_3 ===")
print("  %-7s %8s %12s %12s %12s %10s" % ("aero", "n", "vrai moyen", "predit moyen",
                                          "biais", "vrai med"))
for a in sorted(set(apt)):
    k = g & (apt == a) & no_aobt
    if k.sum() < 20:
        continue
    print("  %-7s %8d %12.0f %12.0f %+12.0f %10.0f" % (
        a, k.sum(), y[k].mean(), pred[k].mean(), pred[k].mean() - y[k].mean(),
        np.median(y[k])))

print("\n=== le meme controle sur les vols QUI ONT un AOBT_3 ===")
print("  %-7s %8s %12s %12s %12s" % ("aero", "n", "vrai moyen", "predit moyen", "biais"))
for a in sorted(set(apt)):
    k = g & (apt == a) & ~no_aobt
    if k.sum() < 500:
        continue
    print("  %-7s %8d %12.0f %12.0f %+12.0f" % (
        a, k.sum(), y[k].mean(), pred[k].mean(), pred[k].mean() - y[k].mean()))

print("\n=== cibles invraisemblables que le filtre d'artefacts ne voit pas ===")
print("  le filtre retient |y - k*86400| < 3600 pour k = 1,2,3")
yy = meta["y"].astype("float64").to_numpy()
valid = (yy > 0) & (yy < 200000)
for lo, lab in ((14400, "> 4 h"), (21600, "> 6 h"), (28800, "> 8 h"), (43200, "> 12 h")):
    m = valid & (yy > lo)
    print("  %-8s %6d vols | dont deja etiquetes artefact: %4d | restants: %4d" % (
        lab, m.sum(), (m & art).sum(), (m & ~art).sum()))
print("\n  les plus grands non etiquetes:")
m = valid & ~art & (yy > 14400)
idx = np.where(m)[0][np.argsort(yy[np.where(m)[0]])[::-1]][:15]
mv2 = pd.read_parquet(PROC / "movements_2025.parquet",
                      columns=["PHASE_mvt", "AIRPORT", "MVT_TIME_UTC_mvt",
                               "BLOCK_TIME_UTC_mvt", "AOBT_3_flt"])
d2 = mv2[mv2["PHASE_mvt"] == "DEP"].reset_index(drop=True)
print("  %-7s %-20s %-20s %9s %6s" % ("aero", "off-block", "decollage", "y", "aobt3"))
for i in idx:
    r = d2.iloc[i]
    print("  %-7s %-20s %-20s %9.0f %6s" % (
        r["AIRPORT"], str(r["BLOCK_TIME_UTC_mvt"])[:19], str(r["MVT_TIME_UTC_mvt"])[:19],
        yy[i], "oui" if pd.notna(r["AOBT_3_flt"]) else "non"))
print("\n  repartition par aeroport des cibles > 4 h non etiquetees:")
sub = pd.Series(apt[m]).value_counts()
print("   ", dict(sub))
