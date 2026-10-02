"""M93: is our January-2026 prediction level actually biased?

Drop PRU: its published figure does not reconstruct the mean taxi-out (M92 showed a
-107s to +366s airport-specific gap against our own d_aobt3, so their filtering and
definitions differ). Use instead the quantity we measure directly on the very flights
being scored: d_aobt3 = MVT_TIME - AOBT_3, Network Manager's own taxi time.

target = d_aobt3 + (BLOCK - AOBT_3 disagreement), and that disagreement is a property
of the airport's recording, roughly stable year to year. So:

    expected Jan-2026 mean  =  observed Jan-2025 mean  +  (d_aobt3 26 - d_aobt3 25)

Same filter on every quantity, and the model's own holdout bias reported alongside so
the comparison has a scale.
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from dc26.data import load_movements_2025, load_ranking_2026

LO, HI = 60, 7200


def prep(df, phase="DEP", month=1):
    d = df[(df["PHASE_mvt"] == phase) & (df["MVT_TIME_UTC_mvt"].dt.month == month)].copy()
    d["da"] = (d["MVT_TIME_UTC_mvt"] - d["AOBT_3_flt"]).dt.total_seconds()
    return d


d25 = prep(load_movements_2025())
d25["y"] = d25["TAXITIME_SEC_mvt"].astype(float)
rk = load_ranking_2026()
d26 = prep(rk)
sub = pd.read_parquet(ROOT / "submissions" / "reliable-hamburger_v11.parquet")
d26 = d26.drop(columns=["TAXITIME_SEC_mvt"]).merge(sub, on="MVT_ID_mvt", how="left")
d26["pred"] = d26["TAXITIME_SEC_mvt"].astype(float)

print("=== niveau janvier : mesure directe, filtre identique partout (%d-%ds) ===" % (LO, HI))
print("  %-7s %10s %10s %10s %11s %11s %8s" % (
    "aero", "cible 25", "d_aobt3 25", "d_aobt3 26", "attendu 26", "notre pred", "ecart"))
tot = 0.0
rows = []
for a in sorted(d26["ADEP_mvt"].astype(str).unique()):
    m25 = (d25["AIRPORT"].astype(str) == a) & d25["da"].between(LO, HI) & d25["y"].between(LO, HI)
    m26 = (d26["ADEP_mvt"].astype(str) == a) & d26["da"].between(LO, HI)
    if m25.sum() < 500 or m26.sum() < 500:
        continue
    y25, a25, a26 = d25.loc[m25, "y"].mean(), d25.loc[m25, "da"].mean(), d26.loc[m26, "da"].mean()
    att = y25 + (a26 - a25)
    pr = d26.loc[m26, "pred"].mean()
    e = pr - att
    n = int(m26.sum())
    tot += n * e ** 2
    rows.append((a, n, e))
    print("  %-7s %10.0f %10.0f %10.0f %11.0f %11.0f %+8.0f" % (a, y25, a25, a26, att, pr, e))

print("\n  cout quadratique du desalignement : %.3e sur un budget de %.3e" % (
    tot, 215876 * 260.13 ** 2))
print("  gain RMSE si les niveaux etaient cales : %+.2f" % (
    np.sqrt(max(215876 * 260.13 ** 2 - tot, 0) / 215876) - 260.13))

# control: the same estimator applied INSIDE 2025 (Jan vs Jul), where truth is known
print("\n=== controle : le meme estimateur sur 2025, ou la verite est connue ===")
j25 = prep(load_movements_2025(), month=7)
j25["y"] = j25["TAXITIME_SEC_mvt"].astype(float)
print("  %-7s %11s %11s %8s" % ("aero", "attendu jul", "reel jul", "erreur"))
err = []
for a in sorted(d26["ADEP_mvt"].astype(str).unique()):
    m1 = (d25["AIRPORT"].astype(str) == a) & d25["da"].between(LO, HI) & d25["y"].between(LO, HI)
    m7 = (j25["AIRPORT"].astype(str) == a) & j25["da"].between(LO, HI) & j25["y"].between(LO, HI)
    if m1.sum() < 500 or m7.sum() < 500:
        continue
    att = d25.loc[m1, "y"].mean() + (j25.loc[m7, "da"].mean() - d25.loc[m1, "da"].mean())
    real = j25.loc[m7, "y"].mean()
    err.append(att - real)
    print("  %-7s %11.0f %11.0f %+8.0f" % (a, att, real, att - real))
print("\n  erreur de l estimateur lui-meme : moyenne %+.0fs, ecart-type %.0fs" % (
    np.mean(err), np.std(err)))
