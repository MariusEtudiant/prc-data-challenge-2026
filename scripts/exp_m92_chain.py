"""M92: the decisive check. PRU measures ATOT - AOBT, which is exactly our d_aobt3
(MVT_TIME - AOBT_3). If PRU's published 2026 figure matches our own d_aobt3, then PRU
is consistent with the data we hold, and the only remaining question is whether our
predicted BLOCK-based taxi sits at the right distance from it.

Chain, per airport:
    PRU published   <->   our d_aobt3        (same quantity, should agree)
    d_aobt3         ->    target             (offset measurable on 2025)
    prediction      vs    d_aobt3 + offset   (what we should be predicting)
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
PROC = ROOT / "data" / "processed"
from dc26.data import load_movements_2025, load_ranking_2026

pru = pd.read_parquet(ROOT / "data" / "external" / "pru_atxot_monthly.parquet")
pru["pru"] = (pru["add_min_per_dep"] + pru["ref_min_per_dep"]) * 60

# ---- 2025 January: PRU vs our d_aobt3 vs our target
mv = load_movements_2025()
d = mv[(mv["PHASE_mvt"] == "DEP") & (mv["MVT_TIME_UTC_mvt"].dt.month == 1)].copy()
d["da"] = (d["MVT_TIME_UTC_mvt"] - d["AOBT_3_flt"]).dt.total_seconds()
d["y"] = d["TAXITIME_SEC_mvt"].astype(float)
ok25 = np.isfinite(d["y"]) & (d["y"] > 0) & (d["y"] < 7200)
p25 = pru[(pru["YEAR"] == 2025) & (pru["MONTH_NUM"] == 1)].set_index("APT_ICAO")["pru"]

# ---- 2026 January: PRU vs our d_aobt3 vs our prediction
rk = load_ranking_2026()
r = rk[(rk["PHASE_mvt"] == "DEP") & (rk["MVT_TIME_UTC_mvt"].dt.month == 1)].copy()
r["da"] = (r["MVT_TIME_UTC_mvt"] - r["AOBT_3_flt"]).dt.total_seconds()
sub = pd.read_parquet(ROOT / "submissions" / "reliable-hamburger_v11.parquet")
r = r.merge(sub, on="MVT_ID_mvt", how="left")
p26 = pru[(pru["YEAR"] == 2026) & (pru["MONTH_NUM"] == 1)].set_index("APT_ICAO")["pru"]

print("=== maillon 1 : PRU publie contre notre d_aobt3 (meme grandeur) ===")
print("  %-7s %10s %10s %8s   %10s %10s %8s" % (
    "aero", "PRU jan25", "d_aobt3", "ecart", "PRU jan26", "d_aobt3", "ecart"))
for a in sorted(p26.index):
    m25 = ok25 & (d["AIRPORT"] == a) & np.isfinite(d["da"]) & (d["da"].between(60, 7200))
    m26 = (r["ADEP_mvt"] == a) & np.isfinite(r["da"]) & (r["da"].between(60, 7200))
    if m25.sum() < 500 or m26.sum() < 500:
        continue
    a25, a26 = d.loc[m25, "da"].mean(), r.loc[m26, "da"].mean()
    print("  %-7s %10.0f %10.0f %+8.0f   %10.0f %10.0f %+8.0f" % (
        a, p25.get(a, np.nan), a25, a25 - p25.get(a, np.nan),
        p26[a], a26, a26 - p26[a]))

print("\n=== maillon 2 : ecart cible - d_aobt3 en janvier 2025, applique a 2026 ===")
print("  %-7s %12s %12s %13s %12s %8s" % (
    "aero", "decal. jan25", "d_aobt3 26", "attendu 2026", "notre pred.", "ecart"))
tot = 0.0
for a in sorted(p26.index):
    m25 = ok25 & (d["AIRPORT"] == a) & np.isfinite(d["da"]) & (d["da"].between(60, 7200))
    m26 = (r["ADEP_mvt"] == a) & np.isfinite(r["da"]) & (r["da"].between(60, 7200))
    if m25.sum() < 500 or m26.sum() < 500:
        continue
    off = (d.loc[m25, "y"] - d.loc[m25, "da"]).mean()
    att = r.loc[m26, "da"].mean() + off
    mine = r.loc[m26, "TAXITIME_SEC_mvt"].mean()
    e = mine - att
    tot += m26.sum() * e ** 2
    print("  %-7s %12.0f %12.0f %13.0f %12.0f %+8.0f" % (a, off, r.loc[m26, "da"].mean(), att, mine, e))
print("\n  cout quadratique du desalignement : %.3e sur %.3e" % (tot, 215876 * 260.13 ** 2))
