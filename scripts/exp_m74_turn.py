"""M74: the turnaround features were our single biggest feature win, yet they
correlate ~0 with the target at LIRF - our worst airport. Is the stand-based
link wrong there?

Validation that needs no ground truth: a genuine turnaround keeps the SAME
aircraft, so the linked arrival should share the departure's aircraft type and
operator. A low match rate means we are linking unrelated flights.
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from dc26.data import load_movements_2025

mv = load_movements_2025()
mv = mv[mv["MVT_TIME_UTC_mvt"].notna()].copy()
dep = mv[mv["PHASE_mvt"] == "DEP"]
arr = mv[mv["PHASE_mvt"] == "ARR"]

print("=== le lien de rotation par poste relie-t-il le meme appareil ? ===")
print("  %-7s %9s %10s %12s %12s %12s" % ("aero", "departs", "lies", "meme type", "meme operat.", "delai median"))
for a in sorted(mv["AIRPORT"].astype(str).unique()):
    d = dep[dep["AIRPORT"] == a][["MVT_TIME_UTC_mvt", "BLOCK_TIME_UTC_mvt", "STAND_mvt",
                                  "AIRCRAFT_TYPE_mvt", "AIRCRAFT_OPERATOR_flt"]].copy()
    r = arr[arr["AIRPORT"] == a][["MVT_TIME_UTC_mvt", "STAND_mvt", "AIRCRAFT_TYPE_mvt",
                                  "AIRCRAFT_OPERATOR_flt"]].copy()
    d = d[d["STAND_mvt"].notna() & d["BLOCK_TIME_UTC_mvt"].notna()]
    r = r[r["STAND_mvt"].notna()]
    if len(d) < 1000 or len(r) < 1000:
        continue
    d = d.sort_values("BLOCK_TIME_UTC_mvt")
    r = r.sort_values("MVT_TIME_UTC_mvt")
    j = pd.merge_asof(d, r, left_on="BLOCK_TIME_UTC_mvt", right_on="MVT_TIME_UTC_mvt",
                      by="STAND_mvt", direction="backward",
                      tolerance=pd.Timedelta("24h"), suffixes=("", "_arr"))
    ok = j["MVT_TIME_UTC_mvt_arr"].notna()
    if ok.sum() < 100:
        continue
    same_t = (j.loc[ok, "AIRCRAFT_TYPE_mvt"] == j.loc[ok, "AIRCRAFT_TYPE_mvt_arr"]).mean()
    same_o = (j.loc[ok, "AIRCRAFT_OPERATOR_flt"] == j.loc[ok, "AIRCRAFT_OPERATOR_flt_arr"]).mean()
    gap = (j.loc[ok, "BLOCK_TIME_UTC_mvt"] - j.loc[ok, "MVT_TIME_UTC_mvt_arr"]).dt.total_seconds()
    print("  %-7s %9d %9.1f%% %11.1f%% %11.1f%% %11.0f s" % (
        a, len(d), 100 * ok.mean(), 100 * same_t, 100 * same_o, np.median(gap)))

print("\n=== codes de poste : structure par aeroport ===")
for a in ("LIRF", "LTFM", "EGLL", "EDDM", "LEBL"):
    s = mv[mv["AIRPORT"] == a]["STAND_mvt"].dropna().astype(str)
    ex = s.value_counts().head(6)
    print("  %-6s %4d codes distincts | plus frequents: %s" % (
        a, s.nunique(), ", ".join("%s(%d)" % (k, v) for k, v in ex.items())))
