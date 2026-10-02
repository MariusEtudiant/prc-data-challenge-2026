"""M91: M90 compared a January prediction against an ANNUAL-average offset, which
is confounded - January taxi-out is longer than the yearly mean. Redo it January
against January, and check the population mismatch (PRU counts its own valid
flights, we count all departures).
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from dc26.data import load_movements_2025

pru = pd.read_parquet(ROOT / "data" / "external" / "pru_atxot_monthly.parquet")
pru["pru_mean_s"] = (pru["add_min_per_dep"] + pru["ref_min_per_dep"]) * 60

mv = load_movements_2025()
d = mv[mv["PHASE_mvt"] == "DEP"].copy()
d["m"] = d["MVT_TIME_UTC_mvt"].dt.month
yv = d["TAXITIME_SEC_mvt"].astype(float)
d = d[np.isfinite(yv) & (yv > 0) & (yv < 7200)]

obs = d.groupby(["AIRPORT", "m"], observed=True)["TAXITIME_SEC_mvt"].agg(["mean", "count"]).reset_index()
obs.columns = ["APT_ICAO", "MONTH_NUM", "obs_mean_s", "n_obs"]
j = obs.merge(pru[pru["YEAR"] == 2025][["APT_ICAO", "MONTH_NUM", "pru_mean_s", "VALID_FL"]],
              on=["APT_ICAO", "MONTH_NUM"], how="inner")
j["ecart"] = j["obs_mean_s"] - j["pru_mean_s"]

print("=== saisonnalite : le decalage varie-t-il selon le mois ? (moyenne des 10 aeroports) ===")
s = j.groupby("MONTH_NUM")[["obs_mean_s", "pru_mean_s", "ecart"]].mean()
print("  %-6s %12s %12s %10s" % ("mois", "notre moy.", "PRU", "ecart"))
for m, r in s.iterrows():
    print("  %-6d %12.0f %12.0f %+10.0f" % (m, r["obs_mean_s"], r["pru_mean_s"], r["ecart"]))

print("\n=== couverture PRU : quelle part de nos departs comptent-ils ? ===")
j["couv"] = j["VALID_FL"] / j["n_obs"]
print("  %-7s %12s %12s %10s" % ("aero", "nos departs", "PRU VALID_FL", "couverture"))
for a, g in j.groupby("APT_ICAO"):
    print("  %-7s %12.0f %12.0f %9.0f%%" % (a, g["n_obs"].mean(), g["VALID_FL"].mean(), 100 * g["couv"].mean()))

print("\n=== JANVIER contre JANVIER ===")
jan25 = j[j["MONTH_NUM"] == 1].set_index("APT_ICAO")
p26 = pru[(pru["YEAR"] == 2026) & (pru["MONTH_NUM"] == 1)].set_index("APT_ICAO")
sub = pd.read_parquet(ROOT / "submissions" / "reliable-hamburger_v11.parquet")
rk = pd.read_parquet(ROOT / "data" / "ranking.parquet",
                     columns=["MVT_ID_mvt", "ADEP_mvt", "PHASE_mvt", "MVT_TIME_UTC_mvt"])
dep = rk[rk["PHASE_mvt"] == "DEP"].merge(sub, on="MVT_ID_mvt")
dep = dep[dep["MVT_TIME_UTC_mvt"].dt.month == 1]
mine = dep.groupby("ADEP_mvt", observed=True)["TAXITIME_SEC_mvt"].agg(["mean", "count"])

print("  %-7s %8s %11s %11s %11s %11s %11s %8s" % (
    "aero", "n", "obs jan25", "PRU jan25", "PRU jan26", "attendu 26", "notre pred", "ecart"))
tot = 0.0
for a in sorted(mine.index):
    if a not in jan25.index or a not in p26.index:
        continue
    off = jan25.loc[a, "ecart"]                     # January-specific offset
    ratio = p26.loc[a, "pru_mean_s"] / jan25.loc[a, "pru_mean_s"]
    att = jan25.loc[a, "obs_mean_s"] * ratio        # scale our own Jan-2025 level by PRU's own change
    e = mine.loc[a, "mean"] - att
    tot += mine.loc[a, "count"] * e ** 2
    print("  %-7s %8d %11.0f %11.0f %11.0f %11.0f %11.0f %+8.0f" % (
        a, mine.loc[a, "count"], jan25.loc[a, "obs_mean_s"], jan25.loc[a, "pru_mean_s"],
        p26.loc[a, "pru_mean_s"], att, mine.loc[a, "mean"], e))
print("\n  cout quadratique : %.3e sur %.3e" % (tot, 215876 * 260.13 ** 2))
print("  gain RMSE si les niveaux etaient cales : %+.2f" % (
    np.sqrt(max(215876 * 260.13 ** 2 - tot, 0) / 215876) - 260.13))
