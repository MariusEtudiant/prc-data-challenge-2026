"""M90: EUROCONTROL publishes the mean taxi-out per airport for January 2026 - the
test period itself. We feed it to the model as an input but never use it as a
calibration constraint on the LEVEL of our predictions.

PRU measures ATOT - AOBT (Network Manager timestamps) over its own valid-flight
population; our target is MVT_TIME - BLOCK_TIME (airport timestamps). The two differ
systematically, so step one is to measure that offset on 2025, where both are known.
Only if the 2025 relation is tight does the 2026 published value say anything about
our predictions.
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
PROC = ROOT / "data" / "processed"
from dc26.data import load_movements_2025

pru = pd.read_parquet(ROOT / "data" / "external" / "pru_atxot_monthly.parquet")
pru["pru_mean_s"] = (pru["add_min_per_dep"] + pru["ref_min_per_dep"]) * 60

mv = load_movements_2025()
d = mv[mv["PHASE_mvt"] == "DEP"].copy()
d["m"] = d["MVT_TIME_UTC_mvt"].dt.month
y = d["TAXITIME_SEC_mvt"].astype(float)
ok = np.isfinite(y) & (y > 0) & (y < 7200)
d, y = d[ok], y[ok]

obs = d.groupby(["AIRPORT", "m"], observed=True)["TAXITIME_SEC_mvt"].agg(["mean", "count"]).reset_index()
obs.columns = ["APT_ICAO", "MONTH_NUM", "obs_mean_s", "n"]
j = obs.merge(pru[pru["YEAR"] == 2025][["APT_ICAO", "MONTH_NUM", "pru_mean_s"]],
              on=["APT_ICAO", "MONTH_NUM"], how="inner")
j["ecart"] = j["obs_mean_s"] - j["pru_mean_s"]

print("=== 2025 : notre cible contre la publication PRU, par aeroport ===")
print("  %-7s %6s %12s %12s %10s %10s" % ("aero", "mois", "notre moy.", "PRU", "ecart moy.", "ecart-type"))
for a, g in j.groupby("APT_ICAO"):
    print("  %-7s %6d %12.0f %12.0f %10.0f %10.1f" % (
        a, len(g), g["obs_mean_s"].mean(), g["pru_mean_s"].mean(), g["ecart"].mean(), g["ecart"].std()))
print("\n  correlation mensuelle (apres retrait de la moyenne par aeroport) : %.3f" % (
    j.assign(o=j["obs_mean_s"] - j.groupby("APT_ICAO")["obs_mean_s"].transform("mean"),
             p=j["pru_mean_s"] - j.groupby("APT_ICAO")["pru_mean_s"].transform("mean"))
     .pipe(lambda x: np.corrcoef(x["o"], x["p"])[0, 1])))

# ---- what does PRU imply for January 2026, and what do we predict?
off = j.groupby("APT_ICAO")["ecart"].mean()
p26 = pru[(pru["YEAR"] == 2026) & (pru["MONTH_NUM"] == 1)].set_index("APT_ICAO")["pru_mean_s"]
sub = pd.read_parquet(ROOT / "submissions" / "reliable-hamburger_v11.parquet")
rk = pd.read_parquet(ROOT / "data" / "ranking.parquet", columns=["MVT_ID_mvt", "ADEP_mvt", "PHASE_mvt", "MVT_TIME_UTC_mvt"])
dep = rk[rk["PHASE_mvt"] == "DEP"].merge(sub, on="MVT_ID_mvt")
dep = dep[dep["MVT_TIME_UTC_mvt"].dt.month == 1]
mine = dep.groupby("ADEP_mvt", observed=True)["TAXITIME_SEC_mvt"].agg(["mean", "count"])

print("\n=== janvier 2026 : notre prediction contre l attendu PRU + decalage 2025 ===")
print("  %-7s %8s %12s %12s %12s %10s" % ("aero", "n", "notre pred.", "PRU brut", "PRU+decal.", "ecart"))
tot = 0.0
for a in sorted(mine.index):
    if a not in p26.index or a not in off.index:
        continue
    att = p26[a] + off[a]
    e = mine.loc[a, "mean"] - att
    tot += mine.loc[a, "count"] * e ** 2
    print("  %-7s %8d %12.0f %12.0f %12.0f %+10.0f" % (
        a, mine.loc[a, "count"], mine.loc[a, "mean"], p26[a], att, e))
n = int(mine["count"].sum())
print("\n  cout quadratique du decalage de niveau : %.3e sur un budget de %.3e" % (tot, 215876 * 260.13 ** 2))
print("  gain RMSE si tous les niveaux etaient corriges : %.2f" % (
    np.sqrt((215876 * 260.13 ** 2 - tot) / 215876) - 260.13))
