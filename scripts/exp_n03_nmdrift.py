"""N03: measure Network Manager's reporting quality inside the test period itself.

Established today: NM is precise on the event radar sees (landing, sd 57s) and six
times worse on the one nobody observes (pushback, sd 355s). Our whole prediction
leans on that noisy pushback timestamp, so anything that says WHEN it is trustworthy
is worth more than another tree.

In the ranking file we cannot measure AOBT_3's error on any departure - BLOCK is
blanked. But arrivals keep BOTH timestamps: the airport's BLOCK_TIME and NM's
ARVT_3. So NM's reporting quality is directly observable in the test period.

Question: does NM's error on arrivals move together with its error on departures at
the same airport and time? If it does, the arrival-side residual is a legitimate
transductive feature for the departure side - computable identically on train and
test, which is exactly the property that made the turnaround features work.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
PROC = ROOT / "data" / "processed"

mv = pd.read_parquet(PROC / "movements_2025.parquet",
                     columns=["AIRPORT", "PHASE_mvt", "MVT_TIME_UTC_mvt",
                              "BLOCK_TIME_UTC_mvt", "AOBT_3_flt", "ARVT_3_flt",
                              "AIRCRAFT_OPERATOR_flt"])
arr = mv[mv["PHASE_mvt"] == "ARR"].copy()
dep = mv[mv["PHASE_mvt"] == "DEP"].copy()

# NM's error on the observed event (arrivals) and on the unobserved one (departures)
arr["err"] = (arr["ARVT_3_flt"] - arr["MVT_TIME_UTC_mvt"]).dt.total_seconds()
dep["err"] = (dep["AOBT_3_flt"] - dep["BLOCK_TIME_UTC_mvt"]).dt.total_seconds()
arr = arr[np.isfinite(arr["err"]) & (arr["err"].abs() < 1800)]
dep = dep[np.isfinite(dep["err"]) & (dep["err"].abs() < 3600)]
for d in (arr, dep):
    d["day"] = d["MVT_TIME_UTC_mvt"].dt.floor("D")
    d["hour"] = d["MVT_TIME_UTC_mvt"].dt.floor("h")

print("=== correlation des erreurs NM, arrivees contre departs ===")
for lab, key in (("par aeroport-jour", ["AIRPORT", "day"]),
                 ("par aeroport-heure", ["AIRPORT", "hour"])):
    a = arr.groupby(key, observed=True)["err"].agg(["mean", "std", "size"]).rename(
        columns={"mean": "a_mean", "std": "a_std", "size": "a_n"})
    d = dep.groupby(key, observed=True)["err"].agg(["mean", "std", "size"]).rename(
        columns={"mean": "d_mean", "std": "d_std", "size": "d_n"})
    j = a.join(d, how="inner")
    j = j[(j["a_n"] >= 20) & (j["d_n"] >= 20)]
    if len(j) < 100:
        continue
    # remove the airport's own level, so we measure co-movement not cross-airport level
    for c in ("a_mean", "d_mean", "a_std", "d_std"):
        j[c + "_c"] = j[c] - j.groupby(level=0, observed=True)[c].transform("mean")
    print("  %-20s n=%6d | corr(moyennes)=%+.3f | corr(ecarts-types)=%+.3f" % (
        lab, len(j),
        np.corrcoef(j["a_mean_c"], j["d_mean_c"])[0, 1],
        np.corrcoef(j["a_std_c"], j["d_std_c"])[0, 1]))

print("\n=== par operateur : NM se trompe-t-il de la meme facon aux deux bouts ? ===")
a = arr.groupby("AIRCRAFT_OPERATOR_flt", observed=True)["err"].agg(["mean", "size"])
d = dep.groupby("AIRCRAFT_OPERATOR_flt", observed=True)["err"].agg(["mean", "size"])
j = a.join(d, how="inner", lsuffix="_a", rsuffix="_d")
j = j[(j["size_a"] >= 200) & (j["size_d"] >= 200)]
print("  %d operateurs | corr = %+.3f" % (len(j), np.corrcoef(j["mean_a"], j["mean_d"])[0, 1]))
print("  dispersion des moyennes: arrivees %.0f s, departs %.0f s" % (
    j["mean_a"].std(), j["mean_d"].std()))

print("\n=== ce que vaudrait le signal : l erreur NM des departs est-elle predictible ? ===")
# how much of the departure-side error variance sits between airport-days at all?
g = dep.groupby(["AIRPORT", "day"], observed=True)["err"]
mu, n = g.transform("mean"), g.transform("size")
between = np.sqrt(np.mean((mu[n >= 20] - dep["err"][n >= 20].mean()) ** 2))
print("  ecart-type total de l erreur NM (departs)      : %.0f s" % dep["err"].std())
print("  part expliquee par le niveau aeroport-jour     : %.0f s (%.1f%% de la variance)" % (
    between, 100 * (between / dep["err"].std()) ** 2))
