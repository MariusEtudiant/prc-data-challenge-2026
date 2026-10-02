"""M63: how much of the test sits on disrupted days, and does 2025 contain
comparable days we can learn from?

A disrupted day = a day whose arrival taxi-in P90 at that airport is far above
the airport's own normal. Arrivals are labelled in both sets, so the measure is
computed identically on train and test.
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from dc26.data import load_movements_2025, load_ranking_2026

mv, rk = load_movements_2025(), load_ranking_2026()


def day_stats(df):
    a = df[df["PHASE_mvt"] == "ARR"].copy()
    a["day"] = a["MVT_TIME_UTC_mvt"].dt.date
    t = a["TAXITIME_SEC_mvt"].astype(float)
    a = a[np.isfinite(t) & (t > 0)]
    g = a.groupby(["AIRPORT", "day"], observed=True)["TAXITIME_SEC_mvt"]
    return g.quantile(0.90).rename("p90").reset_index()


s25, s26 = day_stats(mv), day_stats(rk)
base = s25.groupby("AIRPORT", observed=True)["p90"].median().rename("normal")
s25 = s25.merge(base, on="AIRPORT"); s26 = s26.merge(base, on="AIRPORT")
s25["ratio"] = s25["p90"] / s25["normal"]; s26["ratio"] = s26["p90"] / s26["normal"]

dep = rk[rk["PHASE_mvt"] == "DEP"].copy()
dep["day"] = dep["MVT_TIME_UTC_mvt"].dt.date
d = dep.merge(s26[["AIRPORT", "day", "ratio"]], on=["AIRPORT", "day"], how="left")

print("=== jours perturbes : combien de departs du test, et 2025 en contient-il autant ? ===")
print("  %-24s %12s %10s %14s" % ("severite (p90 / normal)", "departs test", "part", "jours 2025"))
for lo, hi, lab in [(1.5, 2.0, "1,5x a 2x"), (2.0, 3.0, "2x a 3x"), (3.0, 99, "au-dela de 3x")]:
    m = (d["ratio"] >= lo) & (d["ratio"] < hi)
    n25 = int(((s25["ratio"] >= lo) & (s25["ratio"] < hi)).sum())
    print("  %-24s %12d %9.1f%% %14d" % (lab, m.sum(), 100 * m.mean(), n25))
print("  %-24s %12d %9.1f%% %14d" % ("total >= 1,5x", (d["ratio"] >= 1.5).sum(),
                                     100 * (d["ratio"] >= 1.5).mean(), int((s25["ratio"] >= 1.5).sum())))

print("\n=== les 8 journees les plus perturbees du jeu de test ===")
top = s26.sort_values("ratio", ascending=False).head(8)
for _, r in top.iterrows():
    nd = int(((dep["AIRPORT"] == r["AIRPORT"]) & (dep["day"] == r["day"])).sum())
    worse = int((s25[s25["AIRPORT"] == r["AIRPORT"]]["ratio"] >= r["ratio"]).sum())
    print("  %-6s %s  p90=%6.0fs (%.2fx normal) | %4d departs a predire | %d jours 2025 aussi durs" % (
        r["AIRPORT"], r["day"], r["p90"], r["ratio"], nd, worse))
