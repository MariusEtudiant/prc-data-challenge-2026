"""M70: full per-airport error budget on the holdout, split between ordinary
traffic and the tail, so we can see which airport could actually hold 16 points.
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
p = np.load(EXP / "m49_blend_va.npy")
yv, apt = y[va], X["AIRPORT"].astype(str).to_numpy()[va]
art = np.zeros(len(y), bool)
for k in (1, 2, 3):
    art |= np.abs(y - k * 86400) < 3600
av = art[va]
g = np.isfinite(p) & ~av
sq = (yv - p) ** 2
tot = sq[g].sum()

# ranking-set composition, to weight each airport as the test actually does
rk = pd.read_parquet(ROOT / "data" / "ranking.parquet", columns=["ADEP_mvt", "PHASE_mvt"])
share = rk[rk["PHASE_mvt"] == "DEP"]["ADEP_mvt"].astype(str).value_counts(normalize=True)

print("=== budget d erreur par aeroport (holdout nettoye) ===")
print("  %-7s %8s %9s %9s %9s %10s %10s" % ("aero", "n", "RMSE", "ordinaire", "part err2",
                                            "poids test", "err2 test"))
rows = []
for a in sorted(set(apt)):
    m = g & (apt == a)
    if m.sum() < 100:
        continue
    ord_m = m & (yv < 3600)
    r = rmse(yv[m], p[m])
    ro = rmse(yv[ord_m], p[ord_m])
    part = sq[m].sum() / tot
    w = float(share.get(a, 0))
    rows.append((a, m.sum(), r, ro, part, w, w * r ** 2))
tw = sum(x[6] for x in rows)
for a, n, r, ro, part, w, e in sorted(rows, key=lambda x: -x[6]):
    print("  %-7s %8d %9.1f %9.1f %8.1f%% %9.1f%% %9.1f%%" % (
        a, n, r, ro, 100 * part, 100 * w, 100 * e / tw))

print("\n=== si un aeroport atteignait le niveau du meilleur comparable ===")
best_ord = min(x[3] for x in rows)
print("  meilleur RMSE ordinaire observe : %.1f" % best_ord)
cur = np.sqrt(sum(w * r ** 2 for _, _, r, _, _, w, _ in rows) / tw)
for a, n, r, ro, part, w, e in sorted(rows, key=lambda x: -x[6])[:4]:
    # what if this airport's ordinary traffic matched the best airport's?
    frac_ord = ((g & (apt == a) & (yv < 3600)).sum()) / max((g & (apt == a)).sum(), 1)
    new_r2 = r ** 2 - frac_ord * (ro ** 2 - best_ord ** 2)
    tot2 = sum(w2 * (new_r2 if a2 == a else r2 ** 2) for a2, _, r2, _, _, w2, _ in rows)
    print("  %-7s ordinaire %6.1f -> %5.1f : RMSE pondere %.1f -> %.1f (%+.1f)" % (
        a, ro, best_ord, cur, np.sqrt(tot2 / tw), np.sqrt(tot2 / tw) - cur))
