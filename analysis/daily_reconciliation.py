"""Y02: the daily constraint again, applied only where it bites and in proportion to doubt.

The first attempt moved every prediction of the day by one constant and gained 1.09 s. Two
things were wrong with it. The bisection guard assumed a decreasing function, which is true
of a shift and false of a scale, so the scale variant silently did nothing at every single
day. And the shift moved flights whose predicted off-block is *before* their schedule, which
contribute nothing to the published total: they absorbed error without helping the
constraint.

So the correction now reaches only the flights where the constraint is active, and it is
spread in proportion to how uncertain each of them is rather than equally. Under a normal
error model, observing a linear functional of the targets moves each prediction by an amount
proportional to its own variance; a flight the model is sure about should not be asked to
account for a day's missing minutes. Three proxies for that variance are tried, including a
flat one so the added complexity has to earn itself.

Every variant satisfies the published total exactly by construction, so what is being
compared is only how the correction is distributed.

Usage: python scripts/exp_y02_daily.py [pred.npy]
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
PROC, EXP = ROOT / "data" / "processed", ROOT / "experiments"
PRU = ROOT / "data" / "external" / "pru"
from dc26.validation import rmse
import make_submission as V

PRED = sys.argv[1] if len(sys.argv) > 1 else "v17_mirror.npy"
CAP = 86400

pred = np.load(EXP / PRED)
meta = pd.read_parquet(PROC / "meta_train2025.parquet")
y = meta["y"].astype("float64").to_numpy()
ck = V.clock("movements_2025.parquet")
mv = pd.read_parquet(PROC / "movements_2025.parquet", columns=["PHASE_mvt", "MVT_TIME_UTC_mvt"])
date = mv[mv["PHASE_mvt"] == "DEP"]["MVT_TIME_UTC_mvt"].dt.date.to_numpy()
d = ck["mvt"] - ck["sch"]
g = np.isfinite(pred) & (y > 0) & (y < 200000)

p = pd.read_csv(PRU / "all_pre_departure_delays_2025.csv")
p["date"] = pd.to_datetime(p["FLT_DATE"]).dt.date
T = p.set_index(["APT_ICAO", "date"])[["DLY_ALL_PRE_2", "FLT_DEP_1"]]

# three guesses at how much each flight's prediction should be allowed to move
W = {
    "plat": np.ones(len(y)),
    "sans_NM": np.where(ck["nonm"], 6.0, 1.0),
    "retard": 1.0 + np.clip(d, 0, 7200) / 1800.0,
}
OUT = {k: pred.copy() for k in W}


def shift_active(dd, w, target):
    """One scalar s such that sum of max(0, dd - s*w) equals the target, with dd the
    predicted gate delay and w the per-flight share. Decreasing in s, so bisect."""
    f = lambda s: np.maximum(dd - s * w, 0).sum()
    lo, hi = -2000.0, 2000.0
    if f(lo) < target or f(hi) > target:
        return None
    for _ in range(60):
        m = 0.5 * (lo + hi)
        if f(m) > target:
            lo = m
        else:
            hi = m
    return 0.5 * (lo + hi)


df = pd.DataFrame({"i": np.arange(len(y))[g], "apt": ck["apt"][g], "date": date[g]})
rows = []
for (a, dt), sub in df.groupby(["apt", "date"], sort=False):
    try:
        r = T.loc[(a, dt)]
    except KeyError:
        continue
    i = sub["i"].to_numpy()
    tgt = float(r["DLY_ALL_PRE_2"]) * 60.0
    dd = d[i] - pred[i]
    rows.append((a, dt, len(i), int(r["FLT_DEP_1"]), tgt, float(np.maximum(dd, 0).sum())))
    for k, w in W.items():
        s = shift_active(dd, w[i], tgt)
        if s is None:
            continue
        # only the flights whose predicted off-block is after their schedule move: the
        # others contribute nothing to the published total
        OUT[k][i] = np.clip(np.where(dd > 0, d[i] - np.maximum(dd - s * w[i], 0), pred[i]),
                            60, CAP)

S = pd.DataFrame(rows, columns=["apt", "date", "n", "n_pru", "cible", "notre"])
print("%s | %d jours reconcilies" % (PRED, len(S)))
print("  notre total contre le publie: ecart median %+.2f%% | p10 %+.1f%% | p90 %+.1f%%" % (
    100 * ((S["notre"] / S["cible"]) - 1).median(),
    100 * ((S["notre"] / S["cible"]) - 1).quantile(.1),
    100 * ((S["notre"] / S["cible"]) - 1).quantile(.9)))

r0 = rmse(y[g], pred[g])
print("\n=== metrique BRUTE ===")
print("  %-28s %10.2f" % ("avant", r0))
best, br = None, r0
for k in W:
    r = rmse(y[g], OUT[k][g])
    print("  %-28s %10.2f %+10.2f" % ("poids " + k, r, r - r0))
    if r < br:
        best, br = k, r
if best is None:
    print("\naucune variante n'ameliore")
    sys.exit()

print("\n=== %s, par aeroport ===" % best)
print("  %-7s %9s %11s %10s %9s %10s" % ("aero", "n", "avant", "apres", "gain", "comptes"))
for a in sorted(set(ck["apt"])):
    m = g & (ck["apt"] == a)
    if m.sum() < 500:
        continue
    s = S[S["apt"] == a]
    print("  %-7s %9d %11.1f %10.1f %+9.1f %10.3f" % (
        a, m.sum(), rmse(y[m], pred[m]), rmse(y[m], OUT[best][m]),
        rmse(y[m], OUT[best][m]) - rmse(y[m], pred[m]),
        (s["n_pru"] / s["n"]).median() if len(s) else np.nan))

print("\n=== et en ne retenant que les jours ou les comptes concordent ===")
for tol in (0.02, 0.01, 0.005):
    ok = S[(S["n_pru"] / S["n"] - 1).abs() < tol]
    keep = set(zip(ok["apt"], ok["date"]))
    mask = np.array([(a, t) in keep for a, t in zip(ck["apt"], date)])
    v = pred.copy()
    v[mask] = OUT[best][mask]
    print("  tolerance %4.1f%% : %3d jours sur %d | RMSE %.2f (%+.2f)" % (
        100 * tol, len(ok), len(S), rmse(y[g], v[g]), rmse(y[g], v[g]) - r0))
np.save(EXP / "y02_daily.npy", OUT[best])
print("\nenregistre: experiments/y02_daily.npy (%s)" % best)
