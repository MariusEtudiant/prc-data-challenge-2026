"""N02: the stand-occupancy constraint we never used.

We predict off-block, and our dominant input for it (AOBT_3) is wrong by more than a
minute four times out of five - N01b showed Network Manager is precise on radar-
observed events (landing, sd 57s) and six times worse on the one nobody observes
(pushback, sd 355s). So the noise is in the FEATURE, not the target, and any exact
constraint on off-block is worth more than another tree.

turnaround.py links each departure to the PREVIOUS arrival at its stand, giving an
upper bound on taxi. The symmetric constraint is unused: another aircraft can only
block in at that stand AFTER ours has vacated it, so

    BLOCK_out < T(next arrival at same stand)
    => taxi = MVT - BLOCK_out  >  MVT - T(next arrival)

which is a LOWER bound, exact, and computable from data the ranking file keeps.
Question here is only whether it BINDS often enough to matter.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
PROC = ROOT / "data" / "processed"

_EPOCH = pd.Timestamp(0, tz="UTC")


def sec(s):
    return ((s - _EPOCH) / pd.Timedelta(seconds=1)).to_numpy(dtype=float)


def build(mv, tag):
    dep = mv[mv["PHASE_mvt"] == "DEP"].copy()
    arr = mv[mv["PHASE_mvt"] == "ARR"]
    lo = np.full(len(dep), np.nan)
    gap = np.full(len(dep), np.nan)
    key_d = dep["AIRPORT"].astype(str) + "|" + dep["STAND_mvt"].astype(str)
    key_a = arr["AIRPORT"].astype(str) + "|" + arr["STAND_mvt"].astype(str)
    t_dep = sec(dep["MVT_TIME_UTC_mvt"])
    # arrival BLOCK_TIME = when it actually occupied the stand
    t_arr = sec(arr["BLOCK_TIME_UTC_mvt"])
    ok_a = np.isfinite(t_arr)
    idx = {}
    for k, g in pd.Series(np.arange(len(arr))[ok_a]).groupby(key_a.to_numpy()[ok_a]):
        v = np.sort(t_arr[g.to_numpy()])
        idx[k] = v
    kd = key_d.to_numpy()
    for i, (k, td) in enumerate(zip(kd, t_dep)):
        v = idx.get(k)
        if v is None or not np.isfinite(td):
            continue
        j = np.searchsorted(v, td, side="left")
        if j < len(v):
            lo[i] = td - v[j]          # negative unless the next arrival precedes takeoff
            gap[i] = v[j] - td
    return dep, lo, gap


mv = pd.read_parquet(PROC / "movements_2025.parquet",
                     columns=["AIRPORT", "PHASE_mvt", "STAND_mvt", "MVT_TIME_UTC_mvt",
                              "BLOCK_TIME_UTC_mvt", "TAXITIME_SEC_mvt"])
dep, lo, gap = build(mv, "2025")
y = dep["TAXITIME_SEC_mvt"].astype(float).to_numpy()
ok = np.isfinite(y) & (y > 0) & (y < 7200) & np.isfinite(lo)
print("couverture de la contrainte : %.1f%% des departs" % (100 * np.isfinite(lo).mean()))
print("dont la borne est active (arrivee suivante APRES notre decollage) : %.1f%%" % (
    100 * (gap[np.isfinite(gap)] > 0).mean()))

# the bound is informative when the next arrival lands soon after our takeoff:
# taxi must exceed MVT - T(next arrival), which is negative when that arrival is later
b = -lo  # = T(next arrival) - MVT ; taxi > -b  is vacuous when b > 0
print("\n=== la borne inferieure est-elle contraignante ? ===")
print("  %-28s %9s %10s %12s %12s" % ("cas", "n", "part", "taxi median", "borne median"))
for lab, m in [("arrivee suivante < 30 min", ok & (gap < 1800) & (gap > -86400)),
               ("arrivee suivante < 15 min", ok & (gap < 900) & (gap > -86400)),
               ("arrivee suivante < 5 min", ok & (gap < 300) & (gap > -86400)),
               ("arrivee AVANT le decollage", ok & (gap < 0))]:
    if m.sum() < 100:
        continue
    print("  %-28s %9d %9.2f%% %12.0f %12.0f" % (
        lab, m.sum(), 100 * m.sum() / ok.sum(), np.median(y[m]), np.median(-gap[m])))

m = ok & (gap < 0)
if m.sum() > 100:
    viol = (y[m] <= -gap[m]).mean()
    print("\n  quand l arrivee suivante precede notre decollage, la borne taxi > %s tient dans %.2f%% des cas" % (
        "MVT - T_arr", 100 * (1 - viol)))
    print("  ecart median entre le taxi reel et la borne : %.0f s" % np.median(y[m] + gap[m]))

# how much does the bound add over what we already predict?
print("\n=== correlation de la borne avec la cible, la ou elle est definie ===")
for lab, m in [("tous", ok), ("arrivee suivante < 1h", ok & (gap < 3600))]:
    if m.sum() < 500:
        continue
    print("  %-24s n=%8d  corr(-gap, taxi) = %+.3f" % (lab, m.sum(), np.corrcoef(-gap[m], y[m])[0, 1]))
