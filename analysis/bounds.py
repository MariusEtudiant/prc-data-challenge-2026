"""Z01: recovering a LOBT for the flights with no record, and capping the long predictions.

Two bounds, both from how the data was assembled rather than from anything learned.

**The orphan window.** A departure whose `FLIGHT_ID_mvt` is null has no LOBT, so the window
that constrains every other prediction does not reach it - and those flights are 1.5 % of
rows and 40 % of our squared error. But `LOBT_flt` belongs to the *flight*, not to the
movement: the same aircraft's arrival at its destination carries the same LOBT, and when
that destination is one of the ten airports the arrival is in our data, with its Network
Manager record intact. Matching a departure to its own arrival therefore recovers a LOBT
for flights that appear to have none.

**The cap at d.** With d = MVT - SCHED, the target y = MVT - BLOCK satisfies y <= d whenever
the aircraft pushed back at or after its scheduled time. That is not universal - an early
pushback gives y > d - so the cap is only applied above a threshold, where a prediction that
large already means the model thinks the flight was held. Where the threshold should sit is
measured here rather than assumed.

Both are checked for validity on 2025 before being measured as gains, because a bound that is
violated even rarely is worse than no bound: it would clip a correct prediction.

Usage: python scripts/exp_z01_orphans.py [pred.npy]
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
PROC, EXP = ROOT / "data" / "processed", ROOT / "experiments"
from dc26.validation import rmse
import make_submission as V

PRED = sys.argv[1] if len(sys.argv) > 1 else "v17_mirror.npy"
CAP = 86400
COLS = ["AIRPORT", "PHASE_mvt", "ADEP_mvt", "ADES_mvt", "FLIGHT_mvt", "FLIGHT_ID_mvt",
        "MVT_TIME_UTC_mvt", "BLOCK_TIME_UTC_mvt", "SCHED_TIME_UTC_mvt", "LOBT_flt"]


def recover_lobt(mv):
    """A LOBT for the departures that have none, taken from their own arrival.

    The key is the flight itself - origin, destination and the airport's own flight number -
    plus the day, since a route is flown at most a couple of times a day and the arrival
    follows its departure within hours. Only departures with no NM record are filled, and
    only from arrivals that have one."""
    sec = lambda s: ((s - pd.Timestamp(0, tz="UTC")) / pd.Timedelta(seconds=1)).to_numpy()
    dep = mv[mv["PHASE_mvt"] == "DEP"].reset_index(drop=True)
    arr = mv[(mv["PHASE_mvt"] == "ARR") & mv["LOBT_flt"].notna()].copy()
    key = lambda f: (f["ADEP_mvt"].astype(str) + "|" + f["ADES_mvt"].astype(str) + "|"
                     + f["FLIGHT_mvt"].astype(str).str.upper().str.strip())
    arr["k"] = key(arr)
    arr["lobt"] = sec(arr["LOBT_flt"])
    arr["t"] = sec(arr["MVT_TIME_UTC_mvt"])
    dep = dep.assign(k=key(dep), t=sec(dep["MVT_TIME_UTC_mvt"]))
    a = arr.sort_values("t")[["k", "t", "lobt"]]
    d = dep[["k", "t"]].copy()
    d["_i"] = np.arange(len(d))
    # the arrival of a departure lands after it and within the day
    j = pd.merge_asof(d.sort_values("t"), a, on="t", by="k", direction="forward",
                      tolerance=20 * 3600)
    j = j.sort_values("_i")
    return sec(dep["LOBT_flt"]), j["lobt"].to_numpy(), dep


mv = pd.read_parquet(PROC / "movements_2025.parquet", columns=COLS)
own, rec, dep = recover_lobt(mv)
sec = lambda s: ((s - pd.Timestamp(0, tz="UTC")) / pd.Timedelta(seconds=1)).to_numpy()
mvt, blk, sch = sec(dep["MVT_TIME_UTC_mvt"]), sec(dep["BLOCK_TIME_UTC_mvt"]), sec(dep["SCHED_TIME_UTC_mvt"])
y = mvt - blk
d = mvt - sch
nonm = dep["FLIGHT_ID_mvt"].isna().to_numpy()
valid = (y > 0) & (y < 200000)

print("=== la fenetre orphelins ===")
print("  departs sans enregistrement NM : %d (%.2f%%)" % (nonm.sum(), 100 * nonm.mean()))
print("  dont une destination parmi les dix: %.1f%%" % (
    100 * dep.loc[nonm, "ADES_mvt"].isin(sorted(set(dep["ADEP_mvt"]))).mean()))
got = nonm & np.isfinite(rec) & ~np.isfinite(own)
print("  LOBT recupere via l'arrivee     : %d (%.1f%% des sans-NM)" % (got.sum(), 100 * got.sum() / max(nonm.sum(), 1)))
if got.sum():
    e = np.abs(blk[got & valid] - rec[got & valid])
    print("\n  le LOBT recupere borne-t-il vraiment l'off-block ?")
    print("    |BLOCK - LOBT recupere| : median %.0f s | p90 %.0f | p99 %.0f | max %.0f" % (
        np.median(e), np.percentile(e, 90), np.percentile(e, 99), e.max()))
    for lim in (3606, 3630, 7200):
        print("    <= %5d s : %.3f%% (%d violations sur %d)" % (
            lim, 100 * (e <= lim).mean(), (e > lim).sum(), len(e)))
    # controle: la meme recuperation sur les vols QUI ont deja un LOBT doit le retrouver
    both = np.isfinite(own) & np.isfinite(rec) & valid
    dd = np.abs(own[both] - rec[both])
    print("\n  controle sur les %d vols qui ont les deux: identiques a 1 s pres %.2f%%, median %.0f s" % (
        both.sum(), 100 * (dd <= 1).mean(), np.median(dd)))

print("\n=== le plafond a d ===")
v = valid
print("  y > d (poussee avant l'horaire) : %.2f%% de tous les vols" % (100 * (y[v] > d[v]).mean()))
print("  %-16s %10s %12s %14s" % ("si y predit >", "n reels", "dont y > d", "marge mediane"))
for th in (1800, 3600, 7200, 10800, 14400):
    m = v & (y > th)
    print("  %-16s %10d %11.2f%% %14.0f" % ("%d s" % th, m.sum(), 100 * (y[m] > d[m]).mean(),
                                            np.median(d[m] - y[m])))

pred = np.load(EXP / PRED)
meta = pd.read_parquet(PROC / "meta_train2025.parquet")
yy = meta["y"].astype("float64").to_numpy()
g = np.isfinite(pred) & (yy > 0) & (yy < 200000)
assert len(pred) == len(y), "%d != %d" % (len(pred), len(y))
r0 = rmse(yy[g], pred[g])
print("\n=== effet sur %s (metrique BRUTE), depart %.2f ===" % (PRED, r0))

lo = np.where(np.isfinite(own), own, rec)
w_lo, w_hi = mvt - lo - V.LOBT_W, mvt - lo + V.LOBT_W
new = pred.copy()
k = g & np.isfinite(w_lo) & ~np.isfinite(own)          # seulement les orphelins recuperes
new[k] = np.clip(pred[k], np.maximum(w_lo[k], 60), w_hi[k])
moved = k & (np.abs(new - pred) > 1)
print("  fenetre orphelins : %d lignes deplacees -> %.2f (%+.2f)" % (
    moved.sum(), rmse(yy[g], new[g]), rmse(yy[g], new[g]) - r0))

print("\n  %-22s %10s %10s %9s" % ("plafond a d au-dela de", "deplacees", "RMSE", "gain"))
for th in (3600, 7200, 10800, 14400, 21600):
    c = pred.copy()
    m = g & (pred > th) & np.isfinite(d) & (d > 60)
    c[m] = np.minimum(pred[m], np.maximum(d[m], 60))
    print("  %-22s %10d %10.2f %+9.2f" % ("%d s" % th, (m & (c != pred)).sum(),
                                          rmse(yy[g], c[g]), rmse(yy[g], c[g]) - r0))

best_th = 7200
both_v = new.copy()
m = g & (both_v > best_th) & np.isfinite(d) & (d > 60)
both_v[m] = np.minimum(both_v[m], np.maximum(d[m], 60))
print("\n  les deux ensemble (plafond a %d s) : %.2f (%+.2f)" % (
    best_th, rmse(yy[g], both_v[g]), rmse(yy[g], both_v[g]) - r0))
np.save(EXP / "z01_bounds.npy", both_v)
