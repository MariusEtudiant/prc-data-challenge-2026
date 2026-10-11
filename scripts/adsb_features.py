"""ADS-B features per departure: when the aircraft woke, moved, rolled and left.

Not one estimator but the raw evidence, because no single reading of a track is right
everywhere. An aircraft parked at a gate has its transponder off and it comes alive at
pushback, so the first sighting after a silence is the off-block - but only where the
receivers hear the apron. Where they do not, the aircraft is first heard well into its taxi,
and where the transponder stays on while parked there is no silence to find at all. Those
three regimes carry latencies of +240 s, of tens of minutes, and of nothing usable, and which
one a flight is in is itself observable: the length of the silence, how many points precede
it, how far the first point sits from the stand.

So each flight gets the timestamps and the evidence behind them, and a corrector trained on
out-of-fold predictions decides what to trust. That is the shape both GREKI and Vudueprajacu
described - correct a strong base model rather than predict from scratch - and it is the only
shape that survives a feature which exists on a fifth of the training days.

One deliberate choice about drift. GREKI found that adsb.lol's receiver network changed
between 2025 and 2026, so features encoding *where* an aircraft was heard did not transfer
while features describing its *movement* did. Everything here is a duration or a speed
transition except `d_first`, which is kept only because it separates the three regimes, and is
reported separately so its drift can be checked before it is used.

Usage: python scripts/adsb_features.py 2025    (or 2026)
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from dc26 import adsb as AD
import adsb_build as B

ADSB, EXT = ROOT / "data" / "adsb", ROOT / "data" / "external"
EXT.mkdir(parents=True, exist_ok=True)
WINDOW = 10800.0      # three hours back from takeoff
GAP = 900.0           # a silence this long ends a parking
MOVE, ROLL, FAST = 3.0, 30.0, 80.0    # knots: moving, taxiing properly, taking off

COLS = ["adsb_taxi_wake", "adsb_taxi_move", "adsb_taxi_roll", "adsb_gap_s", "adsb_n",
        "adsb_n_before", "adsb_takeoff_err", "adsb_d_first", "adsb_first_gs",
        "adsb_quiet_frac", "adsb_key"]


def load_day(day):
    d = pd.read_parquet(ADSB / ("ground_%s.parquet" % day))
    d["cs"] = d["callsign"].astype(str).str.strip().str.upper()
    named = d[d["cs"].str.len() > 0]
    lookup = {}
    if len(named):
        c = (named.groupby(["airport", "cs", "icao24"], observed=True).size()
             .reset_index(name="n").sort_values("n", ascending=False)
             .drop_duplicates(["airport", "cs"]))
        lookup = dict(zip(zip(c["airport"], c["cs"]), c["icao24"]))
    d["t"] = AD.epoch_s(d["ts"])
    return {k: v for k, v in d.sort_values("t").groupby(["airport", "icao24"],
                                                        observed=True)}, lookup


def features(g, T, stand):
    t = g["t"].to_numpy()
    sel = (t >= T - WINDOW) & (t <= T + 120)
    if sel.sum() < 2:
        return None
    tt = t[sel]
    v = np.nan_to_num(g["gs"].to_numpy(dtype=float)[sel], nan=0.0)
    lat, lon = g["lat"].to_numpy()[sel], g["lon"].to_numpy()[sel]
    d = np.diff(tt)
    big = np.flatnonzero(d > GAP)
    # the departure leg begins after the last long silence; with none, at the window's start
    i = int(big[-1]) + 1 if len(big) else 0
    gap = float(d[big[-1]]) if len(big) else np.nan
    out = dict(adsb_taxi_wake=T - float(tt[i]), adsb_gap_s=gap,
               adsb_n=float(len(tt) - i), adsb_n_before=float(i),
               adsb_first_gs=float(v[i]),
               adsb_quiet_frac=float(np.mean(v[i:] < MOVE)) if len(tt) > i else np.nan)
    for name, lim in (("adsb_taxi_move", MOVE), ("adsb_taxi_roll", ROLL)):
        k = np.flatnonzero((np.arange(len(tt)) >= i) & (v > lim))
        out[name] = T - float(tt[k[0]]) if len(k) else np.nan
    # the takeoff roll is observable, and the given MVT_TIME should land on it: how far it
    # misses says whether this track belongs to this flight at all
    k = np.flatnonzero((np.arange(len(tt)) >= i) & (v > FAST))
    out["adsb_takeoff_err"] = float(tt[k[0]]) - T if len(k) else np.nan
    out["adsb_d_first"] = (AD.metres(stand[0], stand[1], lat[i], lon[i])
                           if stand is not None else np.nan)
    return out


def run(year):
    dep = B.departures(year)
    days = B.days(year)
    if not days:
        print("aucun jour extrait pour %d" % year)
        return
    sf = EXT / "adsb_stands.parquet"
    S = pd.read_parquet(sf).set_index("stand") if sf.exists() else None
    res = []
    for day in days:
        sub = dep[dep["day"] == day]
        if not len(sub):
            continue
        tr, lk = load_day(day)
        for pos, a, cs, fm, fi, st, T in zip(sub["pos"], sub["AIRPORT"].astype(str),
                                             sub["cs"], sub["fm"], sub["fm_icao"],
                                             sub["stand"], sub["T"]):
            g = AD.track_for(tr, lk, a, cs, fi, fm)
            if g is None:
                continue
            key = 0.0 if lk.get((a, cs)) else (1.0 if lk.get((a, fi)) else 2.0)
            stand = None
            if S is not None and st in S.index:
                r = S.loc[st]
                stand = (r["lat"], r["lon"])
            f = features(g, T, stand)
            if f is None:
                continue
            f["pos"] = int(pos)
            f["adsb_key"] = key
            res.append(f)
        print("  %s : %7d cumules" % (day, len(res)), flush=True)
    R = pd.DataFrame(res)
    R.to_parquet(EXT / ("adsb_feat_%d.parquet" % year), index=False)
    nd = dep["day"].isin(days).sum()
    print("\n%d lignes sur %d departs des jours disponibles (%.1f%%)" % (
        len(R), nd, 100 * len(R) / max(nd, 1)))
    print("\n  %-20s %9s %10s %10s %10s" % ("variable", "renseigne", "p10", "mediane", "p90"))
    for c in COLS:
        if c not in R.columns:
            continue
        s = R[c].dropna()
        if not len(s):
            continue
        print("  %-20s %8.1f%% %10.0f %10.0f %10.0f" % (
            c, 100 * R[c].notna().mean(), s.quantile(.1), s.median(), s.quantile(.9)))
    if year != 2025:
        return
    # how good is each timestamp, measured against the truth
    j = dep.set_index("pos").loc[R["pos"]]
    truth = (j["T"] - j["B"]).to_numpy()
    apt = j["AIRPORT"].astype(str).to_numpy()
    print("\n=== erreur de chaque horodatage, apres retrait d'un decalage par aeroport ===")
    print("  %-20s %9s %10s %10s" % ("variable", "n", "sd brute", "sd recalee"))
    for c in ("adsb_taxi_wake", "adsb_taxi_move", "adsb_taxi_roll"):
        v = R[c].to_numpy()
        m = np.isfinite(v)
        r = v[m] - truth[m]
        adj = r.copy()
        for a in np.unique(apt[m]):
            k = apt[m] == a
            adj[k] = r[k] - np.median(r[k])
        print("  %-20s %9d %10.0f %10.0f" % (c, m.sum(), r.std(), adj.std()))
    print("\n=== et en exigeant un silence d'au moins 30 min ===")
    ok = (R["adsb_gap_s"] >= 1800).to_numpy()
    print("  %-20s %9s %10s %10s" % ("variable", "n", "sd brute", "sd recalee"))
    for c in ("adsb_taxi_wake", "adsb_taxi_move", "adsb_taxi_roll"):
        v = R[c].to_numpy()
        m = np.isfinite(v) & ok
        if m.sum() < 50:
            continue
        r = v[m] - truth[m]
        adj = r.copy()
        for a in np.unique(apt[m]):
            k = apt[m] == a
            adj[k] = r[k] - np.median(r[k])
        print("  %-20s %9d %10.0f %10.0f" % (c, m.sum(), r.std(), adj.std()))
    print("\n=== le decollage detecte confirme-t-il MVT_TIME ? ===")
    e = R["adsb_takeoff_err"].dropna()
    print("  renseigne sur %d vols | mediane %+.0f s | a moins de 30 s: %.1f%%" % (
        len(e), e.median(), 100 * (e.abs() < 30).mean()))


if __name__ == "__main__":
    run(int(sys.argv[1]))
