"""Off-block from ADS-B surface tracks: stand positions, matching, and estimation.

Three things had to be right before this worked, and each was wrong in turn.

**The stand's position, not the aircraft's speed.** The first attempt read off-block as
the first point above a groundspeed threshold and showed a +483 s median bias. A speed
threshold cannot tell a pushback from a hold at a taxiway intersection. But in 2025 we
know the true off-block, so every point an aircraft reports in the quarter-hour before it
is, by construction, a point at its stand; collecting those gives each (airport, stand) an
empirical position. Leaving the stand is then a geometric fact. No external source is
needed, and unlike OpenStreetMap there is no name to match and no polygon centroid
standing in for where aircraft actually park.

**Matching on any callsign the track ever carries.** The callsign rides on roughly a
quarter of points, so it has to be propagated - but propagating the *modal* one per
aircraft and airport silently discards the 60 % of airframes that arrive under one
callsign and depart under another. That one rule cost 2.75x coverage: 16.1 % of
departures matched instead of 44.3 %. So the callsign resolves to an `icao24`, and the
aircraft's whole track at that airport is then available whatever the individual points
say.

**A constant bias is not an error.** Once the above were fixed the estimate still ran
+274 s late, which looked like failure until the spread was read separately: sd 121 s
overall and 36 s at Zurich, against 355 s for Network Manager's own `AOBT_3`. The lateness
is mechanical - the aircraft is declared gone once it is 320 m out, which takes minutes -
so three candidate timestamps are emitted with the evidence behind each, and the model
removes what is a per-airport offset.

Coverage is the binding constraint, and it is uneven: EHAM 97 %, LIRF 90 %, LSZH 88 %,
EDDF 82 %, while LFPG reports 1.7 % and LTFM nothing at all - Istanbul has no surface
coverage in either year, which puts 13.8 % of the test out of reach by this route.
"""
import json
import os
import re

import numpy as np
import pandas as pd

_EPOCH = pd.Timestamp("1970-01-01", tz="UTC")
_FLIGHT_RE = re.compile(r"^([A-Z0-9]{2,3}?)(\d.*)$")
SURF_KT = 60.0        # inside an airport box, slower than this is on the surface
PRE_LO, PRE_HI = 900.0, 120.0   # window before true off-block, certainly at the stand
STAND_R, AWAY_R = 180.0, 320.0  # metres: at the stand / demonstrably gone
WINDOW_S = 7200.0     # how far back from takeoff to look
MAX_GAP_S = 900.0     # a wider hole between the stand and the first sighting away means
                      # the aircraft was lost, so the at-stand point is unrelated
TAXI_LO, TAXI_HI = 60.0, 7200.0   # outside this the estimate is not a taxi at all
MAX_BACK_S = 240.0    # the extrapolation may not reach further back than this: STAND_R is
                      # 180 m and even a crawling pushback covers it inside four minutes
MIN_FLIGHTS = int(os.environ.get("ADSB_MIN_FLIGHTS", 3))   # flights needed per stand

FEATURES = ["adsb_taxi_extrap", "adsb_taxi_last", "adsb_taxi_away", "adsb_n_stand",
            "adsb_span_s", "adsb_gap_s", "adsb_dist_last", "adsb_dist_away",
            "adsb_stand_spread", "adsb_stand_n"]


def epoch_s(s):
    """Seconds since the epoch whatever the column's resolution. The official parquet
    carries datetime64[us], so astype("int64") would give microseconds, not nanoseconds -
    a silent factor of a thousand."""
    return ((s - _EPOCH) / pd.Timedelta(seconds=1)).to_numpy(dtype="float64")


def metres(lat0, lon0, lat, lon):
    """Flat local approximation, ample over an airport."""
    return np.hypot((lat - lat0) * 111320.0,
                    (lon - lon0) * 111320.0 * np.cos(np.radians(lat0)))


def load_day(path):
    """Surface points of one day, indexed by (airport, icao24), with a callsign lookup.

    Returns (tracks, lookup): tracks maps (airport, icao24) -> sorted frame, lookup maps
    (airport, callsign) -> icao24 for every callsign the day's points ever carry.
    """
    d = pd.read_parquet(path)
    # not on_ground alone: the flag comes from the transponder and many stay silent -
    # LFPG sets it on 3 % of its points. Inside an airport box anything slower than
    # SURF_KT is on the surface, jets stalling far above it, and the stand test is
    # geometric regardless.
    gs = d["gs"].to_numpy(dtype="float64")
    d = d[(d["on_ground"].to_numpy() == 1) | (gs < SURF_KT)].copy()
    d["cs"] = d["callsign"].astype(str).str.strip().str.upper()
    named = d[d["cs"].str.len() > 0]
    lookup = {}
    if len(named):
        c = (named.groupby(["airport", "cs", "icao24"], observed=True).size()
             .reset_index(name="n").sort_values("n", ascending=False)
             .drop_duplicates(["airport", "cs"]))
        lookup = dict(zip(zip(c["airport"], c["cs"]), c["icao24"]))
    d["t"] = epoch_s(d["ts"])
    d = d.sort_values("t")
    tracks = {k: v for k, v in d.groupby(["airport", "icao24"], observed=True)}
    return tracks, lookup


def track_for(tracks, lookup, airport, *callsigns):
    """The first of several keys that resolves. A flight has up to three: Network
    Manager's ICAO callsign, the airport's own IATA flight number, and that number
    translated to ICAO. The first exists only when NM matched the flight - which is
    exactly the 1.5 % of departures that carry most of the error - so the other two are
    what reach them: they lift the match rate on NM-less flights from 0 % to 59 %."""
    for cs in callsigns:
        if not cs:
            continue
        ic = lookup.get((airport, cs))
        if ic is not None:
            g = tracks.get((airport, ic))
            if g is not None:
                return g
    return None


def iata_to_icao(flight, table):
    """LH946 -> DLH946, from a prefix table derived from the flights where both forms
    are present. 1,095 prefixes come out of 1.63 M matched departures, so no external
    airline-code list is needed."""
    m = _FLIGHT_RE.match(flight or "")
    if not m:
        return ""
    pre = table.get(m.group(1))
    return (pre + m.group(2)) if pre else ""


def stand_points(g, B):
    """Median position over the points known to precede a true off-block B."""
    t = g["t"].to_numpy()
    s = (t >= B - PRE_LO) & (t <= B - PRE_HI)
    if s.sum() < 3:
        return None
    return (float(np.median(g["lat"].to_numpy()[s])),
            float(np.median(g["lon"].to_numpy()[s])))


def estimate(g, T, s_lat, s_lon):
    """Three off-block candidates for one departure, with the evidence behind them.

    `last` is the final sighting within STAND_R, `away` the first beyond AWAY_R, and
    `extrap` carries the radial motion between them back to the stand itself - which is
    what removes most of the mechanical lateness, since the aircraft is only declared
    gone once it is 320 m out. Returns None when the stand was never observed, rather
    than a guess: that distinction is the whole difference between a +483 s bias and a
    +20 s one.
    """
    t = g["t"].to_numpy()
    sel = (t >= T - WINDOW_S) & (t <= T)
    if sel.sum() < 4:
        return None
    tt = t[sel]
    dist = metres(s_lat, s_lon, g["lat"].to_numpy()[sel], g["lon"].to_numpy()[sel])
    at = dist <= STAND_R
    if not at.any():
        return None
    last = int(np.flatnonzero(at)[-1])
    j = last
    while j > 0 and at[j - 1]:
        j -= 1
    away = np.flatnonzero((np.arange(len(tt)) > last) & (dist > AWAY_R))
    if not len(away):
        return None
    k = int(away[0])
    d_last, d_away = float(dist[last]), float(dist[k])
    t_last, t_away = float(tt[last]), float(tt[k])
    if t_away - t_last > MAX_GAP_S:
        return None
    if not (TAXI_LO <= T - t_away and T - t_last <= TAXI_HI):
        return None
    speed = (d_away - d_last) / max(t_away - t_last, 1.0)   # metres per second, radial
    back = min(d_last / speed, MAX_BACK_S) if speed > 0.2 else 0.0
    extrap = t_last - back
    return dict(adsb_taxi_extrap=T - extrap, adsb_taxi_last=T - t_last,
                adsb_taxi_away=T - t_away, adsb_n_stand=int(last - j + 1),
                adsb_span_s=t_last - float(tt[j]), adsb_gap_s=t_away - t_last,
                adsb_dist_last=d_last, adsb_dist_away=d_away)
