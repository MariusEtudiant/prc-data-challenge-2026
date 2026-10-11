"""Build the ADS-B off-block features: stand positions, then one row per departure.

Stage `stands` learns an empirical position for every (airport, stand) from 2025, where
the true off-block tells us which points are at the stand. Stage `offblock` uses that
table on either year and needs no truth, so the table serves the 2026 ranking set -
stands are physical places and do not move between years. Stage `split` is the honest
test: positions from the morning's flights, error measured on the afternoon's.

Usage:
  python scripts/adsb_build.py stands
  python scripts/adsb_build.py offblock 2025     # also prints the error against truth
  python scripts/adsb_build.py offblock 2026
  python scripts/adsb_build.py split 2025-07-15
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
PROC, ADSB, EXT = ROOT / "data" / "processed", ROOT / "data" / "adsb", ROOT / "data" / "external"
EXT.mkdir(parents=True, exist_ok=True)
from dc26 import adsb as AD


def days(year):
    return sorted(p.name[7:17] for p in ADSB.glob("ground_%d-*.parquet" % year))


def departures(year):
    f = "movements_2025.parquet" if year == 2025 else "ranking_2026.parquet"
    m = pd.read_parquet(PROC / f, columns=["AIRPORT", "PHASE_mvt", "STAND_mvt", "RUNWAY_mvt",
                                           "CALLSIGN_flt", "FLIGHT_mvt", "MVT_TIME_UTC_mvt",
                                           "BLOCK_TIME_UTC_mvt"])
    d = m[m["PHASE_mvt"] == "DEP"].copy()
    d["cs"] = (d["CALLSIGN_flt"].astype(str).str.strip().str.upper()
               .replace({"NAN": "", "NONE": "", "<NA>": ""}))
    d["fm"] = d["FLIGHT_mvt"].astype(str).str.strip().str.upper().replace({"NAN": ""})
    tbl = EXT / "iata_icao.json"
    if tbl.exists():
        import json
        t = json.loads(tbl.read_text())
        d["fm_icao"] = [AD.iata_to_icao(v, t) for v in d["fm"]]
    else:
        d["fm_icao"] = ""
    d["stand"] = (d["AIRPORT"].astype(str) + "|"
                  + d["STAND_mvt"].astype(str).str.strip().str.upper())
    d["T"] = AD.epoch_s(d["MVT_TIME_UTC_mvt"])
    d["B"] = AD.epoch_s(d["BLOCK_TIME_UTC_mvt"])
    d["day"] = d["MVT_TIME_UTC_mvt"].dt.strftime("%Y-%m-%d")
    # the feature matrices index departures positionally, in file order, so carry that
    # position alongside the original index for later alignment
    d["pos"] = np.arange(len(d))
    return d


def collect_stands(dep, day_list):
    rows = []
    for day in day_list:
        sub = dep[dep["day"] == day]
        if not len(sub):
            continue
        tr, lk = AD.load_day(ADSB / ("ground_%s.parquet" % day))
        n = 0
        for a, cs, st, B in zip(sub["AIRPORT"].astype(str), sub["cs"], sub["stand"], sub["B"]):
            g = AD.track_for(tr, lk, a, cs)
            if g is None or not np.isfinite(B):
                continue
            p = AD.stand_points(g, B)
            if p is None:
                continue
            rows.append((st, p[0], p[1]))
            n += 1
        print("  %s : %6d postes observes" % (day, n), flush=True)
    return pd.DataFrame(rows, columns=["stand", "lat", "lon"])


def reduce_stands(P, min_flights=AD.MIN_FLIGHTS):
    S = P.groupby("stand").agg(stand_n=("lat", "size"), lat=("lat", "median"),
                               lon=("lon", "median"))
    sp = P.join(S[["lat", "lon"]], on="stand", rsuffix="_c")
    sp["d"] = AD.metres(sp["lat_c"], sp["lon_c"], sp["lat"], sp["lon"])
    S["spread_m"] = sp.groupby("stand")["d"].median()
    return S[S["stand_n"] >= min_flights]


def stage_stands():
    dep = departures(2025)
    P = collect_stands(dep, days(2025))
    if not len(P):
        print("aucun point: il faut des jours de 2025 extraits")
        return
    S = reduce_stands(P).reset_index()
    S.to_parquet(EXT / "adsb_stands.parquet", index=False)
    print("\n%d postes retenus sur %d observes (>=%d vols)" % (
        len(S), P["stand"].nunique(), AD.MIN_FLIGHTS))
    print("dispersion autour de la position retenue: mediane %.0f m | p90 %.0f m" % (
        S["spread_m"].median(), S["spread_m"].quantile(0.9)))
    S["airport"] = S["stand"].str.split("|").str[0]
    print("\n%-7s %8s %22s %22s" % ("aero", "postes", "departs 2025 couverts", "departs 2026 couverts"))
    have = set(S["stand"])
    d26 = departures(2026)
    for a in sorted(dep["AIRPORT"].astype(str).unique()):
        g5 = dep[dep["AIRPORT"].astype(str) == a]
        g6 = d26[d26["AIRPORT"].astype(str) == a]
        print("%-7s %8d %21.1f%% %21.1f%%" % (
            a, (S["airport"] == a).sum(), 100 * g5["stand"].isin(have).mean(),
            100 * g6["stand"].isin(have).mean()))


def run_offblock(dep, S, day_list):
    res, seen = [], 0
    posmap = dep["pos"]
    for day in day_list:
        sub = dep[dep["day"] == day]
        if not len(sub):
            continue
        seen += len(sub)
        tr, lk = AD.load_day(ADSB / ("ground_%s.parquet" % day))
        for i, a, cs, st, T in zip(sub.index, sub["AIRPORT"].astype(str), sub["cs"],
                                   sub["stand"], sub["T"]):
            if st not in S.index:
                continue
            g = AD.track_for(tr, lk, a, cs)
            if g is None:
                continue
            r = S.loc[st]
            e = AD.estimate(g, T, r["lat"], r["lon"])
            if e is None:
                continue
            e["row"] = i
            e["pos"] = int(posmap[i])
            e["adsb_stand_spread"] = float(r["spread_m"])
            e["adsb_stand_n"] = float(r["stand_n"])
            res.append(e)
        print("  %s : %7d estimations cumulees" % (day, len(res)), flush=True)
    R = pd.DataFrame(res)
    return R, seen


def report(R, dep):
    j = dep.loc[R["row"]]
    truth = (j["T"] - j["B"]).to_numpy()
    print("\n%-22s %10s %10s %8s %8s" % ("candidat", "biais med", "sd", "<60s", "RMSE taxi"))
    for c in ("adsb_taxi_extrap", "adsb_taxi_last", "adsb_taxi_away"):
        e = truth - R[c].to_numpy()          # positive = we place off-block too late
        print("%-22s %+10.0f %10.0f %7.1f%% %8.1f" % (
            c, np.median(e), e.std(), 100 * (np.abs(e) < 60).mean(),
            np.sqrt(np.mean((R[c].to_numpy() - truth) ** 2))))
    print("\n  apres retrait d'un biais par aeroport (ce que le modele fera):")
    print("  %-22s %10s %10s" % ("candidat", "sd", "RMSE taxi"))
    apt = j["AIRPORT"].astype(str).to_numpy()
    for c in ("adsb_taxi_extrap", "adsb_taxi_last", "adsb_taxi_away"):
        r = R[c].to_numpy() - truth
        adj = r.copy()
        for a in np.unique(apt):
            m = apt == a
            adj[m] = r[m] - np.median(r[m])
        print("  %-22s %10.0f %10.1f" % (c, adj.std(), np.sqrt(np.mean(adj ** 2))))
    best = "adsb_taxi_extrap"
    e = truth - R[best].to_numpy()
    print("\n=== %s, par aeroport ===" % best)
    print("  %-7s %8s %10s %10s %8s" % ("aero", "n", "biais med", "sd", "<60s"))
    for a in sorted(np.unique(apt)):
        m = apt == a
        if m.sum() < 5:
            continue
        print("  %-7s %8d %+10.0f %10.0f %7.1f%%" % (
            a, m.sum(), np.median(e[m]), e[m].std(), 100 * (np.abs(e[m]) < 60).mean()))
    print("\n=== par quantite de preuve au poste ===")
    print("  %-10s %8s %10s %10s %8s" % ("points", "n", "biais med", "sd", "<60s"))
    n = R["adsb_n_stand"].to_numpy()
    for lo, hi, lab in ((1, 2, "1-2"), (3, 9, "3-9"), (10, 29, "10-29"), (30, 10 ** 9, ">=30")):
        m = (n >= lo) & (n <= hi)
        if m.sum() < 10:
            continue
        print("  %-10s %8d %+10.0f %10.0f %7.1f%%" % (
            lab, m.sum(), np.median(e[m]), e[m].std(), 100 * (np.abs(e[m]) < 60).mean()))


def stage_offblock(year):
    S = pd.read_parquet(EXT / "adsb_stands.parquet").set_index("stand")
    dep = departures(year)
    R, seen = run_offblock(dep, S, days(year))
    if not len(R):
        print("aucune estimation")
        return
    R.to_parquet(EXT / ("adsb_offblock_%d.parquet" % year), index=False)
    print("\n%d estimations sur %d departs des jours disponibles (%.1f%%)" % (
        len(R), seen, 100 * len(R) / max(seen, 1)))
    if year == 2025:
        report(R, dep)


def stage_split(day):
    """Positions from the morning, error measured on the afternoon. No flight informs
    the position used to judge it."""
    cut = pd.Timestamp(day + " 12:00:00", tz="UTC").timestamp()
    dep = departures(2025)
    dep = dep[(dep["day"] == day) & np.isfinite(dep["B"])]
    am, pm = dep[dep["B"] < cut], dep[dep["B"] >= cut]
    print("%s : %d departs (%d matin, %d apres-midi)" % (day, len(dep), len(am), len(pm)))
    P = collect_stands(am, [day])
    S = reduce_stands(P, min_flights=1)
    print("postes appris sur la matinee: %d (depuis %d vols)" % (len(S), len(P)))
    R, seen = run_offblock(pm, S, [day])
    if not len(R):
        print("aucune estimation")
        return
    print("\n%d estimations sur %d departs de l'apres-midi (%.1f%%)" % (
        len(R), seen, 100 * len(R) / max(seen, 1)))
    report(R, pm)


if __name__ == "__main__":
    if sys.argv[1] == "stands":
        stage_stands()
    elif sys.argv[1] == "offblock":
        stage_offblock(int(sys.argv[2]))
    else:
        stage_split(sys.argv[2])
