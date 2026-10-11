"""Airline features from FLIGHT_mvt, for the flights where Network Manager has nothing.

Flights without an NM record are 1.54 % of departures and 39.7 % of our squared error, and
on every one of them `AIRCRAFT_OPERATOR_flt` is missing - the operator comes from the same
record that is absent. So the model is blind to the airline exactly where it matters most.

`FLIGHT_mvt` is the airport's own identifier and is present on 100 % of departures. Its
prefix is the airline, and the rate at which an airline's off-block timestamps equal the
scheduled time varies from 98 % to 0 % between carriers while staying stable across months
(correlation 0.93 between the ten unscored months and January plus July). That rate is what
tells the copy classifier whether to believe a delay.

The rates are target statistics, so they are computed leave-one-month-out: a flight's own
month never contributes to the rate it is given. For the 2026 ranking side there is no such
concern - the rates come from all of 2025 - but the same code path is used so the two sides
cannot drift apart.

Usage: python scripts/build_airline.py
"""
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"
EPS = 60.0
MIN_N = 30          # below this an airline keeps the global rate rather than its own
COLS = ["cie_copy", "cie_copy_nonm", "cie_n", "cie_delay_med", "cie_y_med"]
_RE = re.compile(r"^([A-Z0-9]{2,3}?)\d")


def prefix(s):
    return s.astype(str).str.upper().str.extract(r"^([A-Z0-9]{2,3}?)\d")[0].fillna("?")


def load(year):
    f = "movements_2025.parquet" if year == 2025 else "ranking_2026.parquet"
    c = ["AIRPORT", "PHASE_mvt", "MVT_TIME_UTC_mvt", "BLOCK_TIME_UTC_mvt",
         "SCHED_TIME_UTC_mvt", "FLIGHT_ID_mvt", "FLIGHT_mvt"]
    m = pd.read_parquet(PROC / f, columns=c)
    d = m[m["PHASE_mvt"] == "DEP"].reset_index(drop=True)
    sec = lambda x: ((d[x] - pd.Timestamp(0, tz="UTC")) / pd.Timedelta(seconds=1)).to_numpy()
    d["mvt"], d["sch"] = sec("MVT_TIME_UTC_mvt"), sec("SCHED_TIME_UTC_mvt")
    d["blk"] = sec("BLOCK_TIME_UTC_mvt")
    d["cie"] = prefix(d["FLIGHT_mvt"])
    d["mon"] = d["MVT_TIME_UTC_mvt"].dt.month
    d["nonm"] = d["FLIGHT_ID_mvt"].isna()
    d["ms"] = d["mvt"] - d["sch"]
    return d


src = load(2025)
src["y"] = src["mvt"] - src["blk"]
src["eq"] = (np.abs(src["blk"] - src["sch"]) <= EPS)
good = (src["y"] > 0) & (src["y"] < 200000)
S = src[good]
g_copy = S["eq"].mean()
g_copy_nonm = S.loc[S["nonm"], "eq"].mean()
g_y = S["y"].median()
g_d = S["ms"].median()
print("taux de copie global %.4f | sur les vols sans NM %.4f" % (g_copy, g_copy_nonm))
print("compagnies distinctes: %d | prefixe trouve sur %.2f%% des departs" % (
    S["cie"].nunique(), 100 * (S["cie"] != "?").mean()))


def rates(sub):
    """Copy rates and level statistics for one slice of history."""
    a = sub.groupby("cie").agg(n=("eq", "size"), copy=("eq", "mean"),
                               y_med=("y", "median"), d_med=("ms", "median"))
    b = sub[sub["nonm"]].groupby("cie").agg(n2=("eq", "size"), copy_nonm=("eq", "mean"))
    a = a.join(b)
    a.loc[a["n"] < MIN_N, "copy"] = g_copy
    a.loc[a["n"] < MIN_N, "y_med"] = g_y
    a["copy_nonm"] = np.where(a["n2"].fillna(0) >= 10, a["copy_nonm"], g_copy_nonm)
    return a


def attach(d, table_for):
    out = pd.DataFrame(index=np.arange(len(d)), columns=COLS, dtype="float64")
    for key, idx in d.groupby("mon").groups.items():
        t = table_for(key)
        i = np.asarray(idx)
        c = d["cie"].to_numpy()[i]
        out.iloc[i, 0] = t["copy"].reindex(c).fillna(g_copy).to_numpy()
        out.iloc[i, 1] = t["copy_nonm"].reindex(c).fillna(g_copy_nonm).to_numpy()
        out.iloc[i, 2] = t["n"].reindex(c).fillna(0).to_numpy()
        out.iloc[i, 3] = t["d_med"].reindex(c).fillna(g_d).to_numpy()
        out.iloc[i, 4] = t["y_med"].reindex(c).fillna(g_y).to_numpy()
    out["cie"] = pd.Categorical(d["cie"].to_numpy())
    return out


# 2025: leave the flight's own month out of the statistics it receives
cache = {}


def lomo(month):
    if month not in cache:
        cache[month] = rates(S[S["mon"] != month])
    return cache[month]


tr = attach(src, lomo)
tr.to_parquet(PROC / "cie_train2025.parquet", index=False)
print("\ncie_train2025: %d lignes, %d colonnes" % (len(tr), len(tr.columns)))

# 2026: every rate comes from the whole of 2025, so no month is special
full = rates(S)
rk = load(2026)
rk_out = attach(rk, lambda _m: full)
rk_out.to_parquet(PROC / "cie_ranking2026.parquet", index=False)
print("cie_ranking2026: %d lignes" % len(rk_out))

seen = set(S["cie"])
print("\ncompagnies de 2026 inconnues en 2025: %.2f%% des departs" % (
    100 * (~rk["cie"].isin(seen)).mean()))
print("\n=== ce que la variable separe, sur 2025 ===")
q = pd.qcut(tr["cie_copy"], 5, duplicates="drop")
t = pd.DataFrame({"q": q, "eq": src["eq"], "y": src["y"], "ok": good})
t = t[t["ok"]]
print("  %-24s %10s %12s %12s" % ("quintile du taux", "n", "copie reelle", "y median"))
for k, s in t.groupby("q", observed=True):
    print("  %-24s %10d %11.2f%% %12.0f" % (str(k), len(s), 100 * s["eq"].mean(),
                                            s["y"].median()))
print("\n=== et sur les vols sans NM retardes de plus d'une heure ===")
m = good & src["nonm"] & (src["ms"] > 3600)
t2 = pd.DataFrame({"r": tr.loc[m, "cie_copy_nonm"], "eq": src.loc[m, "eq"],
                   "y": src.loc[m, "y"]})
q2 = pd.qcut(t2["r"], 4, duplicates="drop")
print("  %-24s %10s %12s %12s" % ("quartile du taux sans NM", "n", "copie reelle", "y median"))
for k, s in t2.groupby(q2, observed=True):
    print("  %-24s %10d %11.2f%% %12.0f" % (str(k), len(s), 100 * s["eq"].mean(),
                                            s["y"].median()))
