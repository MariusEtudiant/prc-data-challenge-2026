"""R01: everything we can see about Rome in July.

The decomposition settled where the whole deficit sits. Our January (273.9) and our
three old July hubs (223.4) are second best of the five teams that have scores on both
test versions; on the seven airports July added we are last at 449 against 307. And
inside that segment one airport does the damage: without LIRF our seven would score
234, better than anyone else's whole segment. Rome in July is 15,838 test flights and
essentially our entire gap.

This characterises it from every angle available before proposing anything: what the
target looks like, where the error concentrates in time and space, whether it is bias
or variance, and whether the arrivals - which the ranking file keeps labelled - carry
any warning the departures do not.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
PROC, EXP = ROOT / "data" / "processed", ROOT / "experiments"
from dc26.validation import rmse

CAP = 86400
COLS = ["AIRPORT", "has_fpl", "hour", "RUNWAY", "STAND", "d_aobt3", "d_sched",
        "ref_txot_p10", "AIRCRAFT_OPERATOR_flt", "MARKET_SEGMENT_flt"]
X = pd.read_parquet(PROC / "X_v2_mirror_t.parquet", columns=COLS)
meta = pd.read_parquet(PROC / "meta_train2025.parquet")
y = meta["y"].astype("float64").to_numpy()
va = meta["va"].to_numpy()
art = np.zeros(len(y), bool)
for k in (1, 2, 3):
    art |= np.abs(y - k * 86400) < 3600

n39 = np.load(EXP / "m39_preds.npz")
ng = np.clip(n39["nn_global"], 60, CAP)
na = np.clip(np.where(np.isfinite(n39["nn_airport"]), n39["nn_airport"], n39["nn_global"]), 60, CAP)
V = np.vstack([np.load(EXP / "m96_reference.npy"), np.load(EXP / "v14_cb_mirror.npy"), ng, na])
W = json.loads((EXP / "v15_weights.json").read_text())
apt = X["AIRPORT"].astype(str).to_numpy()[va]
pred = np.full(va.sum(), np.nan)
for a, w in W.items():
    m = apt == a
    if m.any():
        pred[m] = np.clip(np.array(w) @ np.nan_to_num(V[:, m]), 60, CAP)

mvt = pd.read_parquet(PROC / "movements_2025.parquet", columns=["PHASE_mvt", "MVT_TIME_UTC_mvt"])
t = mvt[mvt["PHASE_mvt"] == "DEP"]["MVT_TIME_UTC_mvt"].reset_index(drop=True)
mon, day, hh = t.dt.month.to_numpy()[va], t.dt.day.to_numpy()[va], t.dt.hour.to_numpy()[va]
yv, av = y[va], art[va]
Xv = X[va].reset_index(drop=True)

L = (apt == "LIRF") & (mon == 7) & np.isfinite(pred) & ~av
J = (apt == "LIRF") & (mon == 1) & np.isfinite(pred) & ~av
print("LIRF juillet: n=%d RMSE=%.1f | janvier: n=%d RMSE=%.1f" % (
    L.sum(), rmse(yv[L], pred[L]), J.sum(), rmse(yv[J], pred[J])))
err = yv - pred

print("\n=== 1. concentration temporelle : part de l erreur par jour ===")
e = err[L] ** 2
s = pd.DataFrame({"d": day[L], "e": e, "y": yv[L], "p": pred[L]})
g = s.groupby("d").agg(n=("e", "size"), part=("e", lambda v: 100 * v.sum() / e.sum()),
                       rmse=("e", lambda v: np.sqrt(v.mean())),
                       ymed=("y", "median"), pmed=("p", "median"))
top = g.sort_values("part", ascending=False)
print("  %4s %6s %9s %9s %9s %9s" % ("jour", "n", "part err2", "RMSE", "y median", "p median"))
for d_, r in top.head(8).iterrows():
    print("  %4d %6d %8.1f%% %9.0f %9.0f %9.0f" % (d_, r["n"], r["part"], r["rmse"], r["ymed"], r["pmed"]))
print("  uniforme = %.1f%% par jour | les 3 pires = %.1f%%" % (100 / 31, top["part"].head(3).sum()))

print("\n=== 2. biais ou variance ? ===")
for lab, m in (("juillet", L), ("janvier", J)):
    print("  %-9s biais moyen=%+7.0f s | biais median=%+7.0f s | ecart-type de l erreur=%7.0f" % (
        lab, err[m].mean(), np.median(err[m]), err[m].std()))

print("\n=== 3. par heure de la journee (juillet) ===")
s = pd.DataFrame({"h": hh[L], "e": err[L] ** 2, "y": yv[L]})
g = s.groupby("h").agg(n=("e", "size"), rmse=("e", lambda v: np.sqrt(v.mean())), ymed=("y", "median"))
g["part"] = 100 * s.groupby("h")["e"].sum() / e.sum()
print("  " + " ".join("%02d:%4.0f" % (h, r["rmse"]) for h, r in g.iterrows() if r["n"] > 100))

print("\n=== 4. par piste et par type de vol (juillet) ===")
for col in ("RUNWAY", "MARKET_SEGMENT_flt"):
    v = Xv[col].astype(str).to_numpy()
    s = pd.DataFrame({"k": v[L], "e": err[L] ** 2, "y": yv[L]})
    g = s.groupby("k").agg(n=("e", "size"), rmse=("e", lambda x: np.sqrt(x.mean())), ymed=("y", "median"))
    g = g[g["n"] >= 200].sort_values("rmse", ascending=False)
    print("  %s:" % col)
    for k, r in g.head(6).iterrows():
        print("    %-24s n=%6d RMSE=%7.0f  y median=%6.0f" % (k[:24], r["n"], r["rmse"], r["ymed"]))

print("\n=== 5. les arrivees de Rome previennent-elles ? (elles sont etiquetees dans le test) ===")
mv = pd.read_parquet(PROC / "movements_2025.parquet",
                     columns=["AIRPORT", "PHASE_mvt", "MVT_TIME_UTC_mvt", "TAXITIME_SEC_mvt"])
a = mv[(mv["AIRPORT"] == "LIRF") & (mv["PHASE_mvt"] == "ARR")].copy()
a["d"] = a["MVT_TIME_UTC_mvt"].dt.day
a["m"] = a["MVT_TIME_UTC_mvt"].dt.month
a = a[a["m"] == 7]
ti = a["TAXITIME_SEC_mvt"].astype(float)
a = a[np.isfinite(ti) & (ti > 0) & (ti < 20000)]
ag = a.groupby("d")["TAXITIME_SEC_mvt"].agg(arr_med="median", arr_p90=lambda v: np.percentile(v, 90), arr_n="size")
j = g if False else top.join(ag)
print("  correlation entre la part d erreur du jour et le p90 des arrivees : %+.3f" % (
    np.corrcoef(j["part"], j["arr_p90"])[0, 1]))
print("  %4s %10s %10s %10s" % ("jour", "part err2", "arr median", "arr p90"))
for d_, r in j.sort_values("part", ascending=False).head(6).iterrows():
    print("  %4d %9.1f%% %10.0f %10.0f" % (d_, r["part"], r["arr_med"], r["arr_p90"]))
