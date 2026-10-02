"""S01: stop guessing. Characterise what carries the signal, airport by airport.

Twelve hypotheses tested and rejected in one day is a sign of searching blind. The
question never asked: WHY does Munich reach R2 0.847 while Rome sits at 0.521 - when
Munich's main taxi estimate (d_aobt3) correlates only 0.128 with its target and
Rome's correlates 0.372, three times better? Something other than the Network
Manager timestamp carries Munich's signal, and until we know what, every idea is a
shot in the dark.

Method: train the same per-airport model on one FAMILY of features at a time, then on
all of them, and read off how much each family explains where. No new idea is
proposed here - the point is to see the machine we already have.

  temps       hour, dow, month, doy
  geometrie   stand, runway, the P10 reference (where the aircraft starts and ends)
  plan        the five off-block timestamps and their deltas (d_aobt3, EOBT, LOBT...)
  file        queue and airport-state counters
  rotation    the turnaround chain through the previous arrival at the stand
  meteo       METAR
  regime      anomaly rates, departure-side regime, PRU/ATFM
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
PROC = ROOT / "data" / "processed"
from dc26.validation import rmse

CAP, MIN_TRAIN = 86400, 20000
FILES = ["X_v2_mirror_t.parquet", "flightctx_train2025.parquet", "depregime_train2025.parquet",
         "aobtq_train2025.parquet", "anomreg_train2025.parquet"]

X = pd.concat([pd.read_parquet(PROC / f) for f in FILES], axis=1)
cols = list(X.columns)


def pick(*subs, exact=()):
    out = [c for c in cols if any(s in c for s in subs)] + [c for c in cols if c in exact]
    return sorted(set(out))


FAM = {
    "temps": pick(exact=("hour", "dow", "month", "doy")),
    "geometrie": pick("ref_txot", exact=("STAND", "RUNWAY", "AIRPORT")),
    "plan": pick("d_aobt3", "d_eobt1", "d_lobt", "d_iobt", "d_sched", "has_fpl", "rev_", "dur_",
                 "implied_", "aobt_", "callsign"),
    "file": pick("q_", "deps_", "arrs_", "gap_", "push_", "backlog"),
    "rotation": pick("turnaround", "inbound", "taxi_upper"),
    "meteo": pick("tmpc", "spread", "wind", "gust", "vsby", "ceiling", "wx_", "snow_", "freezing_",
                  "deice", "min_vsby"),
    "regime": pick("arr_anom", "dep_proxy", "atfm_", "atxot", "arr_taxi", "rwy_config", "dep_long"),
}
used = set().union(*FAM.values())
FAM["reste"] = sorted(set(cols) - used)
for k, v in FAM.items():
    print("%-11s %3d variables" % (k, len(v)))
print("total %d sur %d colonnes\n" % (len(set().union(*FAM.values())), len(cols)), flush=True)

meta = pd.read_parquet(PROC / "meta_train2025.parquet")
y = meta["y"].astype("float64")
tr, va, es = meta["tr"].to_numpy(), meta["va"].to_numpy(), meta["es"].to_numpy()
ok = (y > 0).to_numpy() & (y < 200000).to_numpy()
art = np.zeros(len(y), bool)
for k in (1, 2, 3):
    art |= (y - k * 86400).abs().to_numpy() < 3600
ok &= ~art
apt = X["AIRPORT"].astype(str).to_numpy()
apt_va, y_va, clean = apt[va], y[va].to_numpy(), ~art[va]
# ordinary traffic: we are characterising the bulk, the tail is a separate story
base_va = clean & (y_va < 7200)

P = dict(objective="regression", metric="rmse", verbosity=-1, num_threads=0, seed=42,
         learning_rate=0.05, num_leaves=127, min_data_in_leaf=100, feature_fraction=0.9,
         bagging_fraction=0.9, bagging_freq=1, max_bin=255, feature_pre_filter=False)

order = ["temps", "geometrie", "plan", "file", "rotation", "meteo", "regime"]
print("%-7s %8s %8s | %s" % ("aero", "sd cible", "TOUT", " ".join("%9s" % f[:9] for f in order)), flush=True)
t0 = time.time()
for a in sorted(set(apt)):
    f, s = ok & tr & (apt == a) & ~es, ok & tr & (apt == a) & es
    sv = (apt_va == a) & base_va
    if f.sum() < MIN_TRAIN or sv.sum() < 500:
        continue
    sd = y_va[sv].std()
    row = []
    for fam in order + ["ALL"]:
        cs = cols if fam == "ALL" else FAM[fam]
        cs = [c for c in cs if c in X.columns]
        if not cs:
            row.append(np.nan)
            continue
        Xa = X[cs]
        ca = [c for c in cs if Xa[c].dtype.name == "category"]
        m = lgb.train(P, lgb.Dataset(Xa[f], y[f].clip(upper=CAP), categorical_feature=ca,
                                     params={"feature_pre_filter": False, "max_bin": 255}),
                      num_boost_round=3000,
                      valid_sets=[lgb.Dataset(Xa[s], y[s].clip(upper=CAP))],
                      callbacks=[lgb.early_stopping(100, verbose=False)])
        p = np.clip(m.predict(Xa[va][sv], num_iteration=m.best_iteration), 60, CAP)
        row.append(1 - (rmse(y_va[sv], p) / sd) ** 2)
    print("%-7s %8.0f %8.3f | %s   (%.0f min)" % (
        a, sd, row[-1], " ".join("%9.3f" % v for v in row[:-1]), (time.time() - t0) / 60), flush=True)
