"""Rebuild every auxiliary feature matrix, both sides, in one place.

Written on 2026-09-04 when July was regenerated (215,876 -> 344,841 departures, all
ten airports in July) and every ranking-side matrix had to be thrown away. The
builders used to live scattered across experiment and submission scripts; from here
on this is the single entry point, so a data refresh is one command.

  flightctx  plan-revision chain, no fit
  depregime  departure-side regime from other departures, no fit
  rwycfg     runway configuration, no fit
  anomreg    ambient arrival-anomaly regime, no fit
  aobtq      AOBT_3 spike model - FITTED, so it depends on the mirror split and must
             be rebuilt whenever that split changes (train version) and on all of
             2025 (full version, used in production)
"""
import gc
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
PROC = ROOT / "data" / "processed"

from dc26.data import load_movements_2025, load_ranking_2026
from dc26.flight_context import build_flight_context
from dc26.dep_regime import build_dep_regime
from dc26.anomaly_regime import build_anomaly_regime
from dc26.aobt_quality import fit_aobt_spikes, build_aobt_quality
from dc26.state_features import build_runway_config_features


def run(tag, mv, out_suffix, tr_mask=None):
    dep = mv[mv["PHASE_mvt"] == "DEP"].copy()
    print("[%s] %d departs" % (tag, len(dep)), flush=True)
    jobs = [
        ("flightctx", lambda: build_flight_context(dep)),
        ("depregime", lambda: build_dep_regime(dep, mv)),
        ("rwycfg", lambda: build_runway_config_features(dep, mv)),
        ("anomreg", lambda: build_anomaly_regime(dep, mv)),
    ]
    for name, fn in jobs:
        t0 = time.time()
        f = fn().reset_index(drop=True)
        p = PROC / ("%s_%s.parquet" % (name, out_suffix))
        f.to_parquet(p, index=False)
        print("  %-10s %-38s %2d col  %4.0f s" % (name, p.name, f.shape[1], time.time() - t0), flush=True)
        del f
        gc.collect()
    # aobtq: fitted, so two versions on the training side
    if tr_mask is not None:
        for lab, mask in (("train2025", tr_mask), ("full2025", np.ones(len(dep), bool))):
            model = fit_aobt_spikes(dep[mask])
            f = build_aobt_quality(dep, model).reset_index(drop=True)
            f.to_parquet(PROC / ("aobtq_%s.parquet" % lab), index=False)
            print("  %-10s aobtq_%s.parquet  %2d col" % ("aobtq", lab, f.shape[1]), flush=True)
            if lab == "full2025":
                global FULL_MODEL
                FULL_MODEL = model
            del f
            gc.collect()
    else:
        f = build_aobt_quality(dep, FULL_MODEL).reset_index(drop=True)
        f.to_parquet(PROC / ("aobtq_%s.parquet" % out_suffix), index=False)
        print("  %-10s aobtq_%s.parquet  %2d col" % ("aobtq", out_suffix, f.shape[1]), flush=True)


FULL_MODEL = None
meta = pd.read_parquet(PROC / "meta_train2025.parquet")
mv = load_movements_2025()
run("2025", mv, "train2025", tr_mask=meta["tr"].to_numpy())
del mv
gc.collect()

rk = load_ranking_2026()
run("2026", rk, "ranking2026")
print("termine", flush=True)
