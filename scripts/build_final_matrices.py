"""Rebuild every ranking-side matrix for the final four-month set.

The organisers published `final_ranking.parquet` on 8 October: January, February, June and
July 2026, 670,790 departures against the 344,841 of the public board, with February and
June blind to everyone. The training side is unchanged, so only the ranking matrices have to
be rebuilt - but all of them, because every one is keyed positionally to the departures of
`ranking_2026.parquet`, which now holds twice as many rows.

Reference times and the AOBT spike model are fitted on 2025 and only applied here, exactly
as before; nothing is refitted on the new months.

Usage: python scripts/build_final_matrices.py
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
from dc26.weather import load_metar
from dc26.pipeline import fit_reference_times, build_features_v2, align_categories
from dc26.flight_context import build_flight_context
from dc26.dep_regime import build_dep_regime
from dc26.anomaly_regime import build_anomaly_regime
from dc26.aobt_quality import fit_aobt_spikes, build_aobt_quality
from dc26.state_features import build_runway_config_features

t00 = time.time()
rk = load_ranking_2026()
rk_dep = rk[rk["PHASE_mvt"] == "DEP"].copy()
print("jeu final: %d mouvements, %d departs, mois %s" % (
    len(rk), len(rk_dep), sorted(rk_dep["MVT_TIME_UTC_mvt"].dt.month.unique())), flush=True)

mv = load_movements_2025()
dep25 = mv[mv["PHASE_mvt"] == "DEP"].copy()
y25 = dep25["TAXITIME_SEC_mvt"]
refs_full = fit_reference_times(dep25, y25)
spike = fit_aobt_spikes(dep25)
print("references et modele de pics ajustes sur 2025 (%.0f s)" % (time.time() - t00), flush=True)
del dep25, y25
gc.collect()

Xf = pd.read_parquet(PROC / "X_v2_full_t.parquet")
met = load_metar()
t0 = time.time()
Xr = build_features_v2(rk_dep, rk, refs_full, met=met, state_cache="rankingfinal",
                       with_turnaround=True)
Xr = align_categories(Xr, Xf)
Xr.reset_index(drop=True).to_parquet(PROC / "X_v2_ranking_t.parquet", index=False)
rk_dep[["MVT_ID_mvt"]].reset_index(drop=True).to_parquet(PROC / "rank_ids_t.parquet", index=False)
print("X_v2_ranking_t: %s | couverture des references %.3f | %.0f s" % (
    Xr.shape, Xr["ref_txot_p10"].notna().mean(), time.time() - t0), flush=True)
cols = list(Xf.columns)
del Xf, Xr
gc.collect()

for name, fn in (("flightctx", lambda: build_flight_context(rk_dep)),
                 ("depregime", lambda: build_dep_regime(rk_dep, rk)),
                 ("rwycfg", lambda: build_runway_config_features(rk_dep, rk)),
                 ("anomreg", lambda: build_anomaly_regime(rk_dep, rk)),
                 ("aobtq", lambda: build_aobt_quality(rk_dep, spike))):
    t0 = time.time()
    f = fn().reset_index(drop=True)
    f.to_parquet(PROC / ("%s_ranking2026.parquet" % name), index=False)
    print("  %-10s %2d colonnes  %4.0f s" % (name, f.shape[1], time.time() - t0), flush=True)
    del f
    gc.collect()

print("\ntermine en %.0f min" % ((time.time() - t00) / 60))
print("reste a faire: scripts/build_airline.py (cie_ranking2026)")
