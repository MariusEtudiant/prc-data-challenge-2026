# PRC Data Challenge 2026 — team `reliable-hamburger`

Predicting taxi-out time at ten major European airports.
Challenge: <https://prc-data-challenge-2026.netlify.app/>

Licence: GPLv3 (see `LICENSE`). No challenge data is redistributed here — `.gitignore`
excludes `data/`, and every matrix is rebuilt from the official buckets by the scripts
below.

## The problem, restated

The target is `TAXITIME_SEC_mvt = MVT_TIME_UTC_mvt - BLOCK_TIME_UTC_mvt` for departures.
`MVT_TIME_UTC_mvt` (takeoff) **is given** in the ranking file; only `BLOCK_TIME_UTC_mvt`
is blanked. So this is not a forecasting problem: it is the **reconstruction of the
off-block time** of a flight whose takeoff time we already know. Verified exactly on
arrivals, where both are given: `TAXITIME == BLOCK - MVT` to 0.0 s over 344,693 rows.

That reframing drives everything. The most informative input is Network Manager's
`AOBT_3_flt`, its own record of off-block, present for 98.5% of ranking departures. But
NM is only precise on events a sensor observes:

| NM timestamp vs the airport's own | sd | within 60 s |
|---|---:|---:|
| `ARVT_3` vs landing (radar-observed) | 57 s | 87.4 % |
| `AOBT_3` vs off-block (observed by nobody) | 355 s | 19.8 % |

The ratio runs from 4.4× (EDDF) to 10.8× (EGLL). **The noise is in the feature, not in
the target** — `BLOCK_TIME` is clean — so the ceiling is set by how well off-block can be
reconstructed, not by label quality.

## Architecture

Per-airport models throughout: airports differ enough that one global model
compromises between them, and the per-airport split survives changes in test
composition (which mattered when July was reissued from 3 to 10 airports mid-challenge).

```
per-airport LightGBM (Optuna-tuned, 3 seeds)  ─┐
per-airport CatBoost (Optuna-tuned, GPU)      ─┼─ NNLS weights, fitted per airport
global entity-embedding MLP (10 seeds)        ─┤   on odd days, measured on even days
per-airport MLP (3 seeds)                     ─┘
```

Feature families, with the share of ordinary-traffic variance each explains alone
(per-airport, `y < 7200`):

| family | what it is | R² alone |
|---|---|---|
| plan | the five off-block timestamps and their deltas | **0.46–0.73** |
| queue/state | transductive counters built from surrounding traffic | 0.16–0.56 |
| geometry | stand, runway, PRU-style P10 reference time | 0.14–0.43 |
| turnaround | chain to the previous arrival at the same stand | 0.16–0.27 |
| regime | arrival-anomaly rates, departure regime, PRU ATFM | 0.06–0.42 |
| weather | METAR incl. de-icing proxies | 0.00–0.16 |
| time | hour, day of week, day of year | 0.00–0.08 |

The off-block timestamps carry almost everything — and they fail at exactly one
airport, Rome, where every family lands between −0.19 and +0.11. Two Rome flights with
the same `d_aobt3` differ by ±642 s: it is not badly modelled, it is unobservable in
this data. The same holds for Istanbul to a lesser degree; several teams independently
confirmed that no external source moves those two.

## Reproducing

```bash
python scripts/download_data.py                 # official buckets (needs .secrets/)
python scripts/prepare_data.py                  # parse timestamps
python scripts/exp_m13.py build --turnaround    # meta + X_v2_{mirror,full,ranking}
python scripts/rebuild_aux.py                   # all auxiliary matrices, both sides
python scripts/fetch_metar.py                   # Iowa Mesonet METAR
python scripts/prep_pru_data.py                 # EUROCONTROL PRU open data
python scripts/adsb_extract.py 2025-07-01 ...   # adsb.lol ground traces (ODbL)
python scripts/make_submission_v16.py nbr       # then: mirror | prod | blend
```

Validation is a **mirror** of the scored months: January and July 2025 held out, all ten
airports, giving 44.6 % / 55.4 % against the test's 44.3 % / 55.7 %. Blend weights are
fitted on odd days and measured on even days, never on one month alone — weights fitted
on January alone hid CatBoost's value for weeks (worth −8.9 once measured correctly).

Two metrics are reported side by side and both matter. The "cleaned" one excludes
day-offset recording artefacts (`|y − k·86400| < 3600`, 15 flights in 2025); the raw one
does not. Seven such flights move the mirror from 294 to 391, so a change can improve
one and worsen the other — judging only the cleaned metric is what made four
consecutively validated improvements fail to transfer.

## What was tried and rejected

Kept here because negative results with numbers are the useful part, and each cost a
day. All measured on the mirror with the protocol above.

- **Extreme-value handling.** Removing >2 h targets from training costs 42 points (402.8
  vs 360.7). Winsorise at 86,400, never delete; delete only the day-offset artefacts.
  A two-component regime mixture (classifier + conditional regressions) was our champion
  in early September and is *worse* than a tuned per-airport L2 model (+5.96): L2 already
  estimates the conditional mean, which is what RMSE asks for.
- **Explicit hedging on detected artefact candidates** (no flight plan, `d_sched` > 54,000 s,
  14.7 % precision, stable across semesters): +47.8. Break-even needs 1.5 artefacts among
  the candidates and the test holds 1.3–2.1 in total.
- **Model capacity.** 512 and 2,048 leaves both degrade; only a lower learning rate helps
  (−1.2 to −6.8). `max_bin` 255 matches 511 at twice the speed.
- **LightGBM categorical settings.** Loosening `max_cat_threshold`/`cat_smooth` hurts
  (+0.7 to +4.5) — the 32-category default is a regulariser, not a limitation.
- **Hierarchical reference times** adding the missing stand level: −0.14, within noise.
  The "79 % of thin stand×runway pairs" figure is misleading: those pairs are thin by
  flight count, and 90.8 % of flights do have a pair-level reference.
- **Forward-looking arrival windows** (legitimate here, since arrivals keep their taxi
  times in the ranking file): +2.35, and +14.9 at Rome. Day-level aggregates have as many
  degrees of freedom as there are days, and overfit.
- **Per-entity offsets of the `AOBT_3` residual.** The raw residual carries 16.7 % of its
  variance between stands, but the GBDT has already absorbed it: only 7–17 s of true
  between-group structure survives, and applying the offset double-counts (+16 to +22).
- **Bagging (ExtraTrees, RandomForest) in the blend**: weight 0.00 everywhere. Their
  errors correlate 0.949 with the tree consensus while the networks correlate 0.805 — the
  networks are the diverse component, forests are not.
- **The departure queue.** An oracle given the true taxi of the five previous departures
  on the same runway is worth −8.0 overall and −33.5 at LTFM, but only −4.0 at Rome.
  Substituting the neighbours' `d_aobt3` recovers −0.6; substituting our own out-of-fold
  predictions recovers −1.5 overall and −11.7 at LTFM.
- **External aggregates.** PRU daily ATFM delay flags three of Rome's four worst July days
  and is already in the matrix; it does not help, because knowing the day is bad does not
  say which flight suffers. PRU monthly ATXOT cannot be used to calibrate prediction
  levels: its published figures sit −107 s to +366 s from our own `d_aobt3`, airport-specific.
- **A stand-occupancy lower bound** from the next arrival at the same stand is vacuous:
  that arrival lands after our takeoff in 100 % of cases.

## Acknowledgements

Open data: EUROCONTROL PRU (ATFM delays, traffic, ATXOT), Iowa State Mesonet (METAR),
adsb.lol `globe_history` (ODbL). Discussion in the OpenSky Discord
`#prc-data-competition` shaped several decisions, in particular the January↔July
validation scheme and the use of adsb.lol surface traces.
