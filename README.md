# Taxi-out time prediction — PRC Data Challenge 2026

Predicting how long an aircraft spends between leaving its gate and lifting off, for every
departure at ten major European airports. Entry for the [EUROCONTROL / OpenSky PRC Data
Challenge 2026](https://prc-data-challenge-2026.netlify.app/), team `reliable-hamburger`.

**Result: RMSE 282.40 s on 344,841 held-out departures — 96th of 242 teams.**
The public leaderboard runs from 212.85 (first) to 664 (last); the median team scores 297.

```
                     RMSE on the leaderboard
  first  ████████████████████                     212.85
  30th   ██████████████████████▌                  237.57
  ours   ███████████████████████████              282.40
  median ████████████████████████████▍            297.60
```

---

## The problem is not what it looks like

The target is `taxi = takeoff − off-block`. The obvious reading is "forecast how long the
taxi will take". That reading is wrong, and seeing why was worth more than any model.

**The takeoff time is given.** Only the off-block time is hidden. So this is not forecasting
— it is *reconstructing a past timestamp* for a flight whose later history we already know.
Verified exactly on the arrival rows, where both timestamps are published: `taxi == takeoff −
off-block` to 0.0 s across 344,693 rows.

That reframing changes what counts as a good feature. The most informative input is not
weather or congestion, it is Network Manager's own record of the off-block,
`AOBT_3_flt`. And its quality is the whole problem:

| Network Manager timestamp vs the airport's own | sd | within 60 s |
|---|---:|---:|
| landing time (observed by radar) | 57 s | 87.4 % |
| off-block (observed by nobody) | **355 s** | 19.8 % |

The noise is in the *input*, not in the label. The ceiling is set by how well a past
off-block can be reconstructed.

---

## Three exact properties of the data

Each of these is a statement verified over all 2,085,047 departures of 2025. Together they
were worth more than every feature in the model.

### 1. A tenth of the off-block timestamps are a copy of the schedule

`|off-block − schedule| ≤ 60 s` holds for **9.57 %** of departures — and for **84.7 %** of
the departures whose taxi time exceeds three hours. For those flights the answer is simply
`takeoff − schedule`, and the schedule is given for every row.

Most of the 9.57 % are not recording errors at all; they are ordinary flights that pushed
back on time. The distinction does not matter, because the answer is the same either way.
So the model is split in two: a classifier for "is this one of them", and a regressor
trained only on the rest.

### 2. The off-block never leaves the LOBT window

Of the 2,062,440 departures carrying a planned off-block (`LOBT_flt`), **not one** has
`|off-block − LOBT| > 3606 s`. Zero violations in two million rows. Every prediction is
therefore projected onto `takeoff − LOBT ± 3606`.

This is not physics — it is the rule the organisers used to join flight-plan records to
movements. Realising that explained a later failure (see *What did not work*).

### 3. 1.5 % of the rows carry 40 % of the error

Departures with no flight-plan record are 1.54 % of rows and **39.7 %** of our squared
error. On every one of them the airline field is missing too, because it comes from the
record that is absent — so the model was blind to the carrier exactly where the carrier
matters. The airport's own flight number fills the gap, and the rate at which a carrier's
off-block equals its schedule runs from 98 % to 0 % between airlines while staying stable
across months.

---

## The model

```
                organiser data · METAR · EUROCONTROL · ADS-B surface tracks
                                        │
                ┌───────────────────────┴───────────────────────┐
                ▼                                               ▼
       single L2 regressor                              TWO-STAGE
       per airport                          p  = P(off-block is the schedule)
                                            ŷₙ = per-airport regressor on the rest
                                            ŷ  = p·(takeoff − schedule) + (1 − p)·ŷₙ
                └───────────────────────┬───────────────────────┘
                                        ▼
                     per-airport weight, fitted on the body of the
                     other scored month
                                        ▼
                          projection onto the LOBT window
                                        ▼
                  reconciliation with EUROCONTROL's daily totals
```

**The mixture is an expectation, never an argmax.** Picking the likelier class minimises
classification error, which is not the loss being scored. A missed copy at `p = 0.45` costs
hours squared; spending `p·(takeoff − schedule)` seconds of prediction on the possibility is
what a squared loss asks for. The classifier reaches AUC 0.888 and is calibrated to the
third decimal (predicted 0.0088 against 0.0083 observed in the lowest bucket, 0.905 against
0.916 in the highest).

**Neither model is chosen outright.** The mixture is unbiased but noisier than the plain
regressor. Where copies are common and delays are long it wins by a wide margin; where they
are rare it loses.

| | plain + window | two-stage | gated |
|---|---:|---:|---:|
| Rome | 852.1 | **567.9** | 568.1 |
| Barcelona | **221.3** | 241.3 | 221.8 |
| all ten | 362.15 | 318.57 | **315.85** |

The weight is fitted per airport on flights under an hour, on the *other* scored month. Both
choices were forced by the gate failing first: split too finely, several cells had too few
rows and fell back on the base, discarding the single largest win in the model; fitted on
the full RMSE, a handful of multi-hour flights decided the weight and the gate came out
worse than either model it was blending.

---

## Validation

January and July 2025 are held out as a **mirror** of the scored months, matching their
airport and month composition. Each is predicted by a model trained on everything except
itself.

Two things made the measurements trustworthy:

- **The raw metric, always.** A "cleaned" metric that drops recording artefacts reads 279
  where the raw one reads 319 on the same predictions. Judging on the cleaned one is what
  made four consecutively validated changes fail to transfer.
- **A known transfer factor.** Mirror gains do not arrive whole. Measured across two scored
  submissions: a mirror gain of 86.1 s produced an official gain of 60.9 s — **71 %**. Every
  projection in this repository is discounted by that.

| | raw mirror | leaderboard |
|---|---:|---:|
| first submission | 401.93 | 343.31 |
| + LOBT window | 362.15 | — |
| + two-stage and airline rates | 318.57 | — |
| + per-airport gate | **315.85** | **282.40** |

---

## Off-block from ADS-B surface tracks

An aircraft parked at a gate has its transponder off; it comes alive at pushback. So the
first time receivers hear it — after a silence long enough to be a parking — is its
off-block. Matched against the truth on held-out days, with a per-airport offset removed.

Getting there took three attempts, and the two failures are the interesting part:

1. **A groundspeed threshold** put the off-block 483 s late on average. It cannot tell a
   pushback from a hold at a taxiway intersection.
2. **The stand's own position**, learned from our own data, reached 131 s of spread against
   `AOBT_3`'s 355 — but only 11 % coverage, because every flight needed a known stand.
3. **The first sighting after a silence** needs no stand at all: 85 % coverage.

The unlock was not the method but the matching. The callsign comes from the flight-plan
record, so matching on it reaches **0 %** of the flights that have no such record — exactly
the 1.5 % that carry 40 % of the error. Adding the airport's own flight number, plus an
IATA→ICAO prefix table derived from the 1.63 M departures where both forms appear, brings
those to 59 %.

Coverage is uneven and no model fixes that: 84–97 % at Madrid, Munich, Zurich and Barcelona,
70 % at Paris, **3 % at Istanbul**.

---

## What did not work

Negative results with the measurement that killed each one. Three were ideas taken from
other teams' published repositories and rejected after re-deriving them here.

| idea | measured | why |
|---|---:|---|
| **Orphan LOBT window** — recover a missing flight plan from the flight's own arrival record and apply the window | **+23.35** | The matching is right (99.84 % exact on flights that have both), but the window is the organisers' *join rule*. A flight without a record is one that failed that rule, so the window is the one constraint it is guaranteed to violate. |
| **Cap predictions at `takeoff − schedule`** | **+2.56** | 24 % of flights push back *before* their schedule, and 45 % of the long ones. The cap is wrong half the time precisely where it acts. |
| **A line on `takeoff − schedule` for the flights with no record**, fitted globally | 27,324 RMSE on the rows it touches | Fitted per airport *inside the fold* instead, the same idea gives 5,856 on those rows and is worth **−49.8** overall — the largest single component of the model. Same mechanism, opposite sign. |
| Stand-position ADS-B detector | 11 % coverage | Superseded by the first-sighting rule. |
| Shrinking predictions toward the group mean | 0 at nine airports of ten | The model's discrimination already beats any constant (Rome 4,407 against 7,986). |
| "The off-block is the previous arrival's on-block" | 0.9 % within 300 s | A plausible mechanical story for the implausible targets, cleanly refuted. |
| Calibrating levels against published monthly means | artefact | The first version compared a January prediction against an annual average. January and July run 1 to 50 s above the year. |
| Removing targets above 2 h from training | +42 | Winsorise, never delete. |
| LightGBM on GPU | slower | Blocked by categorical cardinality above 255. |

---

## Reproducing

The challenge data is not redistributed here; it needs the organisers' per-team credentials
in `.secrets/s3_credentials.json`.

```bash
pip install -r requirements.txt

python scripts/download_data.py            # organiser parquet files
python scripts/prepare_data.py             # parse timestamps
python scripts/build_features.py build --turnaround
python scripts/build_auxiliary.py          # queue, regime and quality matrices
python scripts/fetch_metar.py              # Iowa Mesonet weather
python scripts/prep_pru_data.py            # EUROCONTROL open data
python scripts/build_airline.py            # airline copy rates, leave-one-month-out

bash   scripts/adsb_download.sh 2025-01-07 2025-07-15   # adsb.lol traces (ODbL)
python scripts/adsb_features.py 2025

python scripts/make_submission.py mirror   # validate on held-out months
python scripts/make_submission.py prod     # two-stage on the full year
python scripts/make_submission.py prodbase # plain regressor, for the gate
python scripts/make_submission.py gate     # per-airport weights
python scripts/make_submission.py daily    # EUROCONTROL daily reconciliation
python scripts/make_submission.py submit   # write the parquet
python scripts/submit.py send submissions/<file>.parquet
```

| | |
|---|---|
| `src/dc26/` | feature construction, queue and regime models, validation |
| `scripts/` | the pipeline, in the order above |
| `analysis/` | the five studies whose numbers this README quotes |

---

## Data

| source | used for | licence |
|---|---|---|
| EUROCONTROL / OpenSky challenge data | movements and flight plans | organisers', not redistributed |
| [adsb.lol](https://github.com/adsblol) `globe_history` | surface tracks | ODbL 1.0 |
| [Iowa State Mesonet](https://mesonet.agron.iastate.edu/) | METAR | public domain |
| [EUROCONTROL PRU](https://ansperformance.eu/) | ATFM delays, traffic, daily pre-departure delay | public, attribution |

---

## Licence and credits

GPLv3, see [`LICENSE`](LICENSE).

The challenge Discord and the public repositories of teams `outgoing-boat` and
`gracious-tornado` were read for ideas. Everything taken that way was re-derived and
re-measured on this data before being implemented, and three of the ideas were **rejected**
by those measurements — they are in the table above with their numbers. No third-party
competition code was copied.
