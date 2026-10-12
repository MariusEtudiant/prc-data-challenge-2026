# Taxi-out time prediction — PRC Data Challenge 2026

Predicting the taxi-out time (off-block to take-off) of departures at ten European airports, for
the [EUROCONTROL / OpenSky PRC Data Challenge 2026](https://prc-data-challenge-2026.netlify.app/)
(team `reliable-hamburger`).

**Result: RMSE 282.40 s, 96th of 242 teams** (median team: 297.60 s).

## Main idea

The take-off time is given and only the off-block time is hidden, so the task is to rebuild a
past timestamp rather than to forecast. Three properties of the data drive the model:

- **9.6% of the off-block times are a copy of the schedule.** A classifier estimates the
  probability of a copy, a regressor handles the other flights, and the two are combined as an
  expectation.
- **The off-block is always within ±3606 s of the planned off-block (LOBT).** Every prediction
  is clipped to this window.
- **Departures without a flight-plan record are 1.5% of the rows but 40% of the error.** The
  airline is recovered from the airport's flight number, and ADS-B surface tracks are used to
  detect the pushback.

## Run

The challenge data is not included: it needs the organisers' credentials in
`.secrets/s3_credentials.json`.

```bash
pip install -r requirements.txt
python scripts/download_data.py
python scripts/prepare_data.py
python scripts/build_features.py build --turnaround
python scripts/build_auxiliary.py
python scripts/fetch_metar.py
python scripts/prep_pru_data.py
python scripts/build_airline.py
bash   scripts/adsb_download.sh 2025-01-07 2025-07-15
python scripts/adsb_features.py 2025
python scripts/make_submission.py mirror     # validation on held-out months
python scripts/make_submission.py prod
python scripts/make_submission.py prodbase
python scripts/make_submission.py gate
python scripts/make_submission.py daily
python scripts/make_submission.py submit
```

- `src/dc26/`: features, models and validation
- `scripts/`: the pipeline steps, in the order above
- `analysis/`: supporting studies

## Data and licence

Challenge data from the organisers (not redistributed), ADS-B tracks from
[adsb.lol](https://github.com/adsblol) (ODbL), METAR from the
[Iowa State Mesonet](https://mesonet.agron.iastate.edu/), and
[EUROCONTROL PRU](https://ansperformance.eu/) open data.

Code under GPLv3 ([`LICENSE`](LICENSE)). Ideas from the public repositories of teams
`outgoing-boat` and `gracious-tornado` were re-derived and measured before use; no code was
copied.
