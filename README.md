# MMA3001 Project — Indoor CO2 Prediction

[![CI](https://github.com/tgup0010-bot/mma3001-occupancy-prediction/actions/workflows/ci.yml/badge.svg)](https://github.com/tgup0010-bot/mma3001-occupancy-prediction/actions/workflows/ci.yml)

Individual project for MMA3001 (Numerical Methods and Machine Learning),
Monash University, 2026 S2.

## Engineering problem

Indoor CO2 concentration is a standard proxy for ventilation adequacy —
rising CO2 signals fresh-air supply isn't keeping up with occupant load.
This project predicts a sensor's CO2 concentration **15 minutes ahead**
from its own recent history and the building's current occupancy, so a
building control system could increase ventilation pre-emptively rather
than only reacting once CO2 is already elevated. It also directly tests
whether occupancy actually helps predict CO2 at all.

**Status: complete.** All four regression methods taught in MMA3001 Week 5
(Linear Regression, Decision Tree Regression, SVR, Neural Network
Regression) are implemented and compared, plus a Week 6 numerical-
integration analysis of accumulated CO2 exposure. See
[Current results](#current-results) below. The full written report is at
[`docs/report/MMA3001_Project_Report.docx`](docs/report/MMA3001_Project_Report.docx)
(also available as [Markdown](docs/report/report.md)).

## Data

Source: MMA3001 Dataset 2 (Monash Smart Infrastructure Occupancy and
Environmental Data), course-supplied — the occupancy event log (5 zones)
and the environmental sensor log (5 sensors: CO2, temperature, humidity,
and other variables in a nested-JSON payload).

**Only 1 of the 5 environmental sensors is used** (`6012002000326`):

- 2 sensors report every reading at the same frozen timestamp (a stuck
  internal clock, discovered while building this project — not usable as
  a time series at all).
- 1 sensor (`G.38`, the only one traceable to a named room) has only a
  4-month data window.
- Of the remaining 2, `6012002000326` was chosen because it's
  independently validated: its indoor temperature/humidity correlate
  r=+0.84/+0.65 with real Bureau of Meteorology weather for the same
  period (`scripts/fetch_bom_weather.py`,
  `scripts/weather_comparison_analysis.py` — see
  [`reports/weather_comparison_results.json`](reports/weather_comparison_results.json)).

A preliminary check (`scripts/sensor_matching_analysis.py`) also tested
whether any occupancy zone's pattern correlates with any environmental
sensor's readings, to see whether occupancy could be trusted as a
per-room feature — see
[`reports/sensor_matching_results.json`](reports/sensor_matching_results.json)
and the report §7 for that result.

Raw data files are not committed (see
[`data/raw/README.md`](data/raw/README.md) for how to obtain them).

## Repository structure

```
├── src/occupancy/       Python package: CO2 regression + integration
│                        (co2_regression.py, co2_models.py, co2_integration.py),
│                        environmental sensor loading, sensor-matching analysis,
│                        BoM external-weather comparison
├── scripts/             co2_prediction_experiment.py, co2_integration_analysis.py,
│                        co2_demo.py (live demo), fetch_bom_weather.py,
│                        weather_comparison_analysis.py, sensor_matching_analysis.py
├── tests/               pytest unit tests (62, all passing)
├── data/raw/            Raw data (gitignored — see data/raw/README.md);
│                        data/raw/bom/ holds fetched BoM weather CSVs
├── data/processed/      Small committed sample data, derived artefacts
├── docs/project_brief/  Unit-supplied project brief and dataset docs
├── docs/api/            Generated HTML code documentation (pdoc)
├── docs/report/         Written project report (.docx submission + .md source)
└── reports/             All experiment results (CO2 prediction, integration,
                         sensor-matching, BoM comparison)
```

## Setup

Requires Python ≥ 3.10.

```bash
pip install -e ".[dev]"
```

## Live demo (for the presentation)

```bash
python scripts/co2_demo.py
```

This is the thing to run live in the presentation/interview. It:

1. **Trains the best model (SVR) fresh** (a few seconds — fast enough not
   to need a saved model file).
2. **Replays 12 real examples** from the test set (data the model never
   trained on) — prints CO2 now, the model's 15-minute-ahead prediction,
   and what actually happened.
3. **Lets you type in your own scenario** — current CO2, occupancy level,
   time of day — and get a live prediction.

## Running the tests

```bash
pytest                                          # 62 unit tests
pytest --junitxml=reports/test_report.xml       # regenerate the committed test report
```

## Generating code documentation

```bash
pdoc --output-dir docs/api src/occupancy
```

## Running the full experiments

Requires the raw CSVs in `data/raw/` (see
[`data/raw/README.md`](data/raw/README.md)):

```bash
python scripts/co2_prediction_experiment.py
```

Loads the occupancy log and environmental sensor log, builds the CO2
supervised-regression table, splits chronologically (never randomly), and
trains/evaluates all four Week 5 regression methods — both with and
without building occupancy as a feature. Results are written to
`reports/co2_prediction_results.json`.

```bash
python scripts/co2_integration_analysis.py
```

Computes accumulated CO2 exposure via trapezoidal and Simpson's-rule
numerical integration (MMA3001 Week 6), including a gap-aware variant that
excludes sensor-outage panels rather than integrating across them, and a
fixed-window convergence comparison across bin sizes. Results are written
to `reports/co2_integration_results.json`.

### Current results

**CO2 regression** (chronological holdout, cutoff 11 March 2026;
17,010 rows after dropping rows with missing lag/rolling history —
79.9% dropped, see `data_info` in `reports/co2_prediction_results.json`
and the report §5 for why):

| Model | MAE (ppm) | RMSE (ppm) | R² |
|---|---|---|---|
| Linear Regression | 9.012 | 25.579 | 0.3581 |
| Decision Tree Regression | 9.353 | 28.708 | 0.1914 |
| **SVR** | **8.350** | 25.894 | 0.3422 |
| Neural Network Regression | 9.350 | 26.015 | 0.3360 |

**SVR has the lowest error.** Critically, **building occupancy does not
improve prediction for any of the four methods** (tested directly, not
assumed — see the report §4 for the full with/without-occupancy
comparison table) — consistent with the preliminary sensor-matching check
finding that no occupancy zone links to this sensor's room.

**Numerical integration:** over a fixed, genuinely gap-free 41-hour
window, trapezoidal and Simpson's rule agree to within 0.1% at every
tested resolution (15/30/60 min) — see the report §6. Getting there
required finding and fixing a real bug: naively integrating across a
636-day sensor outage as if CO2 varied linearly through it.

Full results, method, and discussion: [`docs/report/MMA3001_Project_Report.docx`](docs/report/MMA3001_Project_Report.docx), §§4–6.

## Roadmap

- [x] CO2 regression: 4 Week 5 methods, compared with/without occupancy
- [x] Week 6 numerical integration of accumulated CO2 exposure
- [x] Sensor-matching and BoM external-weather validation (sensor selection)
- [x] Written report (`docs/report/MMA3001_Project_Report.docx`)
- [x] AI-use reflection section in the report (student review still
      recommended before submission — see the report's note to reader)

## AI use

AI assistance (Claude) was used during this project for planning,
scaffolding, and code review. Full disclosure, what was AI-generated vs.
verified/written by the student, and a critical reflection on its use are
in the project report ([`docs/report/`](docs/report)), per the unit's
requirements.

## License

Code is MIT-licensed — see [`LICENSE`](LICENSE). This does not cover the
course-supplied raw data (excluded from version control; see
`data/raw/README.md`).
