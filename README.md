# MMA3001 Project — Room Occupancy Pattern Modelling

Individual project for MMA3001 (Numerical Methods and Machine Learning),
Monash University, 2026 S2.

## Engineering problem

Predicting whether a room will be occupied in the near future lets a
building's HVAC and lighting systems scale back conditioning in empty rooms
and pre-condition ahead of expected arrivals — reducing energy waste without
harming comfort. This project builds and validates a short-term occupancy
prediction model from historical sensor data collected in Monash's Smart
Infrastructure ("Living Lab") building.

**Status: in progress.** Data pipeline, baselines and a first ML model are
implemented and validated end-to-end on the real dataset (see
[Current results](#current-results) below). See
[`docs/report/`](docs/report) for the written report once drafted.

## Data

Source: MMA3001 Dataset 2 (Monash Smart Infrastructure Occupancy and
Environmental Data), course-supplied.

- **Occupancy event log** — 9.09M rows, 5 rooms, continuous coverage from
  Nov 2023 to Apr 2026 (an event is logged only when a room's occupancy
  state changes). This is the dataset this project uses.
- Environmental sensor readings and a second, later 2025–2026 data batch
  were also supplied but are **not used** here — see
  [Limitations](#known-data-limitations) below for why.

Raw data files are not committed to this repository (see
[`data/raw/README.md`](data/raw/README.md) for sizes and how to obtain
them); a small representative sample is committed at
[`data/processed/occupancy_sample_50k.csv`](data/processed/occupancy_sample_50k.csv)
for fast development and testing.

### Known data limitations

- Of the 5 occupancy zones, only 2 could be traced to a named physical room
  via the supplied location spreadsheet (`G.20`, and `G.25` — "Keenan Lab").
  The other 3 are modelled as anonymous zones.
- The environmental sensor data's 5 sensors don't resolve to the same rooms
  as the occupancy zones (only 1 of 5 could be matched at all, to a
  different room, `G.38`), so an occupancy↔environment relationship could
  not be validated against the supplied metadata and was deliberately
  **not** attempted — see the report for the full reasoning.
- A supplementary "Sample Data" batch (`PMV`, `Detailed` files) has no
  column headers and, as of writing, no clarification has been published
  on EdStem explaining what they represent. It is not used.

## Repository structure

```
├── src/occupancy/       Python package: data loading, preprocessing,
│                        feature engineering, baselines, ML model, evaluation
├── scripts/             Runnable end-to-end experiment script
├── tests/               pytest unit tests (23, all passing)
├── data/raw/            Raw data (gitignored — see data/raw/README.md)
├── data/processed/      Small committed sample data, derived artefacts
├── docs/project_brief/  Unit-supplied project brief and dataset docs
├── docs/api/            Generated HTML code documentation (pdoc)
├── docs/report/         Written project report (added once drafted)
└── reports/             Experiment results, test report, figures
```

## Setup

Requires Python ≥ 3.10.

```bash
pip install -e ".[dev]"
```

## Running the tests

```bash
pytest                                          # 23 unit tests
pytest --junitxml=reports/test_report.xml       # regenerate the committed test report
```

## Generating code documentation

```bash
pdoc --output-dir docs/api src/occupancy
```

## Running the full experiment

Requires the raw occupancy CSV in `data/raw/` (see
[`data/raw/README.md`](data/raw/README.md)):

```bash
python scripts/run_experiment.py
```

This loads the full 9.09M-row event log, resamples it to a 15-minute
per-room grid, builds the supervised next-interval-occupancy dataset,
splits chronologically (80/20, never randomly — a random split would leak
future information into training), and evaluates the persistence baseline,
the time-of-day Markov baseline, and the logistic regression model.
Results are written to `reports/experiment_results.json`.

### Current results

On a chronological holdout (train up to 2025-10-15, test after), out of
342,401 resampled 15-minute bins (26.5% dropped for missing lag/rolling
history — see `data_info` in `reports/experiment_results.json`):

| Model | Accuracy | F1 | ROC-AUC | Brier score (↓ better) |
|---|---|---|---|---|
| Persistence baseline | 0.816 | 0.771 | 0.809 | 0.184 |
| Markov (time-of-day) baseline | 0.817 | 0.772 | 0.870 | 0.139 |
| Logistic regression | **0.825** | **0.781** | **0.890** | **0.128** |

The logistic regression model outperforms both baselines on every metric,
including calibration (Brier score) — a first piece of evidence that the
engineered calendar/history features carry real predictive signal beyond
"a room's state tends to persist" and "occupancy follows daily patterns."
This is not yet a final result: see the Roadmap below for sensitivity
analysis and optimisation work still to come.

## Roadmap

- [x] Data loading + schema validation
- [x] Event log → regular time-grid preprocessing (with explicit max-gap
      handling of long sensor outages)
- [x] Feature engineering (calendar cyclical features, lag, rolling mean)
- [x] Persistence baseline
- [x] Time-of-day Markov chain baseline
- [x] Logistic regression ML model
- [x] Chronological validation split + evaluation metrics
- [ ] Sensitivity analysis: bin size and lookback-window trade-offs
- [ ] Runtime/memory profiling comparison across models
- [ ] A second ML model (e.g. random forest) for a richer comparison
- [ ] Written report (`docs/report/`)
- [ ] AI-use reflection section in the report

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
