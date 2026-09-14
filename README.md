# MMA3001 Project — Room Occupancy Pattern Modelling

[![CI](https://github.com/tgup0010-bot/mma3001-occupancy-prediction/actions/workflows/ci.yml/badge.svg)](https://github.com/tgup0010-bot/mma3001-occupancy-prediction/actions/workflows/ci.yml)

Individual project for MMA3001 (Numerical Methods and Machine Learning),
Monash University, 2026 S2.

## Engineering problem

Predicting whether a room will be occupied in the near future lets a
building's HVAC and lighting systems scale back conditioning in empty rooms
and pre-condition ahead of expected arrivals — reducing energy waste without
harming comfort. This project builds and validates a short-term occupancy
prediction model from historical sensor data collected in Monash's Smart
Infrastructure ("Living Lab") building.

**Status: complete.** Data pipeline, two baselines, two ML models,
chronological validation, and a sensitivity/optimisation analysis are all
implemented and run end-to-end against the real dataset (see
[Current results](#current-results) below). The full written report is at
[`docs/report/MMA3001_Project_Report.docx`](docs/report/MMA3001_Project_Report.docx)
(also available as [Markdown](docs/report/report.md)).

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
  not be validated against the supplied metadata, and joining the two by
  assuming a mapping was deliberately **not** attempted.
- **A data-driven alternative was tried instead of the metadata join**
  (`scripts/sensor_matching_analysis.py`): correlating each zone's
  occupancy pattern against each sensor's readings directly, to see if the
  data itself reveals a room correspondence the spreadsheet couldn't. It
  found no evidence of one — see the report §7 and
  [`reports/sensor_matching_results.json`](reports/sensor_matching_results.json)
  for the full results and method. This also surfaced an unrelated
  data-quality issue: 2 of the 5 environmental sensors report every
  reading at the same stuck timestamp.
- A supplementary "Sample Data" batch (`PMV`, `Detailed` files) has no
  column headers and, as of writing, no clarification has been published
  on EdStem explaining what they represent. It is not used.

## Repository structure

```
├── src/occupancy/       Python package: data loading, preprocessing,
│                        feature engineering, baselines, ML models, evaluation,
│                        environmental sensor loading, sensor-matching analysis
├── scripts/             demo.py (live demo), run_experiment.py,
│                        sensitivity_analysis.py, train_and_save_model.py,
│                        sensor_matching_analysis.py
├── models/              Pre-trained model + demo examples (committed, tiny)
│                        — lets the demo run instantly, no raw data needed
├── tests/               pytest unit tests (41, all passing)
├── data/raw/            Raw data (gitignored — see data/raw/README.md)
├── data/processed/      Small committed sample data, derived artefacts
├── docs/project_brief/  Unit-supplied project brief and dataset docs
├── docs/api/            Generated HTML code documentation (pdoc)
├── docs/report/         Written project report (.docx submission + .md source)
└── reports/             Experiment + sensitivity-analysis results, test report
```

## Setup

Requires Python ≥ 3.10.

```bash
pip install -e ".[dev]"
```

## Live demo (for the presentation)

```bash
python scripts/demo.py
```

This is the thing to run live in the presentation/interview. It:

1. **Loads a pre-trained model** from `models/occupancy_model.joblib`
   (committed to the repo — starts in under a second, no need for the
   raw 1.5GB file or to retrain anything).
2. **Replays 15 real examples** from the test set (data the model never
   trained on) — for each one, prints the room, the time, whether it was
   occupied right now, what the model predicted, and what actually
   happened. This is the live "does it actually work" evidence.
3. **Lets you type in your own scenario** — pick a room, a time of day, a
   day of week, and whether it's currently occupied — and the model
   prints its live prediction. This is the "tune it yourself" part.

If `models/occupancy_model.joblib` is ever missing or out of date, regenerate
it (needs the raw CSV in `data/raw/`, takes ~2-3 minutes):

```bash
python scripts/train_and_save_model.py
```

## Running the tests

```bash
pytest                                          # 28 unit tests
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

```bash
python scripts/sensitivity_analysis.py
```

Re-runs the pipeline across a sweep of time-bin resolutions (5/15/30/60
min) and lookback-window lengths (30 min/1 h/2 h/4 h), fitting both the
logistic regression and random forest models at each setting, and records
accuracy, ROC-AUC, fit/predict time, and model size at each point. Results
are written to `reports/sensitivity_results.json`. This is the evidence
behind the optimisation section of the report (§6).

### Current results

On a chronological holdout (train up to 2025-10-15, test after), out of
342,401 resampled 15-minute bins (26.5% dropped for missing lag/rolling
history — see `data_info` in `reports/experiment_results.json`):

| Model | Accuracy | F1 | ROC-AUC | Brier score (↓ better) |
|---|---|---|---|---|
| Persistence baseline | 0.816 | 0.771 | 0.809 | 0.184 |
| Markov (time-of-day) baseline | 0.817 | 0.772 | 0.870 | 0.139 |
| **Logistic regression** | **0.825** | **0.781** | **0.890** | **0.128** |
| Random forest (100 trees) | 0.811 | — | 0.860 | — |

Logistic regression outperforms every baseline **and** the more complex
random forest on every metric, while being ~50× cheaper to fit and to run
per prediction (see the report §6.3) — a genuine, evidence-based
justification for the final model choice, not a default pick. Full
sensitivity analysis (bin size, lookback window) and the model-cost
comparison are in [`docs/report/MMA3001_Project_Report.docx`](docs/report/MMA3001_Project_Report.docx), §6.

## Roadmap

- [x] Data loading + schema validation
- [x] Event log → regular time-grid preprocessing (with explicit max-gap
      handling of long sensor outages)
- [x] Feature engineering (calendar cyclical features, lag, rolling mean)
- [x] Persistence baseline
- [x] Time-of-day Markov chain baseline
- [x] Logistic regression ML model
- [x] Random forest ML model (second, richer comparison)
- [x] Chronological validation split + evaluation metrics
- [x] Sensitivity analysis: bin size and lookback-window trade-offs
- [x] Runtime/model-complexity comparison across models
- [x] Data-driven environmental-sensor-to-zone matching attempt (negative
      result, documented — see report §7)
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
