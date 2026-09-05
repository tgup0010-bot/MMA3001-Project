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

**Status: in progress.** See [`docs/report/`](docs/report) for the current
write-up once available, and the Roadmap section below for what's done.

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
├── src/occupancy/       Python package: data loading, preprocessing, models
├── tests/               pytest unit tests
├── data/raw/            Raw data (gitignored — see data/raw/README.md)
├── data/processed/      Small committed sample data, derived artefacts
├── docs/project_brief/  Unit-supplied project brief and dataset docs
├── docs/report/         Written project report (added once drafted)
└── reports/figures/     Generated plots/figures
```

## Setup

Requires Python ≥ 3.10.

```bash
pip install -e ".[dev]"
```

## Running the tests

```bash
pytest
```

## Generating code documentation

```bash
pdoc --output-dir docs/api src/occupancy
```

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
