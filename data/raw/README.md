# Raw data (not tracked in git)

These files are excluded from version control (see `.gitignore`) because they
are large (up to 1.5 GB) and are course-supplied data (MMA3001, Monash
University) rather than data we own or may redistribute publicly.

A small, representative sample derived from these files **is** committed at
[`data/processed/occupancy_sample_50k.csv`](../processed/occupancy_sample_50k.csv)
for fast development and unit testing.

To reproduce this folder, place the following files (as supplied in the
"MMA3001 Project Datasets" course materials) here:

| File | Approx. size | Description |
|---|---|---|
| `5occupancySensor_MayToDec2024_9MRows.csv` | 1.5 GB | Occupancy event log, 5 rooms, Nov 2023 – Apr 2026 |
| `5EnvSensor_MayToDec2024_180kRows.csv` | 337 MB | Environmental sensor readings (nested JSON payload), 5 sensors |
| `Sensor ID and Locations.xlsx` | 15 KB | Maps device/sensor IDs to physical rooms (partial coverage — see report, Limitations) |
| `sample_data/` | ~340 MB | Supplementary 2025–2026 batch (PMV, Detailed, Env_data_MSI_Lab) — **not used** in this project; columns for PMV/Detailed are undocumented (see report, Limitations) |

This project's committed code only depends on `5occupancySensor_MayToDec2024_9MRows.csv`
(and, for development, the small sample above).
