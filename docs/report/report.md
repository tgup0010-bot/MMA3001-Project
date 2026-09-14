# Short-Term Room Occupancy Prediction for Energy-Aware Building Control

**MMA3001 — Numerical Methods and Machine Learning, Project Report**
**Dataset:** MMA3001 Dataset 2 — Monash Smart Infrastructure Occupancy and Environmental Data
**Repository:** https://github.com/tgup0010-bot/mma3001-occupancy-prediction

---

## 1. Engineering Problem

Heating, cooling and lighting a room that nobody is using wastes energy;
reacting only *after* someone has already entered causes an uncomfortable
lag before conditions become comfortable. Building-management systems that
can anticipate occupancy a few minutes ahead — rather than only reacting to
a live sensor reading — can pre-condition a room just before it is needed
and confidently power down when it will stay empty. This is a standard,
practically important problem in building-services (mechanical)
engineering: occupancy-based HVAC and lighting control is one of the more
reliable low-cost energy-saving interventions available to an existing
building, precisely because it does not require any new hardware beyond
sensors that are frequently already installed.

**Problem statement.** Using the historical occupancy record of five rooms
in Monash's Smart Infrastructure ("Living Lab") building, predict whether a
given room will be occupied during the *next* 15-minute interval, using
only information available up to and including the current interval.

**Intended use.** The output is a per-room, per-interval occupancy
probability intended to feed a rule (e.g. "reduce conditioning if
P(occupied) < 0.2") in a building-management system. It is a decision-support
signal, not a safety-critical one — a wrong prediction costs comfort or a
small amount of energy, not safety, which is an important scoping
consideration when judging how much validation rigour is proportionate.

**Scope and its limits.** This project predicts occupancy from historical
occupancy patterns and calendar structure (time of day, day of week) alone.
It does **not** attempt to explain occupancy from environmental variables
(CO₂, temperature, etc.), and it does not attempt to identify *why* a room
is occupied. Section 7 explains why the environmental data was deliberately
excluded rather than used as a differentiator.

## 2. Data: Inputs, Outputs, and Sources

### 2.1 Source data

The raw input is an occupancy **event log**
(`data/raw/5occupancySensor_MayToDec2024_9MRows.csv`): one row is written
each time a room's sensor-reported occupancy status changes. It contains
9,091,732 rows across 5 rooms (`floorspaceid`, an opaque UUID), spanning
**24 November 2023 to 22 April 2026** — a materially wider window than the
filename ("MayToDec2024") suggests; this was confirmed directly from the
data rather than assumed from the filename. Each row records:

| Column | Type / domain | Meaning |
|---|---|---|
| `deviceid` | UUID (1 distinct value) | Sensor gateway device |
| `floorspaceid` | UUID (5 distinct values) | Room/zone identifier |
| `occupancystatus` | one of `CurrentlyOccupied` (92.5%), `RecentlyOccupied` (4.6%), `NotOccupied` (2.9%) | Sensor-reported state at `collecteddate` |
| `headcount` | integer ≥ 0 | Reported person count (unused in this project) |
| `collecteddate` | timestamp with UTC offset | When the state was recorded |
| `occupancystatuschangedate`, `previousoccupancystatus` | timestamp / status | Redundant with the above; unused |

A companion location spreadsheet (`Sensor ID and Locations.xlsx`) maps
device/room identifiers to physical rooms. Cross-referencing it against the
occupancy log showed only **2 of the 5 rooms** could be traced to a named
physical space: `G.20` and `G.25` ("2.4m — Keenan Lab"). The remaining 3 are
modelled as anonymous zones — the model still learns and predicts per-room
patterns for them, but the report cannot say which physical room they are.

### 2.2 Model inputs and outputs

- **Input:** for a given room and time `t`, the occupancy state at `t`, the
  occupancy state one bin before `t`, the mean occupancy rate over the
  preceding hour, and calendar features (hour-of-day and day-of-week,
  cyclically encoded; weekend flag) derived from `t`.
- **Output:** the predicted probability that the room is occupied
  (`CurrentlyOccupied`) during the *next* 15-minute bin, per room.
- **Invalid/missing input handling:** the raw log is event-driven, so most
  clock intervals have no row at all. `occupancy.preprocessing.resample_occupancy`
  converts this into a regular per-room grid, forward-filling a known state
  for up to a **2-hour** gap; any gap longer than that (e.g. a sensor
  outage) is left as an explicit missing value rather than assumed. Rows
  whose required feature history or target is missing are dropped before
  training/evaluation (never imputed) — see §5 for how much data this
  costs and why imputing was rejected as circular for this specific task.
  An `occupancystatus` value outside the three known states raises an
  error rather than being silently ignored (`occupancy.data_loading._validate`).

## 3. Computational Approach

Three predictors were implemented and compared, from simplest to most
informed:

1. **Persistence baseline** — predicts that the next interval repeats the
   current one. No fitting; a reference point for "doing nothing clever."
2. **Time-of-day Markov chain** (`occupancy.baselines.MarkovBaseline`) — a
   first-order Markov model whose transition probability
   `P(occupied_{t+1}=1 | room, hour-of-day, occupied_t)` is estimated by
   direct counting from training data. This is the classical
   numerical/statistical method in the comparison: no gradient-based
   fitting, its only "parameter" is the table of empirical transition
   rates. An unseen (room, hour, state) combination falls back to the
   overall training occupancy rate rather than failing.
3. **Logistic regression** (`occupancy.model.build_logistic_pipeline`) — a
   linear classifier over the engineered features plus a one-hot-encoded
   room identity, chosen as the first ML model for its interpretability and
   near-zero training cost, giving a clean reference for whether a more
   expensive model is worth it.
4. **Random forest** (`occupancy.model.build_random_forest_pipeline`, 100
   trees) — included as a second, non-linear ML model, to check whether
   feature interactions (e.g. "weekday AND afternoon") that a linear model
   cannot represent directly are actually being missed.

All four share the same feature pipeline
(`occupancy.features.build_supervised_dataset`), so differences in
predictive performance can be attributed to the model, not to inconsistent
inputs. Cyclical (sine/cosine) encoding was used for hour-of-day and
day-of-week so that, e.g., 23:00 and 00:00 are numerically adjacent rather
than maximally distant, which a raw integer encoding would imply.

**Key assumption.** "Occupied" is defined strictly as `CurrentlyOccupied`
only (`RecentlyOccupied` — a sensor cooldown state — counts as
unoccupied). This is a deliberate, documented choice
(`occupancy.preprocessing.resample_occupancy`'s `occupied_statuses`
parameter); a looser definition is a one-line change and was not explored
further given the project's scope.

## 4. Alternative-Solutions Comparison and Validation Results

**Validation method.** The supervised table was split **chronologically**
(never randomly) at the 80th percentile of timestamps — training on data up
to **15 October 2025**, testing only on data after that date. A random
split was deliberately rejected: it would let the model see occupancy
patterns from "the future" relative to some test points, inflating every
metric in a way that would not hold up once the model is actually deployed
forward in time. Because occupancy is imbalanced (~40% positive in the test
set, not evenly split), accuracy alone is not trusted — F1, ROC-AUC and the
Brier score (a calibration measure, lower is better) are all reported
together.

**Main result** (15-minute bins, 1-hour lookback; 342,401 resampled bins,
26.5% dropped for insufficient history — see §6; 201,216 training / 50,305
test rows):

| Model | Accuracy | F1 | ROC-AUC | Brier score ↓ |
|---|---|---|---|---|
| Persistence baseline | 0.816 | 0.771 | 0.809 | 0.184 |
| Markov (time-of-day) baseline | 0.817 | 0.772 | 0.870 | 0.139 |
| **Logistic regression** | **0.825** | **0.781** | **0.890** | **0.128** |
| Random forest (100 trees) | 0.811 | — | 0.860 | — |

Logistic regression beats every baseline on every metric, including
calibration — genuine evidence that the engineered features (not just "the
current state" or "the time of day" alone) carry real signal. The random
forest, despite being the more complex model, performs *worse* than
logistic regression here (§6 discusses cost as well as accuracy). This
result is reported honestly rather than dropped: a more complex model is
not automatically a better one, and for this feature set the added
capacity likely fits noise rather than structure the simpler features
don't already capture (`occupied_now` alone is already a very strong
predictor, so the problem is closer to linear than a tree ensemble is
suited for).

**What this validation cannot establish.** It confirms the model
generalises to a *later time period in the same rooms*, under the same
sensor deployment. It says nothing about generalising to a different
building, a different room's usage pattern not represented in training, or
a period with a structurally different usage pattern (e.g. a semester break
not present in the training window). It also cannot validate accuracy for
the 3 rooms whose physical identity is unknown, beyond the statistical
sense already reported.

## 5. Handling Missing Data

Two independent sources of missingness were encountered and handled
explicitly rather than silently:

1. **Sensor gaps in the raw log.** `resample_occupancy`'s `max_gap`
   parameter (2 hours) means any real outage longer than this appears as
   `NaN` in the resampled grid rather than an assumed, possibly wrong,
   forward-filled state. Per-room missingness on the full dataset ranged
   from **9.5% to 36.1%** of resampled bins (`coverage_summary`,
   `reports/experiment_results.json`) — reported honestly rather than
   hidden, since these numbers should inform how much the resulting model
   should be trusted for the noisier rooms.
2. **Feature-history gaps.** A row needs its previous bin, a full rolling
   window, and a defined next-bin target to be usable for supervised
   learning; 26.5% of the resampled grid was dropped for this reason
   (`build_supervised_dataset`'s `info` dict). These rows were **not**
   imputed: imputing a missing occupancy value in order to then predict
   occupancy would make the exercise partly circular, so the more
   defensible (if data-costlier) choice was to only train and evaluate on
   genuinely observed transitions.

## 6. Optimisation and Performance (Sensitivity Analysis)

Three design choices were quantitatively examined against the real full
dataset (`scripts/sensitivity_analysis.py`, `reports/sensitivity_results.json`),
rather than picked by preference.

### 6.1 Time-bin resolution

| Bin size | Grid memory | Logistic reg. accuracy / ROC-AUC | RF fit time | RF accuracy / ROC-AUC |
|---|---|---|---|---|
| 5 min | 103.8 MB | 0.859 / 0.922 | 55.8 s | 0.834 / 0.892 |
| 15 min | 34.6 MB | 0.825 / 0.890 | 11.9 s | 0.811 / 0.860 |
| 30 min | 17.3 MB | 0.799 / 0.859 | 4.3 s | 0.791 / 0.853 |
| 60 min | 8.7 MB | 0.761 / 0.831 | 1.8 s | 0.760 / 0.830 |

Accuracy improves monotonically as the bin gets finer — finer resolution
genuinely captures more real structure — but at a real cost: 5-minute bins
need **3×** the memory of 15-minute bins and take the random forest
**~30×** longer to fit, for a ROC-AUC gain of only ~0.03 over 15-minute
bins. **15 minutes was kept as the working resolution** for the main
results as the more defensible middle ground for a decision-support signal
that does not need 5-minute precision; a deployment prioritising maximum
accuracy over training cost could reasonably choose 5-minute bins instead —
this is a genuine trade-off, not a uniquely correct answer.

### 6.2 History (lookback window) length

| Rolling window | Logistic reg. accuracy / ROC-AUC | % rows dropped |
|---|---|---|
| 30 min (2 bins) | 0.827 / 0.883 | 23.9% |
| **1 h (4 bins)** | **0.825 / 0.890** | 26.5% |
| 2 h (8 bins) | 0.821 / 0.888 | 31.8% |
| 4 h (16 bins) | 0.815 / 0.883 | 39.3% |

Longer lookback windows do **not** improve ROC-AUC beyond the 1-hour
setting, while steadily discarding more data (up to 39.3% at 4 hours,
since more consecutive bins must all be present). The 1-hour window used
in the main result is close to the best ROC-AUC obtained across the sweep
while keeping data loss lower than the longer settings — a justified
choice, not an arbitrary default.

### 6.3 Model cost comparison

| | Logistic regression | Random forest (100 trees) |
|---|---|---|
| Parameters / structure | 14 coefficients | ~4.0M tree nodes (avg depth 29.2) |
| Fit time (201k rows) | 0.23 s | 11.9 s (~52×) |
| Predict time per row | 0.33 µs | 16.6 µs (~50×) |
| Test accuracy / ROC-AUC | 0.825 / 0.890 | 0.811 / 0.860 |

The random forest is roughly **50× more expensive** to both train and run,
for **worse** accuracy on this task. This is a clean, concrete
justification for selecting logistic regression as the final recommended
model: it is simultaneously cheaper and more accurate here, so there is no
accuracy/cost trade-off to weigh — logistic regression dominates on both
axes for this problem. That will not be true of every occupancy-prediction
problem, and is reported as a property of this feature set and dataset,
not a general claim that simpler models are always better.

## 7. Limitations and Lessons Learned

- **Environmental data was deliberately not used for occupancy prediction.**
  The environmental sensor CSV's 5 sensors were cross-referenced against
  the same location spreadsheet; only 1 could be matched to a physical
  room (`G.38`), and it does not coincide with either of the two occupancy
  rooms that *could* be identified (`G.20`, `G.25`). Joining the two
  datasets by assuming a room correspondence between the remaining
  unmatched sensors and zones would have been an unverifiable guess — this
  was decided *before* writing any modelling code for that direction,
  based on directly inspecting the mapping spreadsheet.
- **A data-driven sensor-matching attempt was made, and found no
  supporting evidence for a hidden mapping.** This scope decision was
  raised on the unit's EdStem forum ("irming scope #81"); Keenan Granland
  suggested trying to link sensors to zones by matching the *data itself*
  rather than the (unfixable) metadata. This was attempted directly:
  `scripts/sensor_matching_analysis.py` correlates each occupancy zone's
  binary occupied/vacant signal against each environmental sensor's
  *rate of change* (not raw level — see below) in Carbon dioxide,
  Temperature and Humidity, searching lags of up to 1 hour in either
  direction to allow for HVAC/room-mixing delay. Raw levels were
  deliberately not used for this test: CO2 and temperature both follow a
  building-wide diurnal cycle (busier, warmer during work hours in every
  room), so correlating levels would find every zone "related" to every
  sensor through that shared confound rather than genuine co-location;
  the first difference is far more specific to actual arrival/departure
  events.

  Full results are in `reports/sensor_matching_results.json`; the
  headline finding is negative. Carbon dioxide — physiologically the most
  direct occupancy signal — showed essentially no relationship with any
  zone (\|r\| ≤ 0.06 for every zone×sensor pair, indistinguishable from
  noise). Temperature showed a somewhat higher correlation for some pairs
  (up to r = 0.27), but this is consistent with the shared building-wide
  climate-control confound described above rather than room-specific
  co-location, since it appears for multiple unrelated zone/sensor
  combinations rather than concentrating on one. **No zone×sensor pair
  showed correlation strong and specific enough to justify treating it as
  a discovered room match.** This is a genuine, if negative, result: it
  turns the original scoping decision from "we assumed no relationship
  exists because the metadata doesn't confirm one" into "we tested for a
  relationship and did not find one" — a stronger basis for the
  occupancy-only scope than the metadata gap alone.

  The analysis also surfaced an independent data-quality problem in the
  environmental sensor log: 2 of the 5 sensors (`6012002000227`,
  `6012002000777`) report **every one of their ~10,000–21,000 readings at
  the exact same timestamp** (a stuck internal clock, not a real reading
  cadence), leaving them with no usable time series at all. This was not
  previously documented and is worth flagging to the unit separately from
  the room-mapping issue.
- **An external-data comparison (Keenan's third suggestion) validated one
  sensor's plausibility, and confirms the other four can't be checked this
  way.** BoM's bulk historical-download service actively blocks automated
  requests (confirmed directly — it returns an explicit anti-scraping
  refusal); only their public monthly Daily Weather Observations pages are
  fetchable, and only for a rolling ~15-month window
  (`scripts/fetch_bom_weather.py`, real data for Moorabbin Airport, station
  086077 — the nearest official BoM station to Monash Clayton). Checking
  which sensors' own data actually falls inside that window: only sensor
  `6012002000326` does (the two stuck-clock sensors and `G.38` have no
  overlap at all; `6012002000241` ends two months too early). For that one
  sensor, indoor temperature correlates **r = +0.84** and indoor humidity
  **r = +0.65** with real outdoor Moorabbin readings across 222 overlapping
  days (`reports/weather_comparison_results.json`) — a strong, expected
  result (a climate-controlled indoor space should track outdoor weather
  loosely, not tightly or not at all) that supports this sensor's readings
  being genuine and sane, not corrupted like its two frozen-timestamp
  siblings. The other four sensors simply have no real external data
  available to check them against, which is itself worth stating plainly
  rather than leaving unexamined.
- **3 of 5 occupancy zones have no known physical identity.** The model
  still learns and is evaluated per-zone for these, but the report cannot
  contextualise their results physically (e.g. "this is a lecture theatre
  vs. an office").
- **A real bug was caught by testing against real data, not just synthetic
  cases.** Melbourne's daylight-saving transition means the raw
  timestamps mix UTC offsets (`+11`/`+10`); parsing them with pandas'
  default `parse_dates` silently produced an unusable `object`-typed
  column instead of raising an error. This was only caught by running the
  pipeline against a real data sample (`data/processed/occupancy_sample_50k.csv`)
  in addition to the synthetic unit tests, and is now fixed and covered by
  the pipeline's design (`occupancy.data_loading._parse_dates`, parsing to
  UTC first). The lesson generalises: synthetic unit tests validate logic,
  but only real data exposes real-world data-quality issues — both are
  necessary, neither is sufficient alone.
- **A more complex model is not automatically better.** The random forest
  under-performing logistic regression on both accuracy and cost was an
  unplanned, and initially counter-intuitive, result. It is reported as
  found rather than tuned-away, since honestly reporting a negative result
  with a plausible explanation is more valuable than presenting only
  favourable numbers.
- **Filenames should not be trusted over the data.** The raw file named
  "MayToDec2024" actually spans November 2023 to April 2026 — confirmed by
  directly profiling the timestamps rather than trusting the filename,
  which materially changed the scoping of what data was actually available.

## 8. AI-Use Reflection

*(Note to reader: this section is a first draft describing how AI was
actually used while building this repository. It should be reviewed and,
where necessary, corrected by the student before submission, since an
accurate account of one's own understanding and verification process
cannot be fully authored by the tool that assisted with the work.)*

**Tools used:** Claude (Claude Code), across two phases: (1) the original
build of the occupancy pipeline, models, and report; (2) a later session
following up on a data-scoping question raised on the unit's EdStem forum
("irming scope #81"), which produced the two supplementary analyses in
§7 (sensor-matching and the BoM weather comparison) and this section's
rewrite.

**What it was used for, phase 1:** exploring and profiling the raw
CSV/XLSX files (row counts, date ranges, schema checks, cross-referencing
the location spreadsheet against the actual sensor IDs present in the
data); scaffolding the Python package (`src/occupancy`), its tests, and
the experiment scripts; drafting docstrings and this report's prose,
grounded in the numbers the code actually produced.

**What it was used for, phase 2:** after the student asked Keenan Granland
directly whether the occupancy-only scope was sufficient, and Keenan
suggested three alternative directions on the forum, the student asked
Claude to investigate whether any of them could be added. This involved:
inspecting the raw environmental sensor JSON and the location spreadsheet
directly (rather than assuming); building and running the sensor-matching
correlation analysis (§7) end-to-end against the real 9M-row and 180k-row
files; separately researching and fetching real Bureau of Meteorology data
for the nearest station once BoM's bulk-download endpoint turned out to
block automated access; checking a third idea (predicting sensor
maintenance from battery-voltage degradation) against the real data before
it was built, finding no usable signal, and not implementing it as a
result; and rewriting this section and §7 to reflect all of the above.

**Approximate level of contribution:** high for code scaffolding,
boilerplate, and the mechanics of running large-file analyses (package
structure, test structure, docstring formatting, data fetching). The
underlying scope decisions were the student's throughout both phases, and
in phase 2 specifically the student: raised the original scoping question
independently on the forum before involving Claude; chose the
sensor-matching analysis over a full pivot to an alternative project,
after being shown the trade-offs; explicitly questioned whether a
negative-result check was actually "enough" against what Keenan's
suggestions implied, rather than accepting it at face value; requested the
BoM comparison as a genuinely separate follow-up; and rejected the
sensor-maintenance-prediction idea after seeing the real battery-voltage
data showed no meaningful degradation trend, rather than having it built
anyway. That sequence of the student pushing back on scope and asking
"is this actually sufficient" is itself part of the record here, not
smoothed over.

**Why AI was used for these tasks:** profiling multiple large
(100 MB–1.5 GB) files by hand is slow and error-prone; having the same
checks (schema validation, cross-referencing IDs, fetching and parsing
external data) run in code rather than manually is also more auditable —
anyone can re-run the scripts in `scripts/` and get the same numbers.

**How AI-generated material was checked:** every numeric claim in this
report was produced by actually running the corresponding script against
the real dataset in this session (not generated from memory or estimated)
— `reports/experiment_results.json`, `reports/sensitivity_results.json`,
`reports/sensor_matching_results.json`, and
`reports/weather_comparison_results.json` are the raw evidence backing
§§4–7. All code additions were exercised by the accompanying pytest suite
(49 tests, currently all passing — up from 26 after phase 1) before being
trusted, and the pipeline was additionally run against real (not only
synthetic) data throughout, which is what caught every bug described
below.

**Errors/limitations of AI assistance encountered:** four real bugs were
introduced and then caught by running against real data rather than
trusting synthetic tests alone: (1) the timestamp-parsing bug in §7
(assuming `pandas.read_csv`'s `parse_dates` would produce one consistent
tz-aware column across a daylight-saving transition, which failed silently
instead of raising); (2) in phase 2, a sensor identifier being read as a
number instead of a string, which silently broke both the sensor-to-room
label lookup and the results' JSON export; (3) unhandled "failed read"
JSON entries in the environmental sensor log (a reading with no actual
value), which crashed the parser until handled explicitly; (4) an inverted
lag-sign convention in the correlation search, caught only because a unit
test's expected direction didn't match the actual output. None of these
were caught by code review alone — each was only found by running the
code against real data or a concrete test case and checking the actual
output.

**Decisions that remained the student's:** dataset selection, the specific
engineering framing (energy-aware HVAC control) and its justification,
acceptance of the room-mapping limitation as a reason to scope out the
environment data, GitHub repository visibility/ownership, final review of
all numeric results before they were written into this report, and — in
phase 2 — the decision of which of Keenan's three suggested directions to
pursue, how far to take each one, and the decision to stop rather than
build a fourth analysis (sensor maintenance) once the data didn't support
it.

## References

- MMA3001 Project Brief, Monash University, 2026.
- MMA3001 Project Datasets document, Monash University, 2026 (Dataset 2:
  Monash Smart Infrastructure Occupancy and Environmental Data).
- Pedregosa, F. et al. (2011). *Scikit-learn: Machine Learning in Python*.
  Journal of Machine Learning Research, 12, 2825-2830.
- McKinney, W. (2010). *Data Structures for Statistical Computing in
  Python*. Proceedings of the 9th Python in Science Conference.
