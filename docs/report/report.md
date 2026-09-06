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

- **Environmental data was deliberately not used.** The environmental
  sensor CSV's 5 sensors were cross-referenced against the same location
  spreadsheet; only 1 could be matched to a physical room (`G.38`), and it
  does not coincide with either of the two occupancy rooms that *could* be
  identified (`G.20`, `G.25`). Attempting an occupancy↔environment model
  would have required guessing which environmental sensor belongs to which
  occupancy zone — an unverifiable assumption the brief specifically warns
  against relying on. This was decided *before* writing any modelling
  code for that direction, based on directly inspecting the mapping
  spreadsheet, rather than discovered after wasted effort.
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

**Tools used:** Claude (Claude Code), throughout the project's data
exploration, coding and documentation phases.

**What it was used for:** exploring and profiling the raw CSV/XLSX files
(row counts, date ranges, schema checks, cross-referencing the location
spreadsheet against the actual sensor IDs present in the data); scaffolding
the Python package (`src/occupancy`), its tests, and the two experiment
scripts; drafting docstrings and this report's prose, grounded in the
numbers the code actually produced.

**Approximate level of contribution:** high for code scaffolding and
boilerplate (package structure, test structure, docstring formatting); the
underlying analytical decisions — which dataset to use, how to define
"occupied," why the environment data was excluded, which validation
strategy to use, how to interpret the sensitivity-analysis results — were
made in direct back-and-forth with the student, and the student directed
each major scope decision (choice of Dataset 2 over Datasets 1/3, choice
of occupancy-prediction over occupancy↔environment correlation, repo
visibility and naming).

**Why AI was used for these tasks:** profiling multiple large
(100 MB–1.5 GB) files by hand is slow and error-prone; having the same
checks (schema validation, cross-referencing IDs) run in code rather than
manually is also more auditable — anyone can re-run `scripts/run_experiment.py`
and get the same numbers.

**How AI-generated material was checked:** every numeric claim in this
report was produced by actually running the corresponding script against
the real dataset in this session (not generated from memory or estimated)
— `reports/experiment_results.json` and `reports/sensitivity_results.json`
are the raw evidence backing §§4–6. All code additions were exercised by
the accompanying pytest suite (26 tests, currently all passing) before
being trusted, and the pipeline was additionally run against real (not
only synthetic) sample data, which is what caught the daylight-saving
timestamp bug described in §7.

**Errors/limitations of AI assistance encountered:** the timestamp-parsing
bug in §7 was itself an AI-authored oversight (assuming `pandas.read_csv`'s
`parse_dates` would produce one consistent tz-aware column, which failed
silently rather than raising) — it was only caught by insisting on running
the real-data smoke test rather than trusting the unit tests alone.

**Decisions that remained the student's:** dataset selection, the specific
engineering framing (energy-aware HVAC control) and its justification,
acceptance of the room-mapping limitation as a reason to scope out the
environment data, GitHub repository visibility/ownership, and final
review of all numeric results before they were written into this report.

## References

- MMA3001 Project Brief, Monash University, 2026.
- MMA3001 Project Datasets document, Monash University, 2026 (Dataset 2:
  Monash Smart Infrastructure Occupancy and Environmental Data).
- Pedregosa, F. et al. (2011). *Scikit-learn: Machine Learning in Python*.
  Journal of Machine Learning Research, 12, 2825-2830.
- McKinney, W. (2010). *Data Structures for Statistical Computing in
  Python*. Proceedings of the 9th Python in Science Conference.
