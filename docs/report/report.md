# Predicting Indoor CO2 Concentration from Sensor History and Building Occupancy

**MMA3001 — Numerical Methods and Machine Learning, Project Report**
**Dataset:** MMA3001 Dataset 2 — Monash Smart Infrastructure Occupancy and Environmental Data
**Repository:** https://github.com/tgup0010-bot/mma3001-occupancy-prediction

---

## 1. Engineering Problem

Indoor CO2 concentration is a standard proxy for ventilation adequacy: as CO2
rises above roughly 800–1000 ppm, it signals that fresh-air supply is not
keeping pace with occupant load, which affects both comfort and cognitive
performance. A building control system that can anticipate a rising CO2
trend — rather than only reacting once it is already elevated — can
pre-emptively increase ventilation, reducing both the lag in air-quality
response and the energy cost of running ventilation harder than needed.

**The problem this project solves:** given a sensor's recent CO2 history and
the building's current occupancy level, predict that sensor's CO2
concentration 15 minutes ahead. **The intended use** is as an early-warning
signal for a building management system: a rising predicted trend justifies
increasing ventilation before the level itself becomes a problem.

**Why regression, and why these methods.** The task above is a genuine
regression problem — a continuous quantity, not a class label — so it is
addressed with the four regression methods taught in MMA3001 Week 5
(Linear Regression, Decision Tree Regression, SVR, Neural Network
Regression), compared against each other on real data (§4), rather than a
single method chosen by preference.

**Limitations of the selected scope.** Only one environmental sensor
(device id `6012002000326`) has enough real, contiguous data to model
reliably — see §2 for why the other four are excluded. The model predicts
that single sensor's CO2, not a building-wide air-quality state.

## 2. Data: Inputs, Outputs, and Sources

**Source.** MMA3001 Dataset 2 (Monash Smart Infrastructure Occupancy and
Environmental Data), course-supplied: the 9.09M-row occupancy event log
(5 zones) and the environmental sensor log (5 sensors, nested-JSON payload
of Carbon dioxide, Temperature, Humidity, and other variables).

**Which sensor, and why only one.** Of the 5 environmental sensors, 2
(`6012002000227`, `6012002000777`) report every reading at the same frozen
timestamp — a stuck internal clock, not usable as a time series at all
(discovered while building this project; see §7). A third (`6012002000869`,
the one sensor traceable to a named room, `G.38`) has only a 4-month window
of data. That leaves two candidates with long real coverage
(`6012002000241`, `6012002000326`); `6012002000326` was chosen because it
was independently validated against real external weather data (its indoor
temperature and humidity correlate r=+0.84 and r=+0.65 respectively with
real Bureau of Meteorology readings for the nearest station across 222 real
overlapping days — see §7), giving confidence its readings are genuine
measurements, not sensor malfunction.

**Inputs (top-level).**

| Input | Format / units | Domain | Source |
|---|---|---|---|
| `co2_now` | float, ppm | typically 350–2000 ppm | sensor `6012002000326`, resampled to 15-min bins |
| `co2_lag1` | float, ppm | same as above | same sensor, one bin (15 min) earlier |
| `co2_rolling_1h` | float, ppm | same as above | trailing 4-bin (1h) mean of CO2 |
| `building_occupancy` | float, count | 0–5 (number of the 5 occupancy zones currently occupied) | aggregated from the occupancy event log |
| `hour_sin`, `hour_cos` | float, unitless | [-1, 1] | cyclical encoding of time-of-day |
| `dow_sin`, `dow_cos` | float, unitless | [-1, 1] | cyclical encoding of day-of-week |
| `is_weekend` | float (0.0/1.0) | {0, 1} | Saturday/Sunday flag |

**Output.** A single continuous value: predicted CO2 concentration (ppm) 15
minutes ahead of the input bin, in the same units and with the same
physical meaning as the measured quantity it predicts.

**Invalid, missing or unsupported inputs.** `occupancy.env_sensors` drops
JSON payload entries that carry no `value` field (a failed sensor read, not
a zero reading — discovered and handled explicitly; see §7).
`occupancy.co2_regression.build_co2_supervised_dataset` drops any row
missing a required lag/rolling feature or target rather than imputing one
— see §5 for how large this effect is and why it is not treated as an
error to hide.

## 3. Computational Solution

Four regression methods — the full set taught in MMA3001 Week 5 — are
implemented and compared (`occupancy.co2_models`), each as a
`scikit-learn` `Pipeline` with a shared `StandardScaler` preprocessing step
(essential for SVR and the neural network; harmless for the other two,
since ordinary least squares is scale-invariant in its predictions and a
decision tree's splits are invariant to any monotonic per-feature
transform — using one shared preprocessing step keeps the comparison fair).

1. **Linear Regression** (Week 5.1) — fits
   `CO2_next = β₀ + β₁·co2_now + β₂·co2_lag1 + ...` by ordinary least
   squares. Included as the simplest, fastest, most interpretable baseline.
2. **Decision Tree Regression** (Week 5.2) — a single regression tree
   (`max_depth=8`), splitting the feature space into if/else regions. Can
   capture interactions (e.g. "high occupancy AND afternoon") a linear
   model cannot.
3. **Support Vector Regression** (Week 5.3) — an RBF-kernel SVR
   (`C=10, epsilon=0.5`), fitting a smooth nonlinear function tolerant of
   small prediction errors within an epsilon-margin.
4. **Neural Network Regression** (Week 5.4) — a small `MLPRegressor`
   (two hidden layers, 32 and 16 units, early stopping), able to
   approximate more complex nonlinear relationships at the cost of being
   the least interpretable and most prone to overfitting on a dataset this
   size.

**Key assumption.** Building occupancy is deliberately included as a
*testable* feature, not assumed useful: §4 reports the result of fitting
every model both with and without it, since the project's own earlier
sensor-matching analysis (§7) found no statistical evidence any occupancy
zone shares this sensor's room.

**Implementation.** `occupancy.co2_regression.build_co2_supervised_dataset`
builds the supervised table; `scripts/co2_prediction_experiment.py` trains
and evaluates all four models against the real data end-to-end.

## 4. Alternative-Solutions Comparison and Validation Results

**Validation method.** The supervised table (17,010 rows after dropping
rows with missing lag/rolling/target values — see §5) was split
**chronologically** (train 80% / test 20% by time, never randomly) at
**11 March 2026**, exactly the discipline taught in Week 5.6 and already
established in this project's earlier occupancy work: a random split would
let a model see "future" CO2 values relative to some test points, inflating
every metric in a way that would not hold once deployed forward in time.

**Main result** (13,608 train / 3,402 test rows; metrics per Week 5.5 —
MAE and RMSE in ppm, R² relative to a mean-prediction baseline):

| Model | MAE (ppm) | RMSE (ppm) | R² |
|---|---|---|---|
| Linear Regression | 9.012 | 25.579 | 0.3581 |
| Decision Tree Regression | 9.353 | 28.708 | 0.1914 |
| **SVR** | **8.350** | 25.894 | 0.3422 |
| Neural Network Regression | 9.350 | 26.015 | 0.3360 |

**SVR has the lowest MAE** and is the model this project selects, though
the margin over Linear Regression and the Neural Network is modest;
Decision Tree Regression is clearly worst on every metric here, likely
overfitting the training period's specific split points rather than
learning a generalisable trend. An R² of ~0.34–0.36 means the models
explain roughly a third of the variance in next-15-minute CO2 — a genuine
but moderate result, not a strong one, and it is reported as such rather
than overstated.

**Does building occupancy actually help? Tested directly, not assumed:**

| Model | R² with occupancy | R² without occupancy | Difference |
|---|---|---|---|
| Linear Regression | 0.3581 | 0.3579 | +0.0002 (none) |
| Decision Tree Regression | 0.1914 | 0.3305 | −0.1391 (**hurts**) |
| SVR | 0.3422 | 0.3438 | −0.0016 (hurts, negligible) |
| Neural Network Regression | 0.3360 | 0.3352 | +0.0008 (none) |

**Occupancy does not improve CO2 prediction for any of the four methods** —
for the Decision Tree it measurably hurts (likely by giving the model an
irrelevant feature to spuriously split on). This is a real, direct answer
to the question this project set out to test, and it is consistent with
(and now confirmed by an actual predictive model, not only a correlation
check) the earlier finding in §7 that no occupancy zone could be
statistically linked to this sensor's room.

**What this validation cannot establish.** It confirms the model
generalises to a *later period from the same sensor*, under the same
deployment and building conditions. It says nothing about a different
sensor, a different building, or a period with a structurally different
occupancy pattern (e.g. a semester break). It also cannot rule out that
occupancy would help for a sensor genuinely co-located with the measured
zones — only that it does not for this sensor, which is the one this
project could validate.

## 5. Handling Missing Data

CO2 readings for sensor `6012002000326` are **not evenly spaced** and
contain real reporting outages, the largest spanning approximately 636
days. Two consequences were found and handled explicitly, both real bugs
caught while building this project rather than designed in from the start:

- **Supervised-table construction drops 79.9% of candidate rows**
  (84,576 → 17,010) because a valid `co2_lag1`/`co2_rolling_1h`/`target`
  needs several consecutive non-missing bins, and most of the calendar
  span sits inside gaps. This is reported plainly, following the same
  practice as the occupancy project's own missing-data handling (§7):
  dropping is preferred over inventing a value for a bin with no real
  reading.
- **Numerical integration (§6) initially bridged outages as if CO2 varied
  linearly across them** — a bug, not a design choice; see §6 for the fix.

## 6. Optimisation and Performance (Numerical Integration)

A CO2 sensor reports an *instantaneous* concentration, but a
ventilation-adequacy assessment may need the *accumulated* exposure over a
period — exactly the local-value-to-accumulated-quantity problem MMA3001
Week 6 addresses. `occupancy.co2_integration` implements the trapezoidal
rule and Simpson's 1/3 rule directly from the Week 6.1 formulas (not via
`scipy`, so the calculation is auditable against the taught method).

**A real bug, caught and fixed.** The first version resampled CO2 onto a
regular 15-minute grid and dropped the missing bins before integrating —
which looks like a clean uniform grid but silently treats readings either
side of a gap (including the 636-day outage) as one bin apart. Applying
Simpson's rule to this produced a wildly wrong answer (~1.1M ppm·h against
a ~9.5M ppm·h trapezoidal estimate on the same nominal data) because
Simpson's rule is far more sensitive to a violated uniform-spacing
assumption than trapezoidal is — this discrepancy is what surfaced the bug.
**The fix, in two parts:**

1. **A gap-aware trapezoidal variant**
   (`trapezoidal_integral_gap_aware`), mirroring
   `occupancy.preprocessing.resample_occupancy`'s own `max_gap` convention:
   panels spanning more than 2 hours are excluded rather than integrated
   across. Over the sensor's full ~20,720-hour nominal span, only
   ~5,228 hours (≈25%) are panels short enough to trust; the naive
   estimate (9,543,357 ppm·h) and the gap-aware estimate (2,305,836 ppm·h)
   differ by a factor of 4 — the outage-bridging assumption alone,
   with no measurement error involved.
2. **A fixed comparison window for trapezoidal-vs-Simpson convergence**:
   the longest genuinely gap-free run at 15-minute resolution (41 hours,
   1–3 October 2025) was used as one fixed interval, then re-sampled at
   15/30/60-minute resolution — refining the *same* interval, not
   independently choosing a different "longest run" at each resolution
   (an earlier version of this comparison made exactly that mistake).

**Result, over the fixed 41-hour window:**

| Bin size | n points | Trapezoidal (ppm·h) | Simpson (ppm·h) | Agreement |
|---|---|---|---|---|
| 15 min | 163 | 17,817.3 | 17,829.9 | 0.071% |
| 30 min | 83 | 18,019.8 | 18,019.9 | 0.001% |
| 60 min | 41 | 17,583.8 | 17,571.8 | 0.068% |

Trapezoidal and Simpson now agree to within 0.1% at every resolution —
strong evidence the fix is correct (Week 6.2's "agreement between methods"
convergence check). Refining from 60→30 min changes the estimate by
+2.42%, and 30→15 min by −1.14%: real, noisy sensor data does not follow
the clean `E(h) ≈ Ch^p` halving pattern of the notes' synthetic examples
exactly, which is itself consistent with Week 6.2's own point that
disagreement can come from sampling or measurement effects, not only
quadrature error.

**Note on 5-minute bins:** deliberately excluded from this comparison —
checked directly, the sensor's real reporting cadence (~10 minutes) means
the longest gap-free run at 5-minute resolution is 2 points, too short to
integrate at all. This is a finding about the sensor, not a limitation of
the method.

## 7. Limitations and Lessons Learned

- **This project's regression result is moderate, not strong (R² ≈ 0.34)
  and occupancy does not help.** Reported honestly rather than
  reframed — a partially successful method, validated properly, is valid
  engineering work (per the unit's own framing).
- **Only 1 of 5 environmental sensors could be modelled at all.** Two
  report every reading at the same frozen timestamp (a stuck internal
  clock, not usable as a time series); a third has only a 4-month window.
  This was not previously documented and is worth flagging to the unit.
- **A preliminary correlation check found no evidence of a room-level
  link between occupancy zones and environmental sensors.**
  `scripts/sensor_matching_analysis.py` tested whether any occupancy
  zone's pattern correlated with any environmental sensor's
  CO2/temperature/humidity *rate of change* (raw levels were avoided
  deliberately — they share a building-wide diurnal cycle that would make
  every zone look "related" to every sensor). Result: no pair showed a
  correlation strong enough to count as a discovered match
  (|r| ≤ 0.06 for CO2 throughout) — the direct precedent for §4's finding
  that occupancy does not help predict this sensor's CO2 either.
- **An external-data comparison validated sensor `6012002000326`
  specifically.** Real Bureau of Meteorology data for the nearest station
  (Moorabbin Airport) was fetched (`scripts/fetch_bom_weather.py` — BoM's
  bulk-download endpoint actively blocks automated access, confirmed
  directly, so only their public rolling ~15-month archive is used) and
  compared against each sensor's indoor readings
  (`scripts/weather_comparison_analysis.py`). Only this one sensor had
  overlapping dates; its indoor temperature/humidity correlated
  r=+0.84/+0.65 with real outdoor readings across 222 days — the basis for
  choosing it for the CO2 regression task in §2.
- **Filenames should not be trusted over the data.** The occupancy log
  named "MayToDec2024" actually spans November 2023 to April 2026,
  discovered by directly profiling timestamps.

## 8. AI-Use Reflection

*(Note to reader: this section is a first draft describing how AI was
actually used while building this repository. It should be reviewed and,
where necessary, corrected by the student before submission, since an
accurate account of one's own understanding and verification process
cannot be fully authored by the tool that assisted with the work.)*

**Tools used:** Claude (Claude Code), throughout the project's data
exploration, coding, analysis, and documentation phases.

**What it was used for:** exploring and profiling the raw occupancy and
environmental sensor files (schema checks, data-quality checks that
surfaced the frozen-clock sensors in §2/§7); reading the unit's Week 5
and Week 6 course material directly to confirm the exact regression and
integration methods and formulas to implement; scaffolding the Python
package (`src/occupancy`), its tests, and the experiment scripts;
fetching and validating external Bureau of Meteorology data for sensor
selection; implementing and running the four-model comparison and the
Week 6 integration analysis end-to-end against the real data; finding and
fixing the real bugs described in §§6–7; and drafting this report's prose,
grounded in the numbers the code actually produced.

**Approximate level of contribution:** high for code scaffolding,
running large-file analyses, and drafting prose. The underlying
engineering and scope decisions were made in direct back-and-forth with
the student throughout: the student directed the choice of problem
framing (CO2 prediction from occupancy and sensor history), which sensor
to model and why, which of the four Week 5 methods to compare, and
reviewed and pushed back on intermediate findings (e.g. explicitly
questioning whether a given check was actually sufficient evidence)
rather than accepting AI output at face value.

**Why AI was used for these tasks:** profiling multiple large files and
cross-referencing course material against code by hand is slow and
error-prone; running the same checks in code is auditable — anyone can
re-run the scripts in `scripts/` and get the same numbers.

**How AI-generated material was checked:** every numeric claim in this
report was produced by actually running the corresponding script against
the real dataset in this session — `reports/co2_prediction_results.json`,
`reports/co2_integration_results.json`,
`reports/sensor_matching_results.json`, and
`reports/weather_comparison_results.json` are the raw evidence behind
§§4–7. All code is covered by the accompanying pytest suite (62 tests,
all passing) and was run against real data throughout, which is what
caught the bugs described below.

**Errors/limitations of AI assistance encountered:** several real bugs
were introduced and then caught by running against real data rather than
trusting synthetic tests alone: a sensor id read as a number instead of a
string, which silently broke label matching; an unhandled
failed-sensor-read JSON shape that crashed the parser; an inverted
lag-sign convention caught only because a unit test's expected direction
didn't match the actual output; and, most significantly, the Simpson's-
rule integration bug in §6 — an early version of the Week 6 convergence
comparison also independently picked the "longest gap-free run" at each
bin size separately, which silently compared different calendar windows
at different resolutions, making the refinement comparison meaningless.
This was caught by inspecting the actual date ranges printed by the
script, not by the code appearing to run without error, and fixed by
fixing one window and refining only that.

**Decisions that remained the student's:** dataset and sensor selection
and their justification; the specific engineering framing (air-quality
early-warning) and its importance; which of the four regression methods
to compare and why; acceptance of the single-sensor scope as a documented
limitation rather than an unstated gap; and final review of all numeric
results before they were written into this report.

## References

- MMA3001 Project Brief, Monash University, 2026.
- MMA3001 Project Datasets document, Monash University, 2026 (Dataset 2:
  Monash Smart Infrastructure Occupancy and Environmental Data).
- MMA3001 Notes, Week 5.1–5.6 (Linear Regression, Decision Tree
  Regression, Support Vector Regression, Neural Network Regression,
  Evaluating Regression Performance, Training Data Evaluation and
  Selection), Monash University, 2026.
- MMA3001 Week 6 (Integration) slides and notebooks — Newton-Cotes
  integration, error analysis, Romberg/Richardson extrapolation, Gaussian
  quadrature, multidimensional integration, Dr Keenan Granland, Monash
  University, 2026.
- Bureau of Meteorology, Daily Weather Observations, Moorabbin Airport
  (station 086077) — http://www.bom.gov.au/climate/dwo/
- scikit-learn documentation: `LinearRegression`, `DecisionTreeRegressor`,
  `SVR`, `MLPRegressor`, `StandardScaler`, model evaluation metrics.
