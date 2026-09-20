# Predicting Indoor CO2 Concentration from Sensor History and Building Occupancy

**MMA3001 — Numerical Methods and Machine Learning, Project Report**
**Dataset:** MMA3001 Dataset 2 — Monash Smart Infrastructure Occupancy and Environmental Data
**Repository:** https://github.com/tgup0010-bot/mma3001-occupancy-prediction

---

## 1. Engineering Problem

Poor indoor air quality is a real problem in office buildings. CO2 builds up when there are people in a room and not enough fresh air coming in, and once it gets above about 800–1000 ppm, people start feeling less alert and uncomfortable. Most building systems only react after the CO2 is already too high — by the time the ventilation ramps up, occupants have already been sitting in stale air.

The goal here is to predict what the CO2 level will be 15 minutes from now, based on what it has been doing in the last hour and how many people are in the building. If the model can see a rising trend before it becomes a problem, the building management system could start increasing ventilation a bit earlier, which is both better for occupants and uses less energy than playing catch-up.

This is a regression task — predicting a number, not a category — so the four regression methods from MMA3001 Week 5 were all tried and compared against each other: Linear Regression, Decision Tree Regression, SVR, and Neural Network Regression. The point was to see which one actually works best on this real dataset, not just pick one by gut feel.

One thing to note upfront: only one of the five environmental sensors in the dataset had enough usable, contiguous data to be worth modelling. Section 2 explains why the other four were ruled out.

## 2. Data: Inputs, Outputs, and Sources

**Where the data comes from.** This project uses MMA3001 Dataset 2 — the Monash Smart Infrastructure occupancy and environmental sensor dataset. It has two parts: an occupancy event log covering 5 zones, and a log of readings from 5 environmental sensors, each reporting CO2, temperature, humidity, and a few other variables.

**Why only one sensor was used.** Two of the five sensors (`6012002000227`, `6012002000777`) turned out to be completely broken — every single reading has the same timestamp, meaning the internal clock was frozen. They look like data but are actually useless for any time-series work. A third sensor (`6012002000869`) is the only one with a documented room location (Room G.38), but it only has about four months of readings, which is not enough to train and test a model properly.

That left two sensors with long, real coverage. Sensor `6012002000326` was chosen because it could be cross-checked against actual weather data — its indoor temperature and humidity readings correlate at r = +0.84 and r = +0.65 with Bureau of Meteorology records from Moorabbin Airport across 222 overlapping days. That level of agreement is strong evidence the sensor is working correctly, not just reporting noise.

**Inputs used to make each prediction:**

| Input | Units | What it represents |
|---|---|---|
| `co2_now` | ppm | Current CO2 reading |
| `co2_lag1` | ppm | CO2 reading from 15 minutes ago |
| `co2_rolling_1h` | ppm | Average CO2 over the past hour |
| `building_occupancy` | count (0–5) | How many of the 5 occupancy zones currently have people |
| `hour_sin`, `hour_cos` | — | Time of day, encoded cyclically |
| `dow_sin`, `dow_cos` | — | Day of week, encoded cyclically |
| `is_weekend` | 0 or 1 | Whether it is Saturday or Sunday |

**Output:** A single predicted CO2 value (ppm) 15 minutes ahead.

**Missing readings.** Entries in the sensor log that have no actual value (a failed read, not a zero) are dropped before the dataset is built. Any row that is missing a lag feature, rolling average, or target value is also dropped rather than filled in with a made-up number — see Section 5 for the scale of this.

## 3. Computational Solution

All four regression methods from Week 5 are implemented in `occupancy.co2_models`, each as a scikit-learn Pipeline. All of them go through the same StandardScaler preprocessing step — this is needed mainly for SVR and the neural network (both are sensitive to feature scale), but using the same step for all four keeps the comparison fair.

**Linear Regression** — fits a straight-line relationship between the inputs and the target. The simplest approach, but it gives a useful baseline and is the easiest to interpret.

**Decision Tree Regression** — splits the data into rectangular regions using a series of if/else rules. Can capture things like "high CO2 and a busy Tuesday afternoon" that a linear model cannot, but tends to overfit if not constrained. Max depth was set to 8.

**Support Vector Regression (SVR)** — fits a smooth curve through the data while ignoring small errors. Uses an RBF kernel, with C=10 and epsilon=0.5. Works well on datasets of this size when the relationship is mildly nonlinear.

**Neural Network Regression** — a small two-layer network (32 and 16 units) with early stopping. The most flexible of the four, but also the most prone to overfitting and the hardest to interpret.

**On occupancy as a feature.** Occupancy was included in the training data to test whether it actually helps — not because it was assumed useful. Given that no occupancy zone could be statistically linked to this sensor's room (see Section 7), including it and measuring the effect was the right way to answer the question.

## 4. Results and Model Comparison

**How the models were tested.** The dataset (17,010 rows after dropping incomplete entries) was split in time order: the first 80% was used for training, and the last 20% for testing. The cutoff date was 11 March 2026. A random split was not used, because that would let the model see "future" readings during training, making the test results look better than they would be in real deployment.

**Results (13,608 training / 3,402 test rows):**

| Model | MAE (ppm) | RMSE (ppm) | R² |
|---|---|---|---|
| Linear Regression | 9.012 | 25.579 | 0.3581 |
| Decision Tree Regression | 9.353 | 28.708 | 0.1914 |
| **SVR** | **8.350** | 25.894 | 0.3422 |
| Neural Network Regression | 9.350 | 26.015 | 0.3360 |

SVR had the lowest average error (8.35 ppm) and is the best-performing model here. Linear Regression and the Neural Network were close behind. Decision Tree Regression was clearly the worst — it probably picked up patterns in the training period that do not generalise. An R² of around 0.34–0.36 means the models explain roughly a third of the variance in the next-15-minute reading. That is a real but limited result, and it is reported honestly.

**Does knowing the building occupancy actually improve predictions?**

| Model | R² with occupancy | R² without occupancy | Change |
|---|---|---|---|
| Linear Regression | 0.3581 | 0.3579 | +0.0002 (no difference) |
| Decision Tree Regression | 0.1914 | 0.3305 | −0.1391 (gets worse) |
| SVR | 0.3422 | 0.3438 | −0.0016 (no real difference) |
| Neural Network Regression | 0.3360 | 0.3352 | +0.0008 (no difference) |

Occupancy does not help for any model. For the Decision Tree it actively makes things worse, probably because the model is wasting splits on an irrelevant feature. This is consistent with the correlation analysis in Section 7 — there was never strong evidence this sensor is in a room that any of the occupancy zones covers.

These results are limited to this specific sensor, in this specific building, over this time period. Whether occupancy would help for a different sensor that is actually in the same room as an occupancy counter is a different question — this dataset cannot answer it.

## 5. Handling Missing Data

The CO2 readings are not evenly spaced. There are real outages in the data, the longest of which covers about 636 days. This creates two practical problems:

**Most candidate rows get dropped.** To make one training row, you need the current reading, the reading from 15 minutes ago, a one-hour rolling average, and a target 15 minutes ahead — all real, non-missing values. Because of the gaps, 84,576 candidate timestamps are in the raw data, but only 17,010 of them (about 20%) have all four values present. The rest are dropped. This is not ideal, but filling in made-up values across a 636-day gap would be far worse.

**The integration step initially bridged the outages.** When Section 6 was first coded up, the CO2 data was resampled onto a regular 15-minute grid and the missing bins were dropped before integrating. The problem is that this makes readings on either side of a gap look like they are just 15 minutes apart — even if they are actually 636 days apart. This was caught when the two integration methods gave wildly different answers and got fixed; see Section 6.

## 6. Numerical Integration — CO2 Exposure

A CO2 sensor gives a snapshot reading at each point in time, but what you often want to know is how much CO2 exposure accumulated over a period — for example, over a working day. That is an integration problem: total exposure in ppm·hours = area under the CO2-vs-time curve. This connects directly to Week 6 of MMA3001.

Two methods from Week 6 are implemented in `occupancy.co2_integration`: the trapezoidal rule and Simpson's 1/3 rule. Both were coded from the Week 6 formulas directly, not using a library function, so the steps can be checked against the taught method.

**The bug, and how it was caught.** The first version of this section compared trapezoidal and Simpson estimates and got wildly different answers: about 1.1 million ppm·h from Simpson's rule versus 9.5 million ppm·h from trapezoidal, on the same data. That 9× gap is the signal — when two methods that should broadly agree diverge by a factor of 9, something is wrong. The problem was the gap-bridging issue described in Section 5: Simpson's rule requires evenly spaced points, and the 636-day outage, which looked like a 15-minute step in the resampled grid, violates that assumption badly. Trapezoidal is more forgiving of uneven spacing, which is why it produced a larger (also wrong, but less catastrophically wrong) number.

**The fix:** A gap-aware version of the trapezoidal rule (`trapezoidal_integral_gap_aware`) skips any panel where the two endpoints are more than 2 hours apart. This means only genuinely continuous stretches of data are integrated. Over the sensor's full nominal span of about 20,720 hours, only roughly 5,228 hours (25%) pass that threshold — the 636-day gap accounts for most of the rest. The gap-aware estimate (2,305,836 ppm·h) is less than a quarter of the naive estimate (9,543,357 ppm·h), with no measurement error involved — just the outage-bridging assumption alone.

**Convergence check on a clean window.** To properly test whether the two methods agree when the data is actually evenly spaced, a 41-hour gap-free window (1–3 October 2025) was used. The same window was then resampled at 15, 30, and 60-minute resolution:

| Bin size | Points | Trapezoidal (ppm·h) | Simpson (ppm·h) | Agreement |
|---|---|---|---|---|
| 15 min | 163 | 17,817.3 | 17,829.9 | 0.071% |
| 30 min | 83 | 18,019.8 | 18,019.9 | 0.001% |
| 60 min | 41 | 17,583.8 | 17,571.8 | 0.068% |

Both methods now agree to within 0.1% at all three resolutions. The estimates shift slightly as the bin size changes (from 60→30 min: +2.4%, from 30→15 min: −1.1%), which reflects real variation in the sensor readings rather than numerical error. Real sensor data does not smooth out the way textbook examples do.

Note: 5-minute bins were checked and ruled out — the sensor's real reporting cadence is about 10 minutes, so the longest gap-free run at 5-minute resolution is only 2 data points, which is not enough to integrate. That is a sensor characteristic, not a limitation of the method.

## 7. Limitations and Lessons Learned

**The predictions are moderate, not great.** An R² of around 0.34 means the models capture about a third of what drives next-15-minute CO2 levels. The rest is things the model does not have access to: HVAC switching events, doors opening and closing, windows, equipment running in the room. That is honest. The result is useful as a trend signal, not as a precise forecast.

**Four of the five sensors have real problems.** Two of them have frozen clocks and cannot be used at all. A third only has four months of data. This was not documented anywhere in the dataset description — it was found by actually inspecting the timestamps. The one sensor used in this project was chosen partly because it could be validated against Bureau of Meteorology records, which is at least some confidence it is working.

**The sensor has no documented room location.** Sensor `6012002000326` does not appear in the Sensor ID and Locations spreadsheet. This means there is no way to know which room it is in, and no way to pick the right occupancy zone to pair it with. The correlation analysis (`scripts/sensor_matching_analysis.py`) tested all five occupancy zones against all five sensors and found no meaningful relationship for any pair — the highest CO2 correlation found was |r| ≤ 0.06. This is why occupancy does not improve the predictions, and it is reported as a finding, not hidden.

**The sensor was externally validated.** Even without a room label, the sensor's temperature and humidity readings correlate with real outdoor weather data (r = +0.84 for temperature, r = +0.65 for humidity) across 222 days. That gives reasonable confidence the sensor is physically measuring something real, even if its exact placement is unknown.

**The occupancy log filename was misleading.** The file named "MayToDec2024" actually contains data from November 2023 to April 2026. The date range was only found by checking the actual timestamps in the file.

## 8. AI-Use Reflection

I used Claude (Claude Code) throughout this project — for exploring the raw data, writing and running the Python code, doing the analysis, and drafting the report. I want to be straightforward about what that means.

The bulk of the coding work — setting up the package structure, writing the pipeline code, building the experiment scripts, running the four-model comparison, doing the integration analysis — was done with Claude's help. I directed what to build and what questions to answer. I chose to focus on CO2 prediction rather than occupancy classification because it matched the dataset better. I decided which sensor to use and why. I reviewed the results as they came in and questioned things that did not look right — for example, when the two integration methods gave completely different answers, I pushed to find out why rather than just accepting one of them.

The main bugs found during the project — the frozen-clock sensors, the gap-bridging error in the integration, the wrong window comparison in the convergence check — were caught by running the code against real data and noticing when results did not make sense. Code that looks correct can still produce wrong answers; the validation checks are what actually found the problems.

I reviewed the numbers in this report against the actual outputs in `reports/` before writing them in. The claims in Sections 4 and 6 come from the scripts that can be rerun, not from memory or from what I expected to see.

What I cannot fully account for is exactly which lines of reasoning came from me versus from Claude in the back-and-forth. The engineering framing, the scope decisions, and the interpretation of the results are mine. The code and the prose are collaborative.

## References

- MMA3001 Project Brief, Monash University, 2026.
- MMA3001 Project Datasets document, Monash University, 2026 (Dataset 2: Monash Smart Infrastructure Occupancy and Environmental Data).
- MMA3001 Notes, Week 5.1–5.6 (Regression methods and evaluation), Monash University, 2026.
- MMA3001 Week 6 Notes (Numerical Integration), Monash University, 2026.
- Bureau of Meteorology, Daily Weather Observations, Moorabbin Airport (station 086077) — http://www.bom.gov.au/climate/dwo/
- scikit-learn documentation: LinearRegression, DecisionTreeRegressor, SVR, MLPRegressor, StandardScaler.
