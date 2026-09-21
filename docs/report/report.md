# Predicting Indoor CO₂ Concentration from Sensor History and Building Occupancy

**MMA3001 — Numerical Methods and Machine Learning, Project Report**
**Dataset:** MMA3001 Dataset 2 — Monash Smart Infrastructure Occupancy and Environmental Data
**Repository:** https://github.com/tgup0010-bot/MMA3001-Project

---

## 1. Engineering Problem

Indoor CO₂ concentration in office buildings is a direct indicator of air quality. When CO₂ exceeds approximately 800–1000 ppm, occupants report reduced alertness and discomfort. Current building management systems respond reactively — ventilation increases only after CO₂ has already risen. The consequence is that occupants are exposed to poor air quality until the system catches up.

The problem addressed here is **short-term predictive CO₂ forecasting**: given the recent history of a room's CO₂ readings and current building state, predict what the CO₂ concentration will be 15 minutes from now. An accurate forecast gives the building management system advance notice to ramp up ventilation before the threshold is crossed, improving both occupant comfort and energy efficiency.

This is a **regression task** — predicting a continuous value (ppm) rather than a class. The four regression methods from MMA3001 Week 5 are compared: Linear Regression, Decision Tree Regression, Support Vector Regression (SVR), and Neural Network Regression. The goal is to determine which method is most suitable for this real sensor dataset, and whether building occupancy data improves predictions.

**Intended use:** A trained model would run as a lightweight inference step within a building management system, issuing a CO₂ forecast every 15 minutes and triggering pre-emptive ventilation adjustments when the forecast exceeds a threshold.

**Limitation noted upfront:** Only one of five environmental sensors in the dataset had sufficient usable, contiguous data to model. The working sensor also has no documented room location, which limits how well occupancy can be linked to CO₂. These constraints are reported fully in Section 6.

---

## 2. Inputs and Outputs

### 2.1 Input Features

The following features are computed from the raw sensor log and used as inputs to all four models:

| Feature | Units / Type | Description |
|---|---|---|
| `co2_now` | ppm (float) | CO₂ reading at the current 15-minute bin |
| `co2_lag1` | ppm (float) | CO₂ reading from the previous 15-minute bin |
| `co2_rolling_1h` | ppm (float) | Mean CO₂ over the preceding four bins (1 hour) |
| `building_occupancy` | integer 0–5 | Number of the 5 monitored zones currently occupied |
| `hour_sin`, `hour_cos` | float (−1 to 1) | Cyclic encoding of hour-of-day |
| `dow_sin`, `dow_cos` | float (−1 to 1) | Cyclic encoding of day-of-week |
| `is_weekend` | binary 0/1 | 1 if Saturday or Sunday, 0 otherwise |

**Why cyclic encoding for time?** Hour 23 and hour 0 are 1 hour apart, not 23 hours apart. A standard integer hour feature would represent them as far apart; sin/cos encoding preserves the circular relationship correctly.

**Handling missing or invalid inputs:** Any row where `co2_now`, `co2_lag1`, `co2_rolling_1h`, or the target value is missing is dropped entirely rather than imputed. Imputing across the 636-day data gap (see Section 5) would introduce fabricated values over a very long outage and distort the model. Dropped rows account for 79.9% of candidate timestamps; see Section 5 for detail.

### 2.2 Output

| Output | Units | Meaning |
|---|---|---|
| Predicted CO₂ | ppm (float) | Forecast CO₂ concentration 15 minutes ahead |

The model outputs a single continuous value. A value above approximately 800–1000 ppm would flag a ventilation action in a deployed system. The model does not output a binary alert — that threshold decision is left to the building management system consuming the forecast.

---

## 3. Computational Solution

### 3.1 How Each Method Works

All four models are implemented as scikit-learn `Pipeline` objects, each with a `StandardScaler` preprocessing step followed by the regression estimator. Using the same preprocessing for all four ensures the comparison is fair.

**Linear Regression** fits a linear combination of the input features to minimise mean squared error. It is the simplest method and provides the baseline. It assumes the relationship between inputs and CO₂ is approximately linear, which is a reasonable first assumption given that CO₂ trends tend to be smooth over short windows.

**Decision Tree Regression** partitions the feature space into rectangular regions using a binary splitting rule, predicting the mean target value in each leaf. It can capture non-linear patterns such as "high CO₂ combined with a midweek afternoon" without any explicit feature engineering. Maximum depth was set to 8 to limit overfitting.

**Support Vector Regression (SVR)** fits a smooth function by finding a hyperplane that keeps most residuals within an ε-tube, using an RBF kernel to handle mild non-linearity. Parameters: C = 10 (regularisation), ε = 0.5 (tube width). SVR works well on medium-sized datasets where the relationship is mildly non-linear and the feature scale is controlled by preprocessing.

**Neural Network Regression** uses a two-layer multilayer perceptron (32 and 16 units, ReLU activations) trained with early stopping (patience = 10) to prevent overfitting. It is the most flexible model but also the least interpretable and most data-hungry.

### 3.2 Why These Methods

All four are regression methods taught in MMA3001 Week 5. The purpose of comparing them is to determine which generalises best to this specific dataset, rather than selecting one by assumption. SVR and neural networks are more capable of representing non-linear relationships than linear regression, while the decision tree provides a non-parametric baseline for non-linearity.

### 3.3 Assumptions and Parameters

| Parameter | Value | Rationale |
|---|---|---|
| Bin size | 15 minutes | Matches sensor reporting cadence |
| Prediction horizon | 15 minutes | One bin ahead — the minimum useful lead time |
| Train/test split | 80% / 20%, time-ordered | Prevents future data leaking into training |
| Training cutoff | 11 March 2026 | 80th percentile of timeline |
| Decision Tree max depth | 8 | Balances complexity and generalisation |
| SVR kernel | RBF | Handles mild non-linearity without overfitting |
| SVR C | 10 | Moderate regularisation |
| SVR ε | 0.5 | Tube width matched to sensor noise level |
| NN hidden layers | 32, 16 units | Small enough to avoid overfitting on 13,608 rows |
| NN early stopping patience | 10 epochs | Stops before overfitting begins |

**Why a time-ordered split?** A random split would allow the model to train on readings from March 2026 and test on January 2025. In deployment the model only ever sees past data. The time-ordered split correctly simulates this: training on all data before the cutoff, testing on all data after.

---

## 4. Alternative Solutions

The table below compares the four regression methods across the criteria from the project brief. All metrics are measured on the held-out test set (3,402 rows).

| Criterion | Linear Regression | Decision Tree | **SVR** | Neural Network |
|---|---|---|---|---|
| MAE (ppm) | 9.012 | 9.353 | **8.350** | 9.350 |
| RMSE (ppm) | 25.579 | 28.708 | 25.894 | 26.015 |
| R² | 0.358 | 0.191 | 0.342 | 0.336 |
| Fit time (s) | 0.005 | 0.023 | 4.26 | 2.36 |
| Predict time (ms) | 1.7 | 2.5 | 2854 | 2.3 |
| Interpretability | High | Medium | Low | Very low |
| Overfitting risk | Low | High | Low | Medium |
| Memory | Very low | Low | Medium | Low |

**SVR** achieved the lowest MAE (8.35 ppm) and competitive RMSE, making it the best-performing model. Its prediction time (2.85 seconds per batch) is substantially longer than the other methods, but for a 15-minute scheduling task this latency is not a practical constraint.

**Decision Tree** was the worst performer (R² = 0.191), likely because it overfit patterns in the training period that do not generalise. Including occupancy as a feature made it worse (R² dropped from 0.331 to 0.191), suggesting it wasted splits on an irrelevant feature.

**Linear Regression** was only marginally below SVR in R² (0.358 vs 0.342), making it the best alternative when interpretability or fast prediction time matters. For an embedded system or a model that must be explainable, linear regression is the practical choice.

**Does adding building occupancy help?**

| Model | R² with occupancy | R² without occupancy |
|---|---|---|
| Linear Regression | 0.358 | 0.358 |
| Decision Tree | 0.191 | 0.331 |
| SVR | 0.342 | 0.344 |
| Neural Network | 0.336 | 0.335 |

Occupancy provides no improvement for any model. The reason is investigated in Section 6: the sensor is not located in any of the five monitored zones, so building occupancy is simply not correlated with this sensor's readings.

---

## 5. Validation

### 5.1 Method

The dataset was split in chronological order: the first 80% (rows 1–13,608) for training, the last 20% (rows 13,609–17,010) for testing. The training period ends on 11 March 2026.

A **random split was deliberately not used** because it would allow models to train on future readings and test on past readings, artificially inflating R². In deployment, the model only ever predicts forward from what it has already seen.

Three metrics were used:

- **MAE (Mean Absolute Error):** Average prediction error in ppm. Directly interpretable — an MAE of 8.35 means the model is on average 8.35 ppm off. Less sensitive to large outliers than RMSE.
- **RMSE (Root Mean Square Error):** Penalises large errors more than MAE. Useful for detecting whether a model occasionally makes very bad predictions.
- **R² (coefficient of determination):** Fraction of variance in the target explained by the model. R² = 1 is perfect; R² = 0 means the model does no better than always predicting the mean.

### 5.2 Results

The best model (SVR) achieved MAE = 8.35 ppm on the test set. Given that CO₂ readings in this dataset range from roughly 400 to 1500 ppm, and the action threshold is around 800–1000 ppm, an average error of 8.35 ppm is small enough to be practically useful for trend-based ventilation control.

An R² of 0.34–0.36 means models explain approximately a third of the variance in next-15-minute CO₂ levels. The remaining variance is attributable to factors not present in the dataset: HVAC switching events, doors and windows, equipment, and the actual occupants in this room (since the occupancy counter does not cover it).

### 5.3 Sensor Validation

The working sensor (`6012002000326`) was externally validated against Bureau of Meteorology weather records from Moorabbin Airport (station 086077), the closest station to the Monash campus, across 222 overlapping days:

| Variable | Pearson r (sensor vs BOM) |
|---|---|
| Temperature | +0.84 |
| Humidity | +0.65 |

These correlations are strong evidence the sensor is physically functioning correctly. A broken or randomly drifting sensor would show r near zero.

### 5.4 Numerical Integration Validation

Two numerical integration methods (trapezoidal rule and Simpson's 1/3 rule, both implemented directly from Week 6 formulas) were tested on a 41-hour gap-free window (1–3 October 2025) to verify convergence. Results at three bin resolutions:

| Bin size | Trapezoidal (ppm·h) | Simpson (ppm·h) | Agreement |
|---|---|---|---|
| 15 min | 17,817.3 | 17,829.9 | 0.071% |
| 30 min | 18,019.8 | 18,019.9 | 0.001% |
| 60 min | 17,583.8 | 17,571.8 | 0.068% |

Both methods agree to within 0.1% at all three resolutions, confirming the implementation is correct. The shift between bin sizes reflects real variation in sensor readings, not numerical error.

---

## 6. Limitations and Known Failure Modes

**Sensor location is unknown.** Sensor `6012002000326` does not appear in the Sensor ID and Locations spreadsheet. This is why occupancy cannot be meaningfully linked to CO₂ for this sensor. A correlation analysis across all five sensors and all five occupancy zones found no meaningful relationship for any pair (maximum |r| ≤ 0.06 for CO₂-vs-occupancy; all Pearson r values were negative, consistent with a HVAC dilution effect rather than a genuine occupancy signal). This is a dataset limitation, not a modelling limitation.

**Four of five sensors are unusable.** Two sensors (`6012002000227`, `6012002000777`) have frozen internal clocks — every reading carries the same timestamp. A third (`6012002000869`) has only four months of data, too short for reliable train/test evaluation. The fourth remaining sensor was not chosen because its temperature/humidity readings cannot be cross-validated against weather records.

**80% of candidate rows are dropped.** The 636-day sensor outage means that 84,576 candidate 15-minute timestamps are in the raw log but only 17,010 (20.1%) have complete lag features and a valid target. Imputing across this gap would be methodologically wrong, so rows are dropped.

**R² is moderate.** The models explain about a third of next-15-minute CO₂ variance. The remaining variance is driven by factors not measured in this dataset. The result is honest: the model provides a useful trend signal, not a precise forecast. For a deployed system, a conservative threshold (e.g., forecast > 750 ppm triggers early ventilation) would account for prediction uncertainty.

**The gap-aware integration is critical.** A naive trapezoidal integration over the full sensor record produces 9.54 million ppm·h. After skipping panels where the two endpoints are more than 2 hours apart (i.e., bridging the 636-day outage), the estimate drops to 2.31 million ppm·h. The factor-of-four difference comes entirely from the integration wrongly bridging the outage, not from measurement error. The gap-aware implementation is the correct result.

---

## 7. AI Use and Critical Reflection

Claude (Claude Code) was used throughout this project for data exploration, Python implementation, analysis, and drafting this report. The following is an honest account of how it was used and what I contributed.

**What Claude did:** Set up the package structure, wrote the pipeline code, ran the four-model comparison, performed the integration analysis, and produced drafts of this report and the README. Claude also suggested the zone-matching validation method (filtering weekends, isolating single-zone bins, checking CO₂ delta) as a way to test whether the sensor is co-located with any monitored zone.

**What I directed and decided:** I chose CO₂ prediction as the project direction rather than occupancy classification, because the regression framing was a better fit for the dataset. I chose which sensor to use and required external validation before accepting it as reliable. I decided to include occupancy as a feature and test it empirically rather than assume it helps. When the two integration methods gave results that differed by a factor of 9, I pushed to understand why rather than accepting either answer — that investigation identified the gap-bridging bug.

**What I verified:** The numbers in this report were checked against the actual outputs in `reports/co2_prediction_results.json` before being written here. The claims about integration estimates come from scripts that can be rerun.

**Honest limitation:** In extended AI-assisted work, it is not always possible to draw a precise boundary between ideas that originated with me versus with Claude. The engineering framing, scope decisions, and interpretation of results are mine. The implementation and prose are collaborative.

---

## References

- MMA3001 Project Brief, Monash University, 2026.
- MMA3001 Project Datasets document, Monash University, 2026 (Dataset 2: Monash Smart Infrastructure Occupancy and Environmental Data).
- MMA3001 Notes, Week 5.1–5.6 (Regression methods and evaluation), Monash University, 2026.
- MMA3001 Notes, Week 6 (Numerical Integration), Monash University, 2026.
- Bureau of Meteorology, Daily Weather Observations, Moorabbin Airport (station 086077). http://www.bom.gov.au/climate/dwo/
- scikit-learn documentation: LinearRegression, DecisionTreeRegressor, SVR, MLPRegressor, StandardScaler, Pipeline.
