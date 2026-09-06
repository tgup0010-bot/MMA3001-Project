"""Train the final model once and save it for instant reuse in the demo.

Running the full pipeline from scratch (scripts/run_experiment.py) takes a
couple of minutes, almost entirely spent reading and parsing the 1.5GB raw
CSV. That is fine for a one-off experiment, but no good for a live
presentation. This script does the slow part ONCE, then saves:

- the fitted logistic regression pipeline (and Markov baseline, for
  comparison) to `models/occupancy_model.joblib`;
- a small, deterministically-chosen sample of real test-set rows (with
  known ground truth) to `models/demo_examples.csv`.

`scripts/demo.py` loads both of these in well under a second, so it can be
run live without waiting on the raw file again.

Usage:
    python scripts/train_and_save_model.py
"""

from __future__ import annotations

from pathlib import Path

import joblib

from occupancy.baselines import MarkovBaseline
from occupancy.data_loading import load_occupancy_log
from occupancy.evaluation import chronological_split
from occupancy.features import CATEGORICAL_FEATURES, NUMERIC_FEATURES, build_supervised_dataset
from occupancy.model import build_logistic_pipeline
from occupancy.preprocessing import resample_occupancy

RAW_PATH = Path("data/raw/5occupancySensor_MayToDec2024_9MRows.csv")
MODEL_PATH = Path("models/occupancy_model.joblib")
EXAMPLES_PATH = Path("models/demo_examples.csv")

X_COLS = list(NUMERIC_FEATURES) + list(CATEGORICAL_FEATURES)

#: How many worked examples to save for the demo's "replay real history"
#: part. Sampled evenly across the (chronologically ordered) test set --
#: not selected for how good the prediction looks, so the demo can't
#: accidentally cherry-pick favourable cases.
N_DEMO_EXAMPLES = 15


def main() -> None:
    print("Loading full occupancy log (this is the slow, one-off step)...")
    df = load_occupancy_log(RAW_PATH)
    print(f"  rows={len(df):,}")

    resampled = resample_occupancy(df, bin_size="15min", max_gap="2h")
    table, info = build_supervised_dataset(resampled, rolling_window=4)
    train, test, cutoff = chronological_split(table, train_frac=0.8)
    print(f"  train={len(train):,}  test={len(test):,}  cutoff={cutoff}")

    print("Fitting final logistic regression model...")
    pipeline = build_logistic_pipeline()
    pipeline.fit(train[X_COLS], train["target"])

    print("Fitting Markov baseline (kept for demo comparison)...")
    markov = MarkovBaseline().fit(train)

    MODEL_PATH.parent.mkdir(exist_ok=True)
    joblib.dump(
        {"logreg": pipeline, "markov": markov, "feature_columns": X_COLS},
        MODEL_PATH,
    )
    print(f"Saved model -> {MODEL_PATH}")

    proba = pipeline.predict_proba(test[X_COLS])[:, 1]
    test = test.copy()
    test["predicted_proba"] = proba

    step = max(1, len(test) // N_DEMO_EXAMPLES)
    sample = test.iloc[::step].head(N_DEMO_EXAMPLES).reset_index(drop=True)
    sample.to_csv(EXAMPLES_PATH, index=False)
    print(f"Saved {len(sample)} demo examples -> {EXAMPLES_PATH}")


if __name__ == "__main__":
    main()
