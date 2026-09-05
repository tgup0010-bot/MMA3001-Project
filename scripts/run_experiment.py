"""End-to-end experiment: load -> resample -> features -> train -> evaluate.

Runs the full pipeline against the real raw occupancy log (not the small
dev sample) and prints a comparison table of the two baselines against the
logistic regression model, plus basic data-completeness diagnostics.

Usage:
    python scripts/run_experiment.py
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import pandas as pd

from occupancy.baselines import MarkovBaseline, persistence_predict_proba
from occupancy.data_loading import load_occupancy_log
from occupancy.evaluation import chronological_split, evaluate_predictions
from occupancy.features import CATEGORICAL_FEATURES, NUMERIC_FEATURES, build_supervised_dataset
from occupancy.model import build_logistic_pipeline
from occupancy.preprocessing import coverage_summary, resample_occupancy

RAW_PATH = Path("data/raw/5occupancySensor_MayToDec2024_9MRows.csv")
RESULTS_PATH = Path("reports/experiment_results.json")


def timed(label: str, fn, *args, **kwargs):
    start = time.perf_counter()
    result = fn(*args, **kwargs)
    elapsed = time.perf_counter() - start
    print(f"[{elapsed:6.1f}s] {label}")
    return result, elapsed


def main() -> None:
    timings: dict[str, float] = {}

    df, t = timed("load_occupancy_log (full file)", load_occupancy_log, RAW_PATH)
    timings["load"] = t
    print(f"  rows={len(df):,}  rooms={df['floorspaceid'].nunique()}")

    resampled, t = timed(
        "resample_occupancy (15min bins, 2h max_gap)",
        resample_occupancy,
        df,
        bin_size="15min",
        max_gap="2h",
    )
    timings["resample"] = t
    print(f"  grid rows={len(resampled):,}")
    print(coverage_summary(resampled))

    table, info = build_supervised_dataset(resampled, rolling_window=4)
    print(f"\nsupervised table: {info}")

    train, test, cutoff = chronological_split(table, train_frac=0.8)
    print(f"\ntrain rows={len(train):,}  test rows={len(test):,}  cutoff={cutoff}")

    results = {}

    # --- Baseline 1: persistence ---
    proba = persistence_predict_proba(test)
    results["persistence_baseline"] = evaluate_predictions(test["target"], proba)

    # --- Baseline 2: time-of-day Markov chain ---
    markov, t = timed("fit MarkovBaseline", MarkovBaseline().fit, train)
    timings["fit_markov"] = t
    proba = markov.predict_proba(test)
    results["markov_baseline"] = evaluate_predictions(test["target"], proba)

    # --- ML model: logistic regression ---
    X_cols = list(NUMERIC_FEATURES) + list(CATEGORICAL_FEATURES)
    pipeline = build_logistic_pipeline()
    _, t = timed(
        "fit logistic regression",
        pipeline.fit,
        train[X_cols],
        train["target"],
    )
    timings["fit_logreg"] = t
    proba = pipeline.predict_proba(test[X_cols])[:, 1]
    results["logistic_regression"] = evaluate_predictions(test["target"], proba)

    print("\n=== Results (test set, chronological holdout) ===")
    print(pd.DataFrame(results).T.to_string(float_format=lambda x: f"{x:.4f}"))

    RESULTS_PATH.parent.mkdir(exist_ok=True)
    RESULTS_PATH.write_text(
        json.dumps(
            {
                "data_info": info,
                "cutoff": str(cutoff),
                "n_train": len(train),
                "n_test": len(test),
                "timings_seconds": timings,
                "results": results,
            },
            indent=2,
            default=str,
        )
    )
    print(f"\nSaved -> {RESULTS_PATH}")


if __name__ == "__main__":
    main()
