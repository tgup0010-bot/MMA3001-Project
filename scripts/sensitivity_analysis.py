"""Sensitivity analysis and performance profiling for the occupancy pipeline.

Answers three "optimisation and performance" questions from the project
brief, using the real dataset (never the small dev sample, since these
numbers are meant to justify real design choices):

1. How does the choice of time-bin resolution trade off data volume/memory
   against predictive accuracy? (Sweep A)
2. How does the length of history used as a feature (the rolling-mean
   lookback window) affect accuracy? (Sweep B)
3. How do the two ML models compare in fit/predict runtime and model
   complexity (a proxy for memory footprint and per-prediction cost)?

Usage:
    python scripts/sensitivity_analysis.py
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
from occupancy.model import build_logistic_pipeline, build_random_forest_pipeline
from occupancy.preprocessing import resample_occupancy

RAW_PATH = Path("data/raw/5occupancySensor_MayToDec2024_9MRows.csv")
RESULTS_PATH = Path("reports/sensitivity_results.json")

X_COLS = list(NUMERIC_FEATURES) + list(CATEGORICAL_FEATURES)


def run_one_config(df: pd.DataFrame, bin_size: str, rolling_window: int) -> dict:
    """Run the full pipeline for one (bin_size, rolling_window) setting."""
    t0 = time.perf_counter()
    resampled = resample_occupancy(df, bin_size=bin_size, max_gap="2h")
    resample_time = time.perf_counter() - t0
    resampled_memory_mb = resampled.memory_usage(deep=True).sum() / 1e6

    table, info = build_supervised_dataset(resampled, rolling_window=rolling_window)
    train, test, cutoff = chronological_split(table, train_frac=0.8)

    row: dict = {
        "bin_size": bin_size,
        "rolling_window_bins": rolling_window,
        "grid_rows": len(resampled),
        "resampled_memory_mb": round(resampled_memory_mb, 2),
        "resample_time_s": round(resample_time, 3),
        "supervised_rows": info["rows_after"],
        "pct_dropped": round(info["pct_dropped"], 2),
        "n_train": len(train),
        "n_test": len(test),
    }

    # Baselines (cheap; included for a like-for-like reference on every config).
    row["persistence_accuracy"] = evaluate_predictions(
        test["target"], persistence_predict_proba(test)
    )["accuracy"]
    markov = MarkovBaseline().fit(train)
    row["markov_accuracy"] = evaluate_predictions(
        test["target"], markov.predict_proba(test)
    )["accuracy"]

    for name, builder in (
        ("logreg", build_logistic_pipeline),
        ("rf", build_random_forest_pipeline),
    ):
        pipeline = builder()
        t0 = time.perf_counter()
        pipeline.fit(train[X_COLS], train["target"])
        fit_time = time.perf_counter() - t0

        t0 = time.perf_counter()
        proba = pipeline.predict_proba(test[X_COLS])[:, 1]
        predict_time = time.perf_counter() - t0

        metrics = evaluate_predictions(test["target"], proba)
        row[f"{name}_accuracy"] = metrics["accuracy"]
        row[f"{name}_roc_auc"] = metrics["roc_auc"]
        row[f"{name}_fit_time_s"] = round(fit_time, 3)
        row[f"{name}_predict_us_per_row"] = round(1e6 * predict_time / len(test), 2)

        if name == "logreg":
            clf = pipeline.named_steps["classify"]
            row["logreg_n_parameters"] = int(clf.coef_.size + clf.intercept_.size)
        else:
            clf = pipeline.named_steps["classify"]
            total_nodes = sum(est.tree_.node_count for est in clf.estimators_)
            avg_depth = sum(est.get_depth() for est in clf.estimators_) / len(clf.estimators_)
            row["rf_total_tree_nodes"] = int(total_nodes)
            row["rf_avg_tree_depth"] = round(avg_depth, 1)

    return row


def main() -> None:
    print("Loading full occupancy log (this is the slow, one-off step)...")
    df = load_occupancy_log(RAW_PATH)
    print(f"  rows={len(df):,}")

    results = {"bin_size_sweep": [], "lookback_sweep": []}

    print("\n=== Sweep A: bin size (rolling_window chosen to hold ~1h lookback) ===")
    # bin_minutes -> rolling_window_bins, clamped to >= 2 so "rolling" is
    # still meaningful even at coarse resolutions.
    bin_configs = [("5min", 12), ("15min", 4), ("30min", 2), ("60min", 2)]
    for bin_size, rolling_window in bin_configs:
        print(f"  running bin_size={bin_size}, rolling_window={rolling_window} bins...")
        row = run_one_config(df, bin_size, rolling_window)
        results["bin_size_sweep"].append(row)
        print(f"    -> {row}")

    print("\n=== Sweep B: lookback window (bin_size fixed at 15min) ===")
    for rolling_window in (2, 4, 8, 16):  # 30min, 1h, 2h, 4h of lookback
        print(f"  running rolling_window={rolling_window} bins (15min bin_size)...")
        row = run_one_config(df, "15min", rolling_window)
        results["lookback_sweep"].append(row)
        print(f"    -> {row}")

    RESULTS_PATH.parent.mkdir(exist_ok=True)
    RESULTS_PATH.write_text(json.dumps(results, indent=2, default=str))
    print(f"\nSaved -> {RESULTS_PATH}")


if __name__ == "__main__":
    main()
