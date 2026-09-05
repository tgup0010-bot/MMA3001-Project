"""Unit tests for occupancy.evaluation."""

import math

import pandas as pd
import pytest

from occupancy.evaluation import chronological_split, evaluate_predictions


def test_chronological_split_respects_train_frac_and_time_order():
    times = pd.date_range("2024-01-01", periods=10, freq="D")
    table = pd.DataFrame({"collecteddate": times, "value": range(10)})

    train, test, cutoff = chronological_split(table, time_col="collecteddate", train_frac=0.7)

    assert len(train) == 7
    assert len(test) == 3
    assert train["collecteddate"].max() < test["collecteddate"].min() or train["collecteddate"].max() == cutoff
    # No overlap, and every training timestamp precedes every test timestamp.
    assert train["collecteddate"].max() <= test["collecteddate"].min()


@pytest.mark.parametrize("bad_frac", [0.0, 1.0, -0.1, 1.5])
def test_chronological_split_rejects_invalid_train_frac(bad_frac):
    table = pd.DataFrame({"collecteddate": pd.date_range("2024-01-01", periods=5, freq="D")})
    with pytest.raises(ValueError):
        chronological_split(table, time_col="collecteddate", train_frac=bad_frac)


def test_evaluate_predictions_perfect_predictions():
    y_true = [1, 0, 1, 0, 1]
    y_prob = [1.0, 0.0, 1.0, 0.0, 1.0]
    metrics = evaluate_predictions(y_true, y_prob)
    assert metrics["accuracy"] == 1.0
    assert metrics["f1"] == 1.0
    assert metrics["brier_score"] == 0.0
    assert metrics["n"] == 5
    assert metrics["positive_rate"] == pytest.approx(0.6)


def test_evaluate_predictions_single_class_gives_nan_roc_auc():
    y_true = [1, 1, 1]
    y_prob = [0.9, 0.8, 0.95]
    metrics = evaluate_predictions(y_true, y_prob)
    assert math.isnan(metrics["roc_auc"])
