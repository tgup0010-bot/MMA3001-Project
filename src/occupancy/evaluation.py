"""Chronological train/test splitting and prediction-quality metrics.

Splitting must be done by time, never at random: a randomly shuffled split
would let the model (or even the Markov baseline) see information from
"after" the point it is meant to predict, silently inflating every metric.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, brier_score_loss, f1_score, roc_auc_score


def chronological_split(
    table: pd.DataFrame,
    *,
    time_col: str = "collecteddate",
    train_frac: float = 0.8,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Timestamp]:
    """Split a table into train/test sets by time, not at random.

    All rows up to a single global cutoff timestamp (computed from
    ``train_frac`` of the pooled, sorted timestamps across every room) go
    to training; everything after goes to test. Using one global cutoff
    (rather than splitting each room independently) keeps the train/test
    boundary at the same wall-clock moment for every room, which is what
    "can this model generalise to the future" should mean here.

    Args:
        table: Supervised table (e.g. from
            :func:`occupancy.features.build_supervised_dataset`).
        time_col: Column holding each row's timestamp.
        train_frac: Fraction of rows (by time, not room-balanced count)
            assigned to training.

    Returns:
        ``(train, test, cutoff_time)``.

    Raises:
        ValueError: If ``train_frac`` is not strictly between 0 and 1, or
            the split would leave either set empty.
    """
    if not 0.0 < train_frac < 1.0:
        raise ValueError(f"train_frac must be in (0, 1), got {train_frac}")

    sorted_table = table.sort_values(time_col).reset_index(drop=True)
    cutoff_idx = int(len(sorted_table) * train_frac)
    if cutoff_idx == 0 or cutoff_idx >= len(sorted_table):
        raise ValueError(
            f"train_frac={train_frac} produced an empty train or test set "
            f"for {len(sorted_table)} rows."
        )

    cutoff_time = sorted_table.loc[cutoff_idx, time_col]
    train = sorted_table.iloc[:cutoff_idx].reset_index(drop=True)
    test = sorted_table.iloc[cutoff_idx:].reset_index(drop=True)
    return train, test, cutoff_time


def evaluate_predictions(y_true, y_prob, *, threshold: float = 0.5) -> dict:
    """Compute a standard set of classification metrics.

    Several metrics are reported together (rather than just accuracy)
    because occupancy is imbalanced (the "occupied" class dominates) --
    accuracy alone can look deceptively good for a model that just always
    predicts the majority class.

    Args:
        y_true: Ground-truth binary labels (0/1).
        y_prob: Predicted probability of the positive (occupied) class.
        threshold: Probability threshold used to derive hard predictions
            for accuracy/F1.

    Returns:
        A dict with ``accuracy``, ``f1``, ``roc_auc`` (``NaN`` if
        ``y_true`` has only one class present), ``brier_score``
        (calibration -- lower is better), ``n`` (rows evaluated) and
        ``positive_rate`` (base rate of the occupied class, for context).
    """
    y_true = np.asarray(y_true, dtype=float)
    y_prob = np.asarray(y_prob, dtype=float)
    y_pred = (y_prob >= threshold).astype(int)

    roc_auc = roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else float("nan")

    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc,
        "brier_score": brier_score_loss(y_true, y_prob),
        "n": int(len(y_true)),
        "positive_rate": float(np.mean(y_true)),
    }
