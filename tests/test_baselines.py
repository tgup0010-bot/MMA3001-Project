"""Unit tests for occupancy.baselines."""

import numpy as np
import pandas as pd
import pytest

from occupancy.baselines import MarkovBaseline, persistence_predict_proba


def test_persistence_equals_occupied_now():
    table = pd.DataFrame({"occupied_now": [1.0, 0.0, 1.0, 0.0]})
    proba = persistence_predict_proba(table)
    np.testing.assert_array_equal(proba, [1.0, 0.0, 1.0, 0.0])


def _training_table() -> pd.DataFrame:
    # Room "r1", hour 9: occupied_now=1 -> target [1, 1, 0] (mean 2/3)
    #                    occupied_now=0 -> target [0, 1]    (mean 1/2)
    # Overall mean across all 5 rows: 3/5 = 0.6
    ts = pd.Timestamp("2024-06-03 09:00", tz="Australia/Melbourne")
    return pd.DataFrame(
        {
            "floorspaceid": ["r1"] * 5,
            "collecteddate": [ts] * 5,
            "occupied_now": [1, 1, 1, 0, 0],
            "target": [1, 1, 0, 0, 1],
        }
    )


def test_markov_baseline_learns_conditional_rates():
    model = MarkovBaseline().fit(_training_table())

    query = pd.DataFrame(
        {
            "floorspaceid": ["r1", "r1"],
            "collecteddate": [pd.Timestamp("2024-06-03 09:00", tz="Australia/Melbourne")] * 2,
            "occupied_now": [1, 0],
        }
    )
    proba = model.predict_proba(query)
    assert proba[0] == pytest.approx(2 / 3)
    assert proba[1] == pytest.approx(1 / 2)


def test_markov_baseline_falls_back_to_global_rate_for_unseen_combo():
    model = MarkovBaseline().fit(_training_table())

    # Hour 14 was never seen during fit for room "r1".
    query = pd.DataFrame(
        {
            "floorspaceid": ["r1"],
            "collecteddate": [pd.Timestamp("2024-06-03 14:00", tz="Australia/Melbourne")],
            "occupied_now": [1],
        }
    )
    proba = model.predict_proba(query)
    assert proba[0] == pytest.approx(0.6)


def test_markov_baseline_predict_before_fit_raises():
    model = MarkovBaseline()
    with pytest.raises(RuntimeError):
        model.predict_proba(_training_table())
