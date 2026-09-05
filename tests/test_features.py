"""Unit tests for occupancy.features.build_supervised_dataset."""

import pandas as pd
import pytest

from occupancy.features import build_supervised_dataset


def _resampled_room(occupied: list[float], room: str = "r1") -> pd.DataFrame:
    """Build a tiny resample_occupancy-shaped table for one room."""
    times = pd.date_range(
        "2024-06-03 09:00", periods=len(occupied), freq="15min", tz="Australia/Melbourne"
    )
    return pd.DataFrame(
        {"floorspaceid": room, "collecteddate": times, "occupied": occupied}
    )


def test_lag_rolling_and_target_are_computed_correctly():
    """Hand-computed expected values for a 6-bin sequence.

    occupied = [1, 1, 0, 0, 1, 1]; with lag1 needing 1 prior bin and a
    4-bin rolling mean needing 4 prior bins, and the last bin having no
    "next" to form a target, only bins index 3 and 4 have every required
    value.
    """
    resampled = _resampled_room([1, 1, 0, 0, 1, 1])
    table, info = build_supervised_dataset(resampled, rolling_window=4)

    assert info == {
        "rows_before": 6,
        "rows_after": 2,
        "rows_dropped": 4,
        "pct_dropped": pytest.approx(4 / 6 * 100),
    }
    assert len(table) == 2

    row3, row4 = table.iloc[0], table.iloc[1]
    assert row3["occupied_now"] == 0.0
    assert row3["occupied_lag1"] == 0.0
    assert row3["rolling_mean_1h"] == pytest.approx(0.5)
    assert row3["target"] == 1.0

    assert row4["occupied_now"] == 1.0
    assert row4["occupied_lag1"] == 0.0
    assert row4["rolling_mean_1h"] == pytest.approx(0.5)
    assert row4["target"] == 1.0


def test_calendar_features_are_bounded():
    resampled = _resampled_room([1, 0, 1, 0, 1, 0, 1])
    table, _ = build_supervised_dataset(resampled, rolling_window=2)
    for col in ("hour_sin", "hour_cos", "dow_sin", "dow_cos"):
        assert (table[col] >= -1.0).all() and (table[col] <= 1.0).all()
    assert table["is_weekend"].isin([0.0, 1.0]).all()


def test_rooms_do_not_leak_into_each_others_lag_or_rolling_features():
    a = _resampled_room([1, 1, 1, 1, 1, 1], room="r1")
    b = _resampled_room([0, 0, 0, 0, 0, 0], room="r2")
    resampled = pd.concat([a, b], ignore_index=True)
    table, _ = build_supervised_dataset(resampled, rolling_window=4)

    assert set(table["floorspaceid"]) == {"r1", "r2"}
    assert (table[table["floorspaceid"] == "r1"]["rolling_mean_1h"] == 1.0).all()
    assert (table[table["floorspaceid"] == "r2"]["rolling_mean_1h"] == 0.0).all()


def test_raises_on_empty_input():
    empty = pd.DataFrame(columns=["floorspaceid", "collecteddate", "occupied"])
    with pytest.raises(ValueError):
        build_supervised_dataset(empty)
