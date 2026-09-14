"""Unit tests for occupancy.env_sensors.

Uses small, hand-built JSON payloads (not the full raw CSV) so the suite
runs in well under a second and each case tests one specific behaviour.
"""

import json

import pandas as pd
import pytest

from occupancy.env_sensors import _flatten, load_env_sensor_log, pivot_variable, sensor_ids


def _row(sensorid: str, time: int, variable: str, unit: str, value: float) -> dict:
    return {
        "sensorid": sensorid,
        "jsondata": json.dumps(
            [
                {
                    "values": [{"time": time, "value": value}],
                    "variable": {"name": variable, "unit": unit},
                }
            ]
        ),
    }


def test_flatten_produces_one_row_per_variable():
    df = pd.DataFrame(
        [
            _row("s1", 1700000000, "Carbon dioxide", "ppm", 600.0),
            _row("s1", 1700000600, "Carbon dioxide", "ppm", 650.0),
        ]
    )
    out = _flatten(df)
    assert list(out.columns) == ["sensorid", "time", "variable", "unit", "value"]
    assert len(out) == 2
    assert out["variable"].unique().tolist() == ["Carbon dioxide"]


def test_flatten_converts_epoch_to_tz_aware_local_time():
    df = pd.DataFrame([_row("s1", 1700000000, "Temperature", "°C", 22.0)])
    out = _flatten(df)
    assert out["time"].dt.tz is not None
    assert str(out["time"].dt.tz) == "Australia/Melbourne"


def test_flatten_skips_rows_with_multiple_variables_correctly():
    """A row with several variables should explode into several output rows."""
    payload = json.dumps(
        [
            {
                "values": [{"time": 1700000000, "value": 600.0}],
                "variable": {"name": "Carbon dioxide", "unit": "ppm"},
            },
            {
                "values": [{"time": 1700000000, "value": 22.0}],
                "variable": {"name": "Temperature", "unit": "°C"},
            },
        ]
    )
    df = pd.DataFrame([{"sensorid": "s1", "jsondata": payload}])
    out = _flatten(df)
    assert len(out) == 2
    assert set(out["variable"]) == {"Carbon dioxide", "Temperature"}


def test_flatten_skips_failed_reads_with_no_value():
    """A 'values' entry with only a time (+ optional errorCode) is a failed
    sensor read, not a measurement, and must not crash or be kept."""
    payload = json.dumps(
        [
            {
                "values": [{"time": 1700000000, "value": 600.0}],
                "variable": {"name": "Carbon dioxide", "unit": "ppm"},
            },
            {
                "values": [{"time": 1700000000, "errorCode": 8}],
                "variable": {"name": "Temperature", "unit": "°C"},
            },
        ]
    )
    df = pd.DataFrame([{"sensorid": "s1", "jsondata": payload}])
    out = _flatten(df)
    assert len(out) == 1
    assert out["variable"].tolist() == ["Carbon dioxide"]


def test_flatten_skips_malformed_json_without_crashing():
    df = pd.DataFrame(
        [
            _row("s1", 1700000000, "Carbon dioxide", "ppm", 600.0),
            {"sensorid": "s2", "jsondata": "not valid json"},
        ]
    )
    out = _flatten(df)
    assert len(out) == 1
    assert out["sensorid"].tolist() == ["s1"]


def test_pivot_variable_filters_to_requested_variable():
    df = pd.DataFrame(
        [
            _row("s1", 1700000000, "Carbon dioxide", "ppm", 600.0),
            _row("s1", 1700000000, "Temperature", "°C", 22.0),
        ]
    )
    long_df = _flatten(df)
    co2 = pivot_variable(long_df, "Carbon dioxide")
    assert list(co2.columns) == ["sensorid", "time", "value"]
    assert co2["value"].tolist() == [600.0]


def test_pivot_variable_raises_on_unknown_variable():
    df = pd.DataFrame([_row("s1", 1700000000, "Carbon dioxide", "ppm", 600.0)])
    long_df = _flatten(df)
    with pytest.raises(ValueError):
        pivot_variable(long_df, "Not A Real Variable")


def test_load_env_sensor_log_keeps_digit_only_sensorid_as_string(tmp_path):
    """sensorid is an opaque id, e.g. "6012002000869" -- pandas' default
    type inference would read an all-digit column as int64, which then
    fails to match occupancy.sensors.SENSOR_LABELS (string keys) and can't
    be used as a JSON dict key. Must stay a string end to end."""
    csv_path = tmp_path / "env.csv"
    csv_path.write_text(
        'sensorid,jsondata\n'
        '"6012002000869","'
        + json.dumps(
            [
                {
                    "values": [{"time": 1700000000, "value": 600.0}],
                    "variable": {"name": "Carbon dioxide", "unit": "ppm"},
                }
            ]
        ).replace('"', '""')
        + '"\n'
    )
    out = load_env_sensor_log(csv_path)
    assert not pd.api.types.is_numeric_dtype(out["sensorid"])
    assert out["sensorid"].iloc[0] == "6012002000869"


def test_sensor_ids_returns_each_sensor_once():
    df = pd.DataFrame(
        [
            _row("s2", 1700000000, "Carbon dioxide", "ppm", 600.0),
            _row("s1", 1700000600, "Carbon dioxide", "ppm", 650.0),
            _row("s2", 1700001200, "Carbon dioxide", "ppm", 700.0),
        ]
    )
    long_df = _flatten(df)
    assert set(sensor_ids(long_df)) == {"s1", "s2"}
    assert len(sensor_ids(long_df)) == 2
