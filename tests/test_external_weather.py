"""Unit tests for occupancy.external_weather.

Uses a small hand-built BoM-format CSV (not the real downloaded files) so
the suite runs in well under a second and does not depend on network
access or the (gitignored) data/raw/bom directory.
"""

import pandas as pd
import pytest

from occupancy.external_weather import load_bom_month, load_bom_weather

_HEADER = (
    '"Daily Weather Observations for Moorabbin, Victoria for July 2025"\r\n'
    '"Prepared at 00:00 UTC on Monday 1 September 2025   IDCJDW3052.202507"\r\n'
    '"Copyright 2003 Commonwealth Bureau of Meteorology"\r\n'
    '"Some cloud observations are from automated equipment."\r\n'
    '"Observations were drawn from Moorabbin Airport {station 086077}"\r\n'
    "\r\n"
    ',"Date","Minimum temperature (\xb0C)","Maximum temperature (\xb0C)","Rainfall (mm)",'
    '"Evaporation (mm)","Sunshine (hours)","Direction of maximum wind gust ",'
    '"Speed of maximum wind gust (km/h)","Time of maximum wind gust",'
    '"9am Temperature (\xb0C)","9am relative humidity (%)","9am cloud amount (oktas)",'
    '"9am wind direction","9am wind speed (km/h)","9am MSL pressure (hPa)",'
    '"3pm Temperature (\xb0C)","3pm relative humidity (%)","3pm cloud amount (oktas)",'
    '"3pm wind direction","3pm wind speed (km/h)","3pm MSL pressure (hPa)"\r\n'
)


def _bom_csv(rows: list[str]) -> bytes:
    return (_HEADER + "\r\n".join(rows) + "\r\n").encode("latin-1")


def test_load_bom_month_parses_dates_and_numeric_columns(tmp_path):
    csv_path = tmp_path / "IDCJDW3052.202507.csv"
    csv_path.write_bytes(
        _bom_csv(
            [
                ",2025-07-1,3.7,14.0,0,,,SE,26,12:39,6.6,90,,NNE,2,1027.6,11.0,75,7,SW,15,1025.2",
                ",2025-07-2,4.7,15.3,0,,,ESE,22,13:07,7.8,94,8,NNE,6,1027.8,13.3,66,,WSW,15,1025.1",
            ]
        )
    )
    df = load_bom_month(csv_path)
    assert list(df["date"]) == [pd.Timestamp("2025-07-01").date(), pd.Timestamp("2025-07-02").date()]
    assert df["temp_9am_c"].tolist() == [6.6, 7.8]
    assert df["humidity_9am_pct"].tolist() == [90, 94]


def test_load_bom_month_treats_blank_reading_as_nan_not_zero(tmp_path):
    csv_path = tmp_path / "IDCJDW3052.202507.csv"
    csv_path.write_bytes(
        _bom_csv(
            [",2025-07-1,3.7,14.0,0,,,SE,26,12:39,6.6,90,,NNE,2,1027.6,11.0,75,7,SW,15,1025.2"]
        )
    )
    df = load_bom_month(csv_path)
    assert pd.isna(df["cloud_9am_oktas"].iloc[0])
    assert pd.isna(df["evaporation_mm"].iloc[0])


def test_load_bom_weather_concatenates_all_files_in_directory(tmp_path):
    (tmp_path / "IDCJDW3052.202507.csv").write_bytes(
        _bom_csv([",2025-07-1,3.7,14.0,0,,,SE,26,12:39,6.6,90,,NNE,2,1027.6,11.0,75,7,SW,15,1025.2"])
    )
    (tmp_path / "IDCJDW3052.202508.csv").write_bytes(
        _bom_csv([",2025-08-1,2.2,13.4,0,,,SE,26,12:39,6.6,90,,NNE,2,1027.6,11.0,75,7,SW,15,1025.2"])
    )
    df = load_bom_weather(tmp_path)
    assert len(df) == 2
    assert list(df["date"]) == sorted(df["date"])


def test_load_bom_weather_raises_when_no_files_found(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_bom_weather(tmp_path)
