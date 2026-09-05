"""Feature engineering for next-interval occupancy prediction.

Turns the regular per-room grid produced by
:func:`occupancy.preprocessing.resample_occupancy` into a supervised
learning table: one row per (room, time bin) with predictor columns and a
``target`` column equal to the *next* bin's occupancy state.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

#: Numeric feature columns produced by :func:`build_supervised_dataset`.
#: Exposed as a constant so model code (which needs to know exactly which
#: columns to feed a classifier) and feature code can't silently drift
#: apart.
NUMERIC_FEATURES = (
    "occupied_now",
    "occupied_lag1",
    "rolling_mean_1h",
    "hour_sin",
    "hour_cos",
    "dow_sin",
    "dow_cos",
    "is_weekend",
)

CATEGORICAL_FEATURES = ("floorspaceid",)


def _add_calendar_features(df: pd.DataFrame, time_col: str) -> pd.DataFrame:
    """Add cyclical time-of-day / day-of-week features.

    Sine/cosine encoding (rather than the raw hour 0-23 or weekday 0-6) is
    used so that, e.g., 23:00 and 00:00 are numerically close -- a plain
    integer encoding would tell a linear model they are as different as
    possible.

    Note:
        These describe the *current* bin's calendar time, used as a proxy
        for the (very slightly later) bin being predicted. At the 15-minute
        default bin size the two are almost always in the same hour, so
        this simplification has negligible effect while keeping the
        feature-building code single-timestamp.
    """
    t = df[time_col]
    hour_frac = t.dt.hour + t.dt.minute / 60.0
    df["hour_sin"] = np.sin(2 * np.pi * hour_frac / 24)
    df["hour_cos"] = np.cos(2 * np.pi * hour_frac / 24)

    dow = t.dt.dayofweek  # Monday=0 ... Sunday=6
    df["dow_sin"] = np.sin(2 * np.pi * dow / 7)
    df["dow_cos"] = np.cos(2 * np.pi * dow / 7)
    df["is_weekend"] = (dow >= 5).astype(float)
    return df


def build_supervised_dataset(
    resampled: pd.DataFrame,
    *,
    room_col: str = "floorspaceid",
    time_col: str = "collecteddate",
    status_col: str = "occupied",
    rolling_window: int = 4,
) -> tuple[pd.DataFrame, dict]:
    """Build a supervised (features + target) table from the resampled grid.

    Args:
        resampled: Output of
            :func:`occupancy.preprocessing.resample_occupancy`.
        room_col: Column identifying the room/zone.
        time_col: Column holding the bin's start time.
        status_col: Binary/`NaN` occupancy column (as produced by
            ``resample_occupancy``).
        rolling_window: Number of trailing bins (including the current
            one) averaged into ``rolling_mean_1h`` -- 4 bins at the default
            15-minute bin size covers the last hour.

    Returns:
        A tuple ``(table, info)`` where:

        - ``table`` has columns ``room_col``, ``time_col``, every name in
          :data:`NUMERIC_FEATURES`, and ``target`` (the *next* bin's
          occupancy, 0.0/1.0) -- rows with any missing feature or target
          have been dropped (see ``info``).
        - ``info`` reports how many rows were dropped and why, so this is
          visible in the report rather than silently absorbed. Rows are
          dropped only because the underlying sensor data was itself
          missing (see :func:`occupancy.preprocessing.resample_occupancy`'s
          ``max_gap``) -- this function does not impute or guess a
          missing occupancy state, since doing so would make the model
          circular (predicting an occupancy value manufactured by the
          feature pipeline itself).

    Raises:
        ValueError: If ``resampled`` is empty after grouping, e.g. because
            ``room_col`` has no rows.
    """
    if resampled.empty:
        raise ValueError("resample_occupancy output is empty; nothing to build features from.")

    df = resampled.sort_values([room_col, time_col]).reset_index(drop=True).copy()
    df = _add_calendar_features(df, time_col)

    grp = df.groupby(room_col)[status_col]
    df["occupied_now"] = df[status_col]
    df["occupied_lag1"] = grp.shift(1)
    df["rolling_mean_1h"] = grp.transform(
        lambda s: s.rolling(rolling_window, min_periods=rolling_window).mean()
    )
    # The target is the *next* bin's state -- shift(-1) looks forward, which
    # is exactly what we want to predict, and is only safe because it's
    # computed per-room (groupby) so a room boundary never leaks into
    # another room's "next" value.
    df["target"] = grp.shift(-1)

    feature_cols = list(NUMERIC_FEATURES)
    needed = feature_cols + ["target"]

    n_before = len(df)
    table = df.dropna(subset=needed).reset_index(drop=True)
    n_after = len(table)

    info = {
        "rows_before": n_before,
        "rows_after": n_after,
        "rows_dropped": n_before - n_after,
        "pct_dropped": 100.0 * (n_before - n_after) / n_before if n_before else 0.0,
    }

    return table[[room_col, time_col, *feature_cols, "target"]], info
