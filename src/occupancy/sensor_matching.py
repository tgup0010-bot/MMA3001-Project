"""Data-driven attempt to link environmental sensors to occupancy zones.

The supplied location spreadsheet cannot tell us which environmental sensor
sits in which occupancy zone (see the project README, Known data
limitations): only 2 of 5 occupancy zones and 1 of 5 environmental sensors
resolve to a named room, and they don't overlap. Rather than assume a
mapping, this module tests whether the *data itself* gives evidence of a
room correspondence -- if a room's occupancy pattern reliably coincides
with (or slightly precedes) a spike in a particular sensor's CO2 reading,
that is independent, data-driven evidence of co-location, not a guess.

Method
------
For each (zone, sensor) pair, both series are resampled onto the same
regular time grid and correlated using the *change* in the environmental
variable (its first difference), not its raw level. This matters: CO2,
temperature etc. all follow a shared diurnal cycle (higher during work
hours in every room), so correlating raw *levels* would find every
zone/sensor pair "related" simply because they're all busier on weekday
afternoons -- a confound, not evidence of co-location. Correlating the
*rate of change* is far more specific to actual arrival/departure events.
A small range of lags is searched (a sensor may lag a few minutes behind
the occupancy event log) and the strongest absolute correlation is kept.

This analysis is exploratory, not a substitute for verified metadata: a
correlation found here is evidence worth reporting, not proof of a shared
room. See the project report for the full discussion and caveats.
"""

from __future__ import annotations

import pandas as pd

from occupancy.env_sensors import pivot_variable

#: Default number of grid bins searched in each direction when looking for
#: the lag at which a sensor's reading best tracks an occupancy zone. At
#: the default 15-minute bin size this covers a ±1 hour window, generous
#: enough for HVAC/room mixing delay without being so wide it starts
#: matching unrelated events by chance.
DEFAULT_MAX_LAG_BINS = 4

#: Minimum number of overlapping, non-missing (occupancy, variable) bins
#: required before a correlation is trusted at all. Below this, a
#: correlation is reported as such but flagged as low-confidence in the
#: interpretation.
MIN_OVERLAP_BINS = 20


def resample_env_variable(
    pivoted: pd.DataFrame,
    sensorid: str,
    bin_size: str,
) -> pd.Series:
    """Resample one sensor's readings of one variable onto a regular grid.

    Args:
        pivoted: Output of :func:`occupancy.env_sensors.pivot_variable`
            (a single variable, all sensors).
        sensorid: Which sensor to extract.
        bin_size: Pandas frequency string, e.g. ``"15min"`` -- should match
            the bin size used for :func:`occupancy.preprocessing.resample_occupancy`
            so the two series can be compared bin-for-bin.

    Returns:
        A ``pandas.Series`` indexed by time bin, mean-aggregated within
        each bin. Bins with no reading are left ``NaN`` (not
        forward-filled) since a fabricated reading would distort the
        correlation, not just the occupancy label.
    """
    sub = pivoted[pivoted["sensorid"] == sensorid].set_index("time")["value"]
    return sub.resample(bin_size).mean()


def lagged_correlation(
    occupied: pd.Series,
    variable: pd.Series,
    max_lag_bins: int = DEFAULT_MAX_LAG_BINS,
    use_diff: bool = True,
    min_overlap_bins: int = MIN_OVERLAP_BINS,
) -> dict:
    """Find the strongest correlation between occupancy and a variable across a small lag search.

    Args:
        occupied: Binary occupancy series (0.0/1.0), time-indexed, as
            produced by :func:`occupancy.preprocessing.resample_occupancy`
            for one room.
        variable: An environmental variable series, time-indexed on the
            same grid, as produced by :func:`resample_env_variable`.
        max_lag_bins: Search lags from ``-max_lag_bins`` to
            ``+max_lag_bins`` (in grid bins). A positive lag means the
            variable is shifted to *follow* occupancy (sensor reacts after
            the fact); negative means it *leads*.
        use_diff: If True (default), correlate against the variable's
            first difference (rate of change) rather than its raw level --
            see the module docstring for why this matters.
        min_overlap_bins: Minimum overlapping non-missing bins required to
            trust a candidate lag (default :data:`MIN_OVERLAP_BINS`,
            appropriate for the real dataset's scale -- tests on small
            hand-built series should pass a smaller value).

    Returns:
        A dict with ``best_lag`` (int or None if no lag had enough
        overlapping data), ``correlation`` (float, Pearson r, NaN if
        undetermined), and ``n_bins`` (overlapping non-missing bins used
        for the best lag).
    """
    series = variable.diff() if use_diff else variable

    candidates = []
    for lag in range(-max_lag_bins, max_lag_bins + 1):
        # shift(-lag) aligns occupied[t] with variable[t + lag]: a positive
        # lag compares occupancy *now* to the variable *lag bins later*,
        # i.e. "the variable follows occupancy" -- matching the docstring.
        shifted = series.shift(-lag)
        aligned = pd.concat(
            [occupied.rename("occupied"), shifted.rename("variable")], axis=1
        ).dropna()
        if len(aligned) < min_overlap_bins:
            continue
        if aligned["occupied"].nunique() < 2 or aligned["variable"].nunique() < 2:
            continue  # a constant series has an undefined correlation
        corr = aligned["occupied"].corr(aligned["variable"])
        candidates.append((lag, corr, len(aligned)))

    if not candidates:
        return {"best_lag": None, "correlation": float("nan"), "n_bins": 0}

    best_lag, best_corr, n = max(candidates, key=lambda c: abs(c[1]))
    return {"best_lag": best_lag, "correlation": best_corr, "n_bins": n}


def build_correlation_matrix(
    occ_resampled: pd.DataFrame,
    env_long: pd.DataFrame,
    variable: str,
    bin_size: str,
    max_lag_bins: int = DEFAULT_MAX_LAG_BINS,
    room_col: str = "floorspaceid",
    time_col: str = "collecteddate",
    min_overlap_bins: int = MIN_OVERLAP_BINS,
) -> tuple[pd.DataFrame, dict]:
    """Correlate every (occupancy zone, environmental sensor) pair.

    Args:
        occ_resampled: Output of
            :func:`occupancy.preprocessing.resample_occupancy`, covering
            all rooms.
        env_long: Output of
            :func:`occupancy.env_sensors.load_env_sensor_log`.
        variable: Which environmental variable to test, e.g.
            ``"Carbon dioxide"``.
        bin_size: Must match the bin size used to produce ``occ_resampled``.
        max_lag_bins: See :func:`lagged_correlation`.
        room_col: Column in ``occ_resampled`` identifying the room/zone.
        time_col: Column in ``occ_resampled`` holding the time bin.
        min_overlap_bins: See :func:`lagged_correlation`.

    Returns:
        A tuple ``(matrix, detail)``:

        - ``matrix``: DataFrame indexed by room id, one column per sensor
          id, holding the best-lag Pearson correlation for that pair.
        - ``detail``: dict keyed by ``(room, sensorid)`` with the full
          :func:`lagged_correlation` result (including ``n_bins``), for
          honestly reporting confidence alongside the headline number.
    """
    pivoted = pivot_variable(env_long, variable)
    rooms = list(pd.unique(occ_resampled[room_col]))
    sensors = list(pd.unique(pivoted["sensorid"]))

    matrix = pd.DataFrame(index=rooms, columns=sensors, dtype=float)
    detail: dict = {}

    env_series_by_sensor = {
        sensorid: resample_env_variable(pivoted, sensorid, bin_size)
        for sensorid in sensors
    }

    for room in rooms:
        occ_series = (
            occ_resampled[occ_resampled[room_col] == room]
            .set_index(time_col)["occupied"]
        )
        for sensorid in sensors:
            result = lagged_correlation(
                occ_series,
                env_series_by_sensor[sensorid],
                max_lag_bins,
                min_overlap_bins=min_overlap_bins,
            )
            matrix.loc[room, sensorid] = result["correlation"]
            detail[(room, sensorid)] = result

    return matrix, detail


def best_match_per_room(matrix: pd.DataFrame, detail: dict) -> pd.DataFrame:
    """Summarise the strongest-correlated sensor for each room.

    Args:
        matrix: First element of :func:`build_correlation_matrix`'s return.
        detail: Second element of :func:`build_correlation_matrix`'s return.

    Returns:
        A DataFrame indexed by room, with columns ``best_sensor``,
        ``correlation``, ``best_lag`` and ``n_bins`` -- one row per room,
        picking the sensor with the largest absolute correlation. This is
        a candidate for "possibly co-located", not a confirmed match --
        see the module docstring's caveats.
    """
    rows = []
    for room in matrix.index:
        row = matrix.loc[room]
        if row.isna().all():
            rows.append(
                {"room": room, "best_sensor": None, "correlation": float("nan"), "best_lag": None, "n_bins": 0}
            )
            continue
        best_sensor = row.abs().idxmax()
        d = detail[(room, best_sensor)]
        rows.append(
            {
                "room": room,
                "best_sensor": best_sensor,
                "correlation": row[best_sensor],
                "best_lag": d["best_lag"],
                "n_bins": d["n_bins"],
            }
        )
    return pd.DataFrame(rows).set_index("room")
