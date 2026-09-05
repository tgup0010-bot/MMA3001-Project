"""Convert the occupancy event log into a regular per-room time grid.

The raw data is *event-driven*: a row exists only when a room's occupancy
status changes, so timestamps are irregular and gaps of any length can
appear (a room that is quiet overnight simply has no rows). Naively
resampling this (e.g. ``DataFrame.resample().ffill()``) would silently
assume a room's state holds forever across *any* gap, including gaps caused
by a sensor going offline for months (as is the case for part of this
dataset). :func:`resample_occupancy` instead forward-fills only up to a
configurable ``max_gap``, leaving longer gaps as missing (``NaN``) so they
are visible to validation rather than quietly fabricated.
"""

from __future__ import annotations

import pandas as pd

#: Default bin width for the regular time grid. 15 minutes balances
#: resolution against the number of rows produced (see the report's
#: optimisation/sensitivity-analysis section for the accuracy/cost
#: trade-off across bin sizes).
DEFAULT_BIN_SIZE = "15min"

#: Default maximum gap that is forward-filled. Chosen to be long enough to
#: bridge normal quiet periods (e.g. an unoccupied room overnight) but
#: short enough not to paper over a sensor outage as if the last known
#: state simply continued. This is an explicit, documented assumption, not
#: a derived value -- it should be revisited against real gap-length
#: statistics before being relied on for a specific room.
DEFAULT_MAX_GAP = "2h"


def resample_occupancy(
    df: pd.DataFrame,
    *,
    bin_size: str = DEFAULT_BIN_SIZE,
    occupied_statuses: tuple[str, ...] = ("CurrentlyOccupied",),
    max_gap: str | pd.Timedelta = DEFAULT_MAX_GAP,
    room_col: str = "floorspaceid",
    time_col: str = "collecteddate",
    status_col: str = "occupancystatus",
) -> pd.DataFrame:
    """Resample an occupancy event log onto a regular per-room time grid.

    Args:
        df: Occupancy event log as returned by
            :func:`occupancy.data_loading.load_occupancy_log`. Must contain
            ``room_col``, ``time_col`` and ``status_col``.
        bin_size: Pandas frequency string for the output grid, e.g.
            ``"15min"``, ``"30min"``, ``"1h"``.
        occupied_statuses: Which values of ``status_col`` count as
            "occupied" for the binary label. Defaults to only
            ``"CurrentlyOccupied"`` (a strict definition); pass
            ``("CurrentlyOccupied", "RecentlyOccupied")`` for a looser one
            that also counts the sensor's post-departure cooldown state as
            occupied.
        max_gap: The longest real gap between consecutive observations
            that is still forward-filled. Grid points falling in a larger
            gap are set to ``NaN`` (missing) rather than assumed occupied
            or unoccupied.
        room_col: Column identifying the room/zone.
        time_col: Column holding the (timezone-aware) event timestamp.
        status_col: Column holding the raw occupancy status string.

    Returns:
        A tidy DataFrame with one row per (room, time bin), columns
        ``[room_col, time_col, "occupied"]``, where ``"occupied"`` is
        ``1.0``, ``0.0``, or ``NaN`` (gap longer than ``max_gap``).

    Notes:
        Rooms are processed independently (looped in Python) rather than
        vectorised across the whole file at once. With only 5 rooms in
        this dataset the loop overhead is negligible, and it keeps the
        per-room grid construction (which depends on that room's own
        min/max timestamp) simple and easy to verify.
    """
    max_gap = pd.Timedelta(max_gap)

    work = (
        df[[room_col, time_col, status_col]]
        .dropna(subset=[time_col])
        .copy()
    )
    work["occupied"] = work[status_col].isin(occupied_statuses).astype(float)

    # If a room logged two events at the exact same timestamp, keep the
    # later-recorded one as the authoritative state at that instant.
    work = work.sort_values([room_col, time_col]).drop_duplicates(
        subset=[room_col, time_col], keep="last"
    )

    grids = []
    for room, g in work.groupby(room_col, sort=False):
        g = g.sort_values(time_col)
        tz = g[time_col].dt.tz
        start = g[time_col].min().floor(bin_size)
        end = g[time_col].max().ceil(bin_size)
        grid = pd.DataFrame(
            {time_col: pd.date_range(start, end, freq=bin_size, tz=tz)}
        )

        # merge_asof with a tolerance is the vectorised equivalent of
        # "look up the most recent known state, but only if it's not too
        # stale" -- this is what enforces max_gap.
        merged = pd.merge_asof(
            grid,
            g[[time_col, "occupied"]],
            on=time_col,
            direction="backward",
            tolerance=max_gap,
        )
        merged[room_col] = room
        grids.append(merged)

    out = pd.concat(grids, ignore_index=True)
    return out[[room_col, time_col, "occupied"]]


def coverage_summary(resampled: pd.DataFrame, room_col: str = "floorspaceid") -> pd.DataFrame:
    """Summarise data completeness per room after resampling.

    Args:
        resampled: Output of :func:`resample_occupancy`.
        room_col: Column identifying the room/zone.

    Returns:
        A DataFrame indexed by room with columns ``n_bins``,
        ``n_missing`` and ``pct_missing`` -- useful both for choosing
        ``max_gap``/``bin_size`` and for reporting data-quality
        limitations honestly in the write-up.
    """
    grp = resampled.groupby(room_col)["occupied"]
    summary = grp.agg(n_bins="size", n_missing=lambda s: s.isna().sum())
    summary["pct_missing"] = 100 * summary["n_missing"] / summary["n_bins"]
    return summary
