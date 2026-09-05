"""Loading and basic validation of the raw occupancy sensor log.

The source file is an *event log*: one row is written each time a room's
occupancy state changes (not a fixed time grid), so every downstream
consumer must treat gaps between rows as "state held" rather than "no
data" up to some limit (see :mod:`occupancy.preprocessing`).
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Iterator, Union

import pandas as pd

#: Columns required to be present in the raw CSV. If the supplied file is
#: missing any of these, :func:`load_occupancy_log` raises rather than
#: silently continuing with an incomplete schema.
REQUIRED_COLUMNS = (
    "deviceid",
    "floorspaceid",
    "occupancystatus",
    "headcount",
    "collecteddate",
)

#: Occupancy states observed in the raw data. Kept here (rather than only
#: implied by the data) so a status outside this set is treated as a data
#: error, not silently ignored.
KNOWN_STATUSES = ("CurrentlyOccupied", "RecentlyOccupied", "NotOccupied")

PathLike = Union[str, Path]


def load_occupancy_log(
    path: PathLike,
    chunksize: int | None = None,
) -> pd.DataFrame | Iterator[pd.DataFrame]:
    """Load the raw occupancy event-log CSV.

    Args:
        path: Path to the occupancy CSV (e.g.
            ``data/raw/5occupancySensor_MayToDec2024_9MRows.csv``, or the
            small ``data/processed/occupancy_sample_50k.csv`` for
            development).
        chunksize: If given, the file is read lazily in chunks of this many
            rows (see `pandas.read_csv` `chunksize`) and an iterator of
            validated DataFrames is returned instead of a single DataFrame.
            Use this for the full ~1.5 GB file; omit it for the small
            sample file.

    Returns:
        A DataFrame (or iterator of DataFrames, one per chunk) with:

        - ``collecteddate`` parsed to a timezone-aware ``datetime64``
          column (the raw file stores a UTC offset, e.g. ``+11``, which
          pandas preserves rather than silently converting to naive local
          time).
        - ``occupancystatus`` validated against :data:`KNOWN_STATUSES`.

    Raises:
        ValueError: If any column in :data:`REQUIRED_COLUMNS` is missing,
            or if ``occupancystatus`` contains a value outside
            :data:`KNOWN_STATUSES` (this would indicate either a corrupt
            file or an undocumented status the model has not been designed
            to handle -- both should fail loudly rather than be guessed
            at).
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Occupancy file not found: {path}. See data/raw/README.md "
            "for where to obtain it."
        )

    if chunksize is not None:
        raw_iter = pd.read_csv(path, chunksize=chunksize)
        return (_validate(_parse_dates(chunk)) for chunk in raw_iter)

    df = pd.read_csv(path)
    return _validate(_parse_dates(df))


#: Timestamps in the raw file carry a fixed UTC offset (e.g. "+11"), but the
#: offset itself changes across the file because Melbourne observes
#: daylight saving (+11 AEDT / +10 AEST). Letting pandas infer each
#: timestamp's dtype independently in that situation produces an *object*
#: column of per-row Timestamps rather than one proper tz-aware
#: datetime64 column -- so every subsequent `.dt` operation breaks. Parsing
#: with utc=True first (normalising everything to a single UTC instant),
#: then converting to a named zone, avoids that trap and also makes
#: "local time of day" well-defined for feature engineering later.
LOCAL_TZ = "Australia/Melbourne"


def _parse_dates(df: pd.DataFrame) -> pd.DataFrame:
    """Parse timestamp columns to a single tz-aware dtype (see LOCAL_TZ)."""
    for col in ("collecteddate", "occupancystatuschangedate"):
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], utc=True).dt.tz_convert(LOCAL_TZ)
    return df


def _validate(df: pd.DataFrame) -> pd.DataFrame:
    """Check schema and status values for one loaded chunk/DataFrame.

    Kept as a private helper (rather than inline in
    :func:`load_occupancy_log`) so both the whole-file and chunked code
    paths apply exactly the same checks.
    """
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(
            f"Occupancy CSV is missing expected column(s): {missing}. "
            f"Expected at least: {REQUIRED_COLUMNS}"
        )

    bad_status = ~df["occupancystatus"].isin(KNOWN_STATUSES)
    if bad_status.any():
        bad_values = sorted(df.loc[bad_status, "occupancystatus"].unique())
        raise ValueError(
            "Found occupancystatus value(s) not in KNOWN_STATUSES: "
            f"{bad_values}. Update KNOWN_STATUSES if this is a legitimate "
            "new status, after checking how it should be interpreted."
        )

    return df


def room_ids(df: pd.DataFrame) -> Iterable[str]:
    """Return the distinct room identifiers (``floorspaceid``) present.

    Args:
        df: A DataFrame as returned by :func:`load_occupancy_log`.

    Returns:
        The unique ``floorspaceid`` values, in first-seen order.
    """
    return pd.unique(df["floorspaceid"])
