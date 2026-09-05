"""Baseline predictors for next-interval occupancy.

Two baselines, from simplest to more informed, are provided so the ML
model's benefit (or lack of it) can be judged against a real reference
point rather than an arbitrary number -- this is the "alternative
solutions" comparison the project report is required to make.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def persistence_predict_proba(table: pd.DataFrame) -> np.ndarray:
    """Naive "nothing changes" baseline: predict next = current state.

    Args:
        table: A supervised table as returned by
            :func:`occupancy.features.build_supervised_dataset` (must
            contain an ``occupied_now`` column).

    Returns:
        Predicted P(occupied at next bin) for each row, equal to
        ``occupied_now`` (i.e. 0.0 or 1.0 -- this baseline is not
        probabilistic, it just repeats the last known state).
    """
    return table["occupied_now"].to_numpy(dtype=float)


class MarkovBaseline:
    """First-order Markov chain, conditioned on room and hour-of-day.

    Estimates ``P(occupied_next=1 | room, hour_of_day, occupied_now)``
    directly from training data by counting/averaging observed
    transitions, then looks up that probability at prediction time. This
    is a classical statistical/numerical baseline: no gradient-based
    fitting, no hyperparameters beyond the discretisation already fixed by
    the time-grid bin size.
    """

    def __init__(self) -> None:
        self.table_: pd.Series | None = None
        self.global_rate_: float | None = None

    def fit(
        self,
        table: pd.DataFrame,
        *,
        room_col: str = "floorspaceid",
        time_col: str = "collecteddate",
    ) -> "MarkovBaseline":
        """Estimate transition probabilities from a training table.

        Args:
            table: Training supervised table (see
                :func:`occupancy.features.build_supervised_dataset`); must
                contain ``room_col``, ``time_col``, ``occupied_now`` and
                ``target``.
            room_col: Column identifying the room/zone.
            time_col: Column holding the bin's start time.

        Returns:
            ``self``, fitted.
        """
        d = table.copy()
        d["hour"] = d[time_col].dt.hour
        self.global_rate_ = float(d["target"].mean())
        self.table_ = (
            d.groupby([room_col, "hour", "occupied_now"])["target"]
            .mean()
            .rename("p_next_occupied")
        )
        return self

    def predict_proba(
        self,
        table: pd.DataFrame,
        *,
        room_col: str = "floorspaceid",
        time_col: str = "collecteddate",
    ) -> np.ndarray:
        """Predict P(occupied at next bin) for each row of ``table``.

        A (room, hour, occupied_now) combination never seen during
        ``fit`` falls back to the overall training-set occupancy rate
        (``global_rate_``) rather than raising or returning ``NaN`` --
        this can genuinely happen for a room/hour combination with very
        little training data, and failing silently to a sensible default
        is preferable to crashing evaluation.

        Args:
            table: Table to predict on; same column requirements as
                ``fit``.
            room_col: Column identifying the room/zone.
            time_col: Column holding the bin's start time.

        Returns:
            Predicted probabilities, one per row of ``table``.

        Raises:
            RuntimeError: If called before :meth:`fit`.
        """
        if self.table_ is None:
            raise RuntimeError("MarkovBaseline.predict_proba called before fit().")

        d = table.copy()
        d["hour"] = d[time_col].dt.hour
        keys = pd.MultiIndex.from_frame(d[[room_col, "hour", "occupied_now"]])
        p = self.table_.reindex(keys).to_numpy()
        return np.where(np.isnan(p), self.global_rate_, p)
