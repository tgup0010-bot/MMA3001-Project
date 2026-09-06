"""Live demo: replay real history, then predict occupancy for scenarios you type in.

Loads a pre-trained model (`models/occupancy_model.joblib`) rather than the
1.5GB raw file, so it starts in under a second -- suitable for running live
in a presentation. If that file doesn't exist yet, run
`scripts/train_and_save_model.py` first (one-off, ~2-3 minutes).

Usage:
    python scripts/demo.py
"""

from __future__ import annotations

import math
from pathlib import Path

import joblib
import pandas as pd

from occupancy.rooms import ROOM_LABELS, room_label

MODEL_PATH = Path("models/occupancy_model.joblib")
EXAMPLES_PATH = Path("models/demo_examples.csv")

DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]


def replay_real_examples(bundle: dict) -> None:
    """Part 1: show the model's predictions on real, held-out test data."""
    print("=" * 72)
    print("PART 1 -- Replaying real examples from the test set")
    print("(these intervals were never seen during training)")
    print("=" * 72)

    # Parsed the same way as occupancy.data_loading._parse_dates: the
    # saved examples' timestamps carry a UTC offset that changes across
    # the file (Melbourne daylight saving), so pandas' plain parse_dates
    # can silently leave this column as unparsed strings instead of
    # raising -- see that function's docstring for the full explanation.
    examples = pd.read_csv(EXAMPLES_PATH)
    examples["collecteddate"] = pd.to_datetime(
        examples["collecteddate"], utc=True
    ).dt.tz_convert("Australia/Melbourne")
    pipeline = bundle["logreg"]
    x_cols = bundle["feature_columns"]

    proba = pipeline.predict_proba(examples[x_cols])[:, 1]
    n_correct = 0
    for i, row in examples.iterrows():
        p = proba[i]
        predicted_occupied = p >= 0.5
        actual_occupied = row["target"] == 1
        is_correct = predicted_occupied == actual_occupied
        n_correct += is_correct

        print(
            f"  {room_label(row['floorspaceid']):<30} "
            f"{row['collecteddate']:%a %d-%b %H:%M}  "
            f"now={'occupied' if row['occupied_now'] == 1 else 'empty':<8}  "
            f"model: {p * 100:5.1f}% -> {'occupied' if predicted_occupied else 'empty':<8}  "
            f"actual: {'occupied' if actual_occupied else 'empty':<8}  "
            f"[{'correct' if is_correct else 'WRONG'}]"
        )

    print(f"\n  {n_correct}/{len(examples)} correct on this sample "
          f"(the full test-set accuracy, {len(proba):,}x more rows, is in "
          "reports/experiment_results.json).\n")


def _build_query_row(x_cols: list[str], *, room: str, hour: int, dow: int,
                      occupied_now: bool, rolling_mean: float) -> pd.DataFrame:
    """Turn a plain-English scenario into the model's expected feature row.

    Simplification for ad-hoc queries: `occupied_lag1` (the state one bin
    earlier) isn't something a user would naturally know off-hand, so it
    is approximated as equal to `occupied_now`. This is a demo
    convenience, not used anywhere in the actual validated pipeline
    (`occupancy.features.build_supervised_dataset` always computes it from
    real history).
    """
    row = {
        "occupied_now": float(occupied_now),
        "occupied_lag1": float(occupied_now),
        "rolling_mean_1h": rolling_mean,
        "hour_sin": math.sin(2 * math.pi * hour / 24),
        "hour_cos": math.cos(2 * math.pi * hour / 24),
        "dow_sin": math.sin(2 * math.pi * dow / 7),
        "dow_cos": math.cos(2 * math.pi * dow / 7),
        "is_weekend": float(dow >= 5),
        "floorspaceid": room,
    }
    return pd.DataFrame([row])[x_cols]


def interactive_loop(bundle: dict) -> None:
    """Part 2: let the user type in a scenario and get a live prediction."""
    print("=" * 72)
    print("PART 2 -- Try your own scenario")
    print("=" * 72)
    print("(leave the room choice blank at any time to quit)\n")

    rooms = list(ROOM_LABELS.keys()) + [
        r for r in [
            "941f0352-bb69-47d0-a9c5-ac29eb4f7319",
            "0f3f37f8-0335-41e0-ae87-a1d677dfc7d7",
            "3e5a14df-4b14-4ea5-bb9d-46577c6dc9c8",
        ]
    ]
    pipeline = bundle["logreg"]
    x_cols = bundle["feature_columns"]

    while True:
        print("Rooms:")
        for i, r in enumerate(rooms, start=1):
            print(f"  {i}. {room_label(r)}")
        choice = input("Pick a room number (blank to quit): ").strip()
        if not choice:
            print("Bye!")
            return
        room = rooms[int(choice) - 1]

        hour = int(input("Hour of day right now, 0-23 (e.g. 14 for 2pm): ").strip())
        print("Days: " + ", ".join(f"{i + 1}={d}" for i, d in enumerate(DAYS)))
        dow = int(input("Day of week, 1-7: ").strip()) - 1
        occupied_now = input("Is the room occupied right now? (y/n): ").strip().lower().startswith("y")
        busy_raw = input(
            "Roughly how busy has it been in the last hour, 0-100% "
            "(blank = same as 'right now'): "
        ).strip()
        rolling_mean = float(busy_raw) / 100 if busy_raw else float(occupied_now)

        query = _build_query_row(
            x_cols, room=room, hour=hour, dow=dow,
            occupied_now=occupied_now, rolling_mean=rolling_mean,
        )
        p = pipeline.predict_proba(query)[0, 1]

        print(
            f"\n  --> Model predicts {p * 100:.1f}% chance that "
            f"{room_label(room)} is occupied in the next 15 minutes.\n"
        )


def main() -> None:
    if not MODEL_PATH.exists():
        raise SystemExit(
            f"{MODEL_PATH} not found.\n"
            "Run `python scripts/train_and_save_model.py` first "
            "(one-off, ~2-3 minutes) -- or if you cloned this repo without "
            "the raw data, the model file should already be committed; "
            "check you're running this from the repo root."
        )
    bundle = joblib.load(MODEL_PATH)

    replay_real_examples(bundle)
    try:
        interactive_loop(bundle)
    except (KeyboardInterrupt, EOFError):
        print("\nBye!")


if __name__ == "__main__":
    main()
