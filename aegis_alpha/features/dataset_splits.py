from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SplitPlan:
    train_end: int
    validation_windows: list[tuple[int, int]]
    holdout: tuple[int, int]


def walk_forward_splits(n_rows: int, window: int = 4032, holdout: int = 90 * 288) -> SplitPlan:
    if n_rows < window * 4:
        train_end = int(n_rows * 0.70)
        val_start = train_end
        return SplitPlan(train_end, [(val_start, n_rows)], (val_start, n_rows))

    holdout_start = max(0, n_rows - holdout)
    train_end = max(window, holdout_start - window * 4)
    validation_windows = []
    cursor = train_end
    while cursor + window <= holdout_start:
        validation_windows.append((cursor, cursor + window))
        cursor += window
    if not validation_windows:
        validation_windows.append((train_end, holdout_start))
    return SplitPlan(train_end=train_end, validation_windows=validation_windows, holdout=(holdout_start, n_rows))
