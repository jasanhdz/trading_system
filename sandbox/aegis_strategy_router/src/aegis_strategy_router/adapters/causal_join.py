"""Small backward-as-of primitive with explicit availability semantics."""

from __future__ import annotations

from bisect import bisect_right
from datetime import datetime
from typing import Callable, Iterable, TypeVar

from aegis_strategy_router.domain.serialization import utc_datetime


T = TypeVar("T")


def causal_asof(
    query_at: datetime,
    rows: Iterable[T],
    *,
    available_at: Callable[[T], datetime],
    allow_exact_matches: bool = True,
) -> T | None:
    """Return the newest row available no later than the causal boundary."""
    boundary = utc_datetime(query_at)
    ordered = sorted(((utc_datetime(available_at(row)), row) for row in rows), key=lambda pair: pair[0])
    timestamps = [pair[0] for pair in ordered]
    index = bisect_right(timestamps, boundary) if allow_exact_matches else bisect_right(timestamps, boundary, 0, len(timestamps))
    if not allow_exact_matches:
        while index and timestamps[index - 1] == boundary:
            index -= 1
    return ordered[index - 1][1] if index else None
