"""Frozen split boundaries and label-free sample-count auditing."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Iterable

from aegis_strategy_router.domain.serialization import utc_datetime


@dataclass(frozen=True, slots=True)
class Partition:
    name: str
    start: datetime
    end_exclusive: datetime

    def __post_init__(self) -> None:
        object.__setattr__(self, "start", utc_datetime(self.start))
        object.__setattr__(self, "end_exclusive", utc_datetime(self.end_exclusive))
        if self.start >= self.end_exclusive:
            raise ValueError("partition start must precede end")

    def contains(self, timestamp: datetime) -> bool:
        value = utc_datetime(timestamp)
        return self.start <= value < self.end_exclusive


UTC = timezone.utc
FRESH_TRAIN_START = datetime(2026, 8, 17, 21, 14, 26, 93_000, tzinfo=UTC)
FROZEN_PARTITIONS = (
    Partition("FRESH_TRAIN", FRESH_TRAIN_START, datetime(2026, 11, 1, tzinfo=UTC)),
    Partition("FRESH_CALIBRATION", datetime(2026, 11, 1, tzinfo=UTC), datetime(2026, 11, 16, tzinfo=UTC)),
    Partition("SPECIALIST_VALIDATION", datetime(2026, 11, 16, tzinfo=UTC), datetime(2026, 12, 16, tzinfo=UTC)),
    Partition("ROUTER_VALIDATION", datetime(2026, 12, 16, tzinfo=UTC), datetime(2027, 1, 16, tzinfo=UTC)),
    Partition("FINAL_SYSTEM_HOLDOUT", datetime(2027, 1, 16, tzinfo=UTC), datetime(2027, 3, 1, tzinfo=UTC)),
)


def partition_at(timestamp: datetime) -> str:
    value = utc_datetime(timestamp)
    if value < FROZEN_PARTITIONS[0].start:
        return "DISCOVERY_QUARANTINE"
    for partition in FROZEN_PARTITIONS:
        if partition.contains(value):
            return partition.name
    return "OUTSIDE_FROZEN_TIMELINE"


@dataclass(frozen=True, slots=True)
class CoverageAudit:
    total_rows: int
    first_at: datetime | None
    last_at: datetime | None
    counts_by_partition: tuple[tuple[str, int], ...]


def audit_timestamps(timestamps: Iterable[datetime]) -> CoverageAudit:
    values = tuple(sorted(utc_datetime(value) for value in timestamps))
    counts = Counter(partition_at(value) for value in values)
    return CoverageAudit(
        total_rows=len(values),
        first_at=values[0] if values else None,
        last_at=values[-1] if values else None,
        counts_by_partition=tuple(sorted(counts.items())),
    )


@dataclass(frozen=True, slots=True)
class EpisodeAudit:
    rows: int
    independent_episodes: int
    duplicate_rows: int


def audit_episode_ids(episode_ids: Iterable[str]) -> EpisodeAudit:
    values = tuple(str(value) for value in episode_ids)
    independent = len(set(values))
    return EpisodeAudit(len(values), independent, len(values) - independent)
