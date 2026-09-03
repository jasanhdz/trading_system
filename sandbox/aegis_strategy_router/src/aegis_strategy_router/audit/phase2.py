"""Outcome-free Phase 2 snapshot, candidate, and stream coverage audit."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Iterable, Mapping

from aegis_strategy_router.audit.coverage import partition_at
from aegis_strategy_router.candidates.contracts import CandidateEvaluation
from aegis_strategy_router.domain.serialization import utc_datetime


FORBIDDEN_AUDIT_TOKENS = (
    "pnl", "profit", "win", "loss", "mfe", "mae", "future", "outcome",
    "return", "barrier", "edge", "target", "label",
)


class OutcomeFieldProhibited(ValueError):
    pass


def assert_label_free_fields(fields: Iterable[str]) -> None:
    prohibited = sorted(
        field for field in fields
        if any(token in field.lower() for token in FORBIDDEN_AUDIT_TOKENS)
    )
    if prohibited:
        raise OutcomeFieldProhibited(f"Phase 2 audit fields are not label-free: {prohibited}")


@dataclass(frozen=True, slots=True)
class FreshStreamRecord:
    timestamp: datetime
    symbol: str
    stream: str
    valid: bool
    gap_ms: float | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "timestamp", utc_datetime(self.timestamp))
        if self.gap_ms is not None and self.gap_ms < 0:
            raise ValueError("gap_ms cannot be negative")


@dataclass(frozen=True, slots=True)
class Phase2CoverageReport:
    stream_rows: int
    valid_stream_rows: int
    snapshot_count: int
    candidate_episode_count: int
    first_at: datetime | None
    last_at: datetime | None
    counts_by_partition: tuple[tuple[str, int], ...]
    counts_by_stream: tuple[tuple[str, int], ...]
    counts_by_symbol: tuple[tuple[str, int], ...]
    candidate_counts: tuple[tuple[str, str, str, str, int], ...]
    max_gap_ms_by_symbol: tuple[tuple[str, float], ...]

    def to_primitive(self) -> dict[str, Any]:
        return {
            "stream_rows": self.stream_rows,
            "valid_stream_rows": self.valid_stream_rows,
            "snapshot_count": self.snapshot_count,
            "candidate_episode_count": self.candidate_episode_count,
            "first_at": self.first_at.isoformat() if self.first_at else None,
            "last_at": self.last_at.isoformat() if self.last_at else None,
            "counts_by_partition": dict(self.counts_by_partition),
            "counts_by_stream": dict(self.counts_by_stream),
            "counts_by_symbol": dict(self.counts_by_symbol),
            "candidate_counts": [
                {"strategy": strategy, "symbol": symbol, "side": side, "status": status, "count": count}
                for strategy, symbol, side, status, count in self.candidate_counts
            ],
            "max_gap_ms_by_symbol": dict(self.max_gap_ms_by_symbol),
        }


def audit_phase2_coverage(
    streams: Iterable[FreshStreamRecord],
    candidates: Iterable[CandidateEvaluation] = (),
) -> Phase2CoverageReport:
    stream_values = tuple(streams)
    candidate_values = tuple(candidates)
    timestamps = tuple(sorted(record.timestamp for record in stream_values))
    partitions = Counter(partition_at(record.timestamp) for record in stream_values)
    stream_counts = Counter(record.stream for record in stream_values)
    symbol_counts = Counter(record.symbol for record in stream_values)
    candidate_counts = Counter(
        (item.strategy.value, str(dict(item.metadata).get("symbol", "UNKNOWN")), item.side.value, item.status.value)
        for item in candidate_values
    )
    gaps: dict[str, list[float]] = defaultdict(list)
    for record in stream_values:
        if record.gap_ms is not None:
            gaps[record.symbol].append(record.gap_ms)
    return Phase2CoverageReport(
        stream_rows=len(stream_values),
        valid_stream_rows=sum(record.valid for record in stream_values),
        snapshot_count=len({item.snapshot_id for item in candidate_values}),
        candidate_episode_count=len({item.candidate_episode_id for item in candidate_values}),
        first_at=timestamps[0] if timestamps else None,
        last_at=timestamps[-1] if timestamps else None,
        counts_by_partition=tuple(sorted(partitions.items())),
        counts_by_stream=tuple(sorted(stream_counts.items())),
        counts_by_symbol=tuple(sorted(symbol_counts.items())),
        candidate_counts=tuple((*key, count) for key, count in sorted(candidate_counts.items())),
        max_gap_ms_by_symbol=tuple(sorted((symbol, max(values)) for symbol, values in gaps.items())),
    )
