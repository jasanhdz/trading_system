"""Read-only boundary for neutral market data shared by existing systems."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable

import pandas as pd
import pyarrow.parquet as pq

from aegis_strategy_router.domain.serialization import utc_datetime
from aegis_strategy_router.replay.fresh_pipeline import (
    CandleCoverage,
    FreshPipelineDataError,
    ParquetMinuteCandleSource,
)


SHARED_ADAPTER_SCHEMA = "aegis-strategy-router-shared-neutral-market-data-v1"

# Candle inputs must contain observations, not an upstream interpretation.
FORBIDDEN_DECISION_COLUMNS = frozenset({
    "action",
    "aegis_direction",
    "aegis_signal",
    "classification",
    "committee_output",
    "confidence",
    "decision",
    "entry_quality",
    "final_decision",
    "label",
    "outcome",
    "proposed_side",
    "reason_codes",
    "score",
    "signal_id",
    "side",
    "target",
})
FORBIDDEN_PREFIXES = ("aegis_", "decision_", "future_", "outcome_", "target_")


class DecisionDerivedMarketDataError(FreshPipelineDataError):
    """A purported market source contains an upstream decision field."""


@dataclass(frozen=True, slots=True)
class SharedSourceAudit:
    schema_id: str
    symbol: str
    files: tuple[str, ...]
    columns: tuple[str, ...]
    source_manifest_hash: str
    coverage: CandleCoverage
    read_only: bool = True
    aegis_decisions_loaded: bool = False
    network_access: bool = False
    financial_capability: bool = False


class SharedNeutralMinuteCandleSource:
    """Validate complete source schemas, then delegate deterministic candle merge.

    The class deliberately has no downloader, websocket, writer, signal reader,
    or exchange client. Existing roots are opened only through Parquet reads.
    """

    def __init__(self, roots: Iterable[Path]) -> None:
        self.roots = tuple(Path(root) for root in roots)
        self._delegate = ParquetMinuteCandleSource(self.roots)

    def load(self, symbol: str) -> tuple[pd.DataFrame, CandleCoverage]:
        frame, audit = self.load_with_audit(symbol)
        return frame, audit.coverage

    def load_with_audit(self, symbol: str) -> tuple[pd.DataFrame, SharedSourceAudit]:
        normalized = symbol.upper()
        files = tuple(
            root / f"{normalized}_1m.parquet"
            for root in self.roots
            if (root / f"{normalized}_1m.parquet").is_file()
        )
        if not files:
            raise FreshPipelineDataError(f"NO_SHARED_NEUTRAL_CANDLE_SOURCE:{normalized}")

        all_columns: set[str] = set()
        file_identities = []
        for path in files:
            columns = tuple(pq.read_schema(path).names)
            _assert_market_only_columns(columns)
            all_columns.update(columns)
            stat = path.stat()
            file_identities.append((str(path.resolve()), stat.st_size, stat.st_mtime_ns, columns))

        frame, coverage = self._delegate.load(normalized)
        return frame, SharedSourceAudit(
            schema_id=SHARED_ADAPTER_SCHEMA,
            symbol=normalized,
            files=tuple(str(path.resolve()) for path in files),
            columns=tuple(sorted(all_columns)),
            source_manifest_hash=hashlib.sha256(repr(tuple(file_identities)).encode("utf-8")).hexdigest(),
            coverage=coverage,
        )

    def assert_fresh_for(self, symbol: str, decision_at: datetime) -> SharedSourceAudit:
        _, audit = self.load_with_audit(symbol)
        boundary = utc_datetime(decision_at)
        expected_latest_open_ms = int(boundary.timestamp() * 1_000) // 60_000 * 60_000 - 60_000
        if audit.coverage.last_open_ms < expected_latest_open_ms:
            raise FreshPipelineDataError(
                f"STALE_SHARED_MARKET_DATA:{symbol.upper()}:"
                f"{audit.coverage.last_open_ms}:{expected_latest_open_ms}"
            )
        return audit


def _assert_market_only_columns(columns: Iterable[str]) -> None:
    normalized = {str(column).strip().lower() for column in columns}
    prohibited = sorted(
        column for column in normalized
        if column in FORBIDDEN_DECISION_COLUMNS
        or any(column.startswith(prefix) for prefix in FORBIDDEN_PREFIXES)
    )
    if prohibited:
        raise DecisionDerivedMarketDataError(
            f"UNSAFE_OR_DECISION_DERIVED_COLUMNS:{','.join(prohibited)}"
        )
