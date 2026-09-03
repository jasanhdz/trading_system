"""Label-free wiring from fresh candle/signal files to snapshots and candidates."""

from __future__ import annotations

import json
import hashlib
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

from aegis_strategy_router.candidates.contracts import CandidateEvaluation
from aegis_strategy_router.candidates.registry import CandidateGeneratorRegistry, CandidateReplayContext
from aegis_strategy_router.domain.serialization import canonical_json_bytes, utc_datetime
from aegis_strategy_router.domain.types import DataStatus, MarketSnapshot, Side
from aegis_strategy_router.replay.snapshot_builder import DeterministicSnapshotBuilder


CANDLE_COLUMNS = (
    "open_time_ms", "open", "high", "low", "close", "volume", "taker_buy_volume",
)


def causal_candle_source_hash(frame: pd.DataFrame, decision_at: datetime) -> str:
    """Hash only candles that were fully available at the decision boundary."""
    boundary = utc_datetime(decision_at)
    latest_closed_open_ms = int(boundary.timestamp() * 1_000) // 60_000 * 60_000 - 60_000
    causal = frame.loc[frame.open_time_ms.le(latest_closed_open_ms), list(CANDLE_COLUMNS)]
    return hashlib.sha256(
        pd.util.hash_pandas_object(causal, index=False).values.tobytes()
    ).hexdigest()


class FreshPipelineDataError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class FreshSignal:
    signal_id: str
    timestamp: datetime
    symbol: str
    side: Side
    reference_price: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "timestamp", utc_datetime(self.timestamp))


@dataclass(frozen=True, slots=True)
class CandleCoverage:
    symbol: str
    rows: int
    first_open_ms: int
    last_open_ms: int
    duplicate_rows_removed: int
    gaps: int
    source_hash: str


@dataclass(frozen=True, slots=True)
class FreshPipelineResult:
    snapshots: tuple[MarketSnapshot, ...]
    candidates: tuple[CandidateEvaluation, ...]
    candle_coverage: tuple[CandleCoverage, ...]
    rejected_signals: tuple[tuple[str, str], ...]

    def manifest(self) -> dict[str, Any]:
        status_counts = Counter(item.status.value for item in self.candidates)
        strategy_counts = Counter((item.strategy.value, item.status.value) for item in self.candidates)
        detailed_counts = Counter(
            (
                item.strategy.value,
                str(dict(item.metadata).get("symbol", "UNKNOWN")),
                item.side.value,
                item.status.value,
            )
            for item in self.candidates
        )
        snapshot_counts = Counter((item.symbol, item.proposed_side.value) for item in self.snapshots if item.proposed_side)
        eligible_counts = Counter(
            (item.strategy.value, str(dict(item.metadata).get("symbol", "UNKNOWN")), item.side.value)
            for item in self.candidates
            if item.status.value == "ELIGIBLE"
        )
        evaluated_groups = sorted({
            (item.strategy.value, str(dict(item.metadata).get("symbol", "UNKNOWN")), item.side.value)
            for item in self.candidates
        })
        return {
            "schema": "aegis-strategy-router-fresh-pipeline-v1",
            "snapshots": len(self.snapshots),
            "candidate_episodes": len(self.candidates),
            "candidate_status_counts": dict(sorted(status_counts.items())),
            "candidate_counts": {
                f"{strategy}:{status}": count
                for (strategy, status), count in sorted(strategy_counts.items())
            },
            "candidate_counts_by_strategy_symbol_side_status": {
                f"{strategy}:{symbol}:{side}:{status}": count
                for (strategy, symbol, side, status), count in sorted(detailed_counts.items())
            },
            "eligible_event_rate_by_strategy_symbol_side": {
                f"{strategy}:{symbol}:{side}": eligible_counts[(strategy, symbol, side)] / snapshot_counts[(symbol, side)]
                for strategy, symbol, side in evaluated_groups
                if snapshot_counts[(symbol, side)]
            },
            "rejected_signals": [
                {"signal_id": signal_id, "reason": reason}
                for signal_id, reason in self.rejected_signals
            ],
            "candle_coverage": [asdict(coverage) for coverage in self.candle_coverage],
            "outcomes_loaded": False,
            "edge_validation_performed": False,
        }


class ParquetMinuteCandleSource:
    """Merge immutable public candle partitions and reject gaps/conflicts."""

    def __init__(self, roots: Iterable[Path]) -> None:
        self.roots = tuple(Path(root) for root in roots)

    def load(self, symbol: str) -> tuple[pd.DataFrame, CandleCoverage]:
        frames: list[pd.DataFrame] = []
        for root in self.roots:
            path = root / f"{symbol}_1m.parquet"
            if path.exists():
                frames.append(pd.read_parquet(path, columns=list(CANDLE_COLUMNS)))
        if not frames:
            raise FreshPipelineDataError(f"NO_CANDLE_SOURCE:{symbol}")
        normalized = []
        for frame in frames:
            value = frame.copy()
            for column in CANDLE_COLUMNS:
                value[column] = pd.to_numeric(value[column], errors="raise")
            normalized.append(value.sort_values("open_time_ms", kind="mergesort", ignore_index=True))
        combined = normalized[0]
        before = len(combined)
        for newer in normalized[1:]:
            before += len(newer)
            overlap = combined.merge(newer, on="open_time_ms", suffixes=("_old", "_new"))
            conflict_times = set()
            for column in CANDLE_COLUMNS[1:]:
                conflict_times.update(
                    int(value)
                    for value in overlap.loc[
                        overlap[f"{column}_old"].ne(overlap[f"{column}_new"]), "open_time_ms"
                    ]
                )
            if conflict_times and conflict_times != {int(combined.open_time_ms.max())}:
                raise FreshPipelineDataError(f"CONFLICTING_DUPLICATE_CANDLE:{symbol}")
            if conflict_times:
                combined = combined.loc[~combined.open_time_ms.isin(conflict_times)]
            combined = pd.concat([combined, newer], ignore_index=True).drop_duplicates(
                "open_time_ms", keep="last"
            ).sort_values("open_time_ms", kind="mergesort", ignore_index=True)
        deltas = combined["open_time_ms"].diff().dropna()
        gap_count = int(deltas.ne(60_000).sum())
        if gap_count:
            raise FreshPipelineDataError(f"CANDLE_GAPS:{symbol}:{gap_count}")
        coverage = CandleCoverage(
            symbol=symbol,
            rows=len(combined),
            first_open_ms=int(combined.iloc[0].open_time_ms),
            last_open_ms=int(combined.iloc[-1].open_time_ms),
            duplicate_rows_removed=before - len(combined),
            gaps=gap_count,
            source_hash=hashlib.sha256(
                pd.util.hash_pandas_object(
                    combined.loc[:, list(CANDLE_COLUMNS)], index=False
                ).values.tobytes()
            ).hexdigest(),
        )
        return combined, coverage


class FreshSnapshotCandidatePipeline:
    def __init__(self) -> None:
        self.builder = DeterministicSnapshotBuilder()
        self.generators = CandidateGeneratorRegistry()

    def run(
        self, signals: Iterable[FreshSignal], candle_source: ParquetMinuteCandleSource
    ) -> FreshPipelineResult:
        snapshots: list[MarketSnapshot] = []
        candidates: list[CandidateEvaluation] = []
        rejected: list[tuple[str, str]] = []
        coverage_by_symbol: dict[str, CandleCoverage] = {}
        candles_by_symbol: dict[str, pd.DataFrame] = {}
        replay_context = CandidateReplayContext()
        for signal in sorted(signals, key=lambda item: (item.timestamp, item.symbol, item.signal_id)):
            try:
                if signal.symbol not in candles_by_symbol:
                    candles, coverage = candle_source.load(signal.symbol)
                    candles_by_symbol[signal.symbol] = candles
                    coverage_by_symbol[signal.symbol] = coverage
                decision_ms = int(signal.timestamp.timestamp() * 1_000)
                expected_latest_closed_open_ms = decision_ms // 60_000 * 60_000 - 60_000
                if coverage_by_symbol[signal.symbol].last_open_ms < expected_latest_closed_open_ms:
                    raise FreshPipelineDataError(
                        f"STALE_CANDLE_SOURCE:{signal.symbol}:"
                        f"{coverage_by_symbol[signal.symbol].last_open_ms}:"
                        f"{expected_latest_closed_open_ms}"
                    )
                snapshot = self.builder.build(
                    symbol=signal.symbol,
                    decision_at=signal.timestamp,
                    built_at=signal.timestamp,
                    reference_price=signal.reference_price,
                    one_minute=candles_by_symbol[signal.symbol],
                    proposed_side=signal.side,
                    signal_id=signal.signal_id,
                    source_versions={
                        "fresh_pipeline": "fresh-snapshot-candidate-pipeline-v1",
                        "fresh_candle_source_hash": causal_candle_source_hash(
                            candles_by_symbol[signal.symbol], signal.timestamp
                        ),
                    },
                )
                incomplete = [
                    state.timeframe.value
                    for state in snapshot.timeframes
                    if state.status is not DataStatus.AVAILABLE
                    or (state.structural is not None and state.structural.status is not DataStatus.AVAILABLE)
                ]
                if incomplete:
                    rejected.append((signal.signal_id, f"INCOMPLETE_SNAPSHOT:{','.join(incomplete)}"))
                    continue
                snapshots.append(snapshot)
                candidates.extend(
                    self.generators.generate_all_replay(snapshot, signal.side, replay_context)
                )
            except (FreshPipelineDataError, ValueError) as error:
                rejected.append((signal.signal_id, str(error)))
        return FreshPipelineResult(
            snapshots=tuple(snapshots),
            candidates=tuple(candidates),
            candle_coverage=tuple(coverage_by_symbol[symbol] for symbol in sorted(coverage_by_symbol)),
            rejected_signals=tuple(rejected),
        )


def persist_pipeline_result(result: FreshPipelineResult, output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    payloads = {
        "snapshots.jsonl": [snapshot.to_primitive() for snapshot in result.snapshots],
        "candidates.jsonl": [candidate.to_primitive() for candidate in result.candidates],
    }
    for name, rows in payloads.items():
        temporary = output / f".{name}.tmp"
        with temporary.open("wb") as handle:
            for row in rows:
                handle.write(canonical_json_bytes(row) + b"\n")
        temporary.replace(output / name)
    temporary_manifest = output / ".manifest.json.tmp"
    temporary_manifest.write_text(
        json.dumps(result.manifest(), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary_manifest.replace(output / "manifest.json")
