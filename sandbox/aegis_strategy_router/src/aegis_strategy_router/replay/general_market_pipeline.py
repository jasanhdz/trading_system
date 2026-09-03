"""Independent, label-free Phase 2 replay over the general market timeline."""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

from aegis_strategy_router.candidates.contracts import (
    CandidateEvaluation,
    CandidateStatus,
    Strategy,
    SubstateDisposition,
)
from aegis_strategy_router.candidates.registry import CandidateGeneratorRegistry, CandidateReplayContext
from aegis_strategy_router.domain.serialization import canonical_json_bytes, content_hash, utc_datetime
from aegis_strategy_router.domain.types import DataStatus, MarketSnapshot, Side
from aegis_strategy_router.replay.fresh_pipeline import (
    CandleCoverage,
    ParquetMinuteCandleSource,
    causal_candle_source_hash,
)
from aegis_strategy_router.replay.snapshot_builder import DeterministicSnapshotBuilder


GENERAL_MARKET_SCHEMA = "aegis-strategy-router-general-market-v1"
ANCHOR_MINUTES = 15
OVERLAP_HORIZON = timedelta(minutes=60)
TRAIN_MINIMUM_PER_SPECIALIST = 2_000
MINIMUM_SYMBOLS = 6
MINIMUM_WEEKLY_BLOCKS = 4


@dataclass(frozen=True, slots=True)
class IndependentCandidateEpisode:
    independent_episode_id: str
    source_candidate_episode_id: str
    logical_setup_id: str
    snapshot_id: str
    strategy: Strategy
    symbol: str
    side: Side
    decision_at: datetime
    overlap_until: datetime

    def __post_init__(self) -> None:
        object.__setattr__(self, "decision_at", utc_datetime(self.decision_at))
        object.__setattr__(self, "overlap_until", utc_datetime(self.overlap_until))
        if self.overlap_until <= self.decision_at:
            raise ValueError("independent episode overlap window must be positive")
        expected = content_hash(self.identity_payload())
        if self.independent_episode_id != expected:
            raise ValueError("independent_episode_id does not match canonical identity")

    @classmethod
    def from_candidate(cls, candidate: CandidateEvaluation) -> "IndependentCandidateEpisode":
        metadata = dict(candidate.metadata)
        symbol = str(metadata["symbol"])
        logical_setup_id = str(metadata.get("setup_episode_id", candidate.candidate_episode_id))
        identity = {
            "source_candidate_episode_id": candidate.candidate_episode_id,
            "logical_setup_id": logical_setup_id,
            "snapshot_id": candidate.snapshot_id,
            "strategy": candidate.strategy,
            "symbol": symbol,
            "side": candidate.side,
            "decision_at": candidate.decision_at,
        }
        return cls(
            independent_episode_id=content_hash(identity),
            source_candidate_episode_id=candidate.candidate_episode_id,
            logical_setup_id=logical_setup_id,
            snapshot_id=candidate.snapshot_id,
            strategy=candidate.strategy,
            symbol=symbol,
            side=candidate.side,
            decision_at=candidate.decision_at,
            overlap_until=candidate.decision_at + OVERLAP_HORIZON,
        )

    def identity_payload(self) -> dict[str, Any]:
        return {
            "source_candidate_episode_id": self.source_candidate_episode_id,
            "logical_setup_id": self.logical_setup_id,
            "snapshot_id": self.snapshot_id,
            "strategy": self.strategy,
            "symbol": self.symbol,
            "side": self.side,
            "decision_at": self.decision_at,
        }

    def to_primitive(self) -> dict[str, Any]:
        return {
            "independent_episode_id": self.independent_episode_id,
            **self.identity_payload(),
            "overlap_until": self.overlap_until,
        }


def is_candidate_population_event(candidate: CandidateEvaluation) -> bool:
    return (
        candidate.status is CandidateStatus.ELIGIBLE
        and candidate.disposition in {
            SubstateDisposition.CANDIDATE,
            SubstateDisposition.ENTERABLE,
        }
    )


def select_independent_episodes(
    candidates: Iterable[CandidateEvaluation],
) -> tuple[tuple[IndependentCandidateEpisode, ...], tuple[tuple[str, str], ...]]:
    grouped: dict[tuple[Strategy, str, Side], list[CandidateEvaluation]] = defaultdict(list)
    for candidate in candidates:
        if is_candidate_population_event(candidate):
            symbol = str(dict(candidate.metadata)["symbol"])
            grouped[(candidate.strategy, symbol, candidate.side)].append(candidate)

    selected: list[IndependentCandidateEpisode] = []
    suppressed: list[tuple[str, str]] = []
    for key in sorted(grouped, key=lambda item: (item[0].value, item[1], item[2].value)):
        seen_setups: set[str] = set()
        last_selected_at: datetime | None = None
        values = sorted(grouped[key], key=lambda item: (item.decision_at, item.candidate_episode_id))
        for candidate in values:
            metadata = dict(candidate.metadata)
            setup_id = str(metadata.get("setup_episode_id", candidate.candidate_episode_id))
            if setup_id in seen_setups:
                suppressed.append((candidate.candidate_episode_id, "DUPLICATE_SETUP_EPISODE"))
                continue
            seen_setups.add(setup_id)
            if last_selected_at is not None and candidate.decision_at < last_selected_at + OVERLAP_HORIZON:
                suppressed.append((candidate.candidate_episode_id, "TEMPORAL_OVERLAP_60M"))
                continue
            episode = IndependentCandidateEpisode.from_candidate(candidate)
            selected.append(episode)
            last_selected_at = candidate.decision_at
    return (
        tuple(sorted(selected, key=lambda item: (
            item.decision_at, item.symbol, item.strategy.value, item.side.value,
            item.independent_episode_id,
        ))),
        tuple(sorted(suppressed)),
    )


@dataclass(frozen=True, slots=True)
class GeneralMarketPipelineResult:
    snapshots: tuple[MarketSnapshot, ...]
    candidates: tuple[CandidateEvaluation, ...]
    independent_episodes: tuple[IndependentCandidateEpisode, ...]
    suppressed_candidates: tuple[tuple[str, str], ...]
    candle_coverage: tuple[CandleCoverage, ...]
    rejected_anchors: tuple[tuple[str, str], ...]

    def manifest(self) -> dict[str, Any]:
        snapshot_counts = Counter(snapshot.symbol for snapshot in self.snapshots)
        evaluation_counts = Counter(
            (
                candidate.strategy.value,
                str(dict(candidate.metadata).get("symbol", "UNKNOWN")),
                candidate.side.value,
                candidate.status.value,
            )
            for candidate in self.candidates
        )
        population_counts = Counter(
            (
                candidate.strategy.value,
                str(dict(candidate.metadata).get("symbol", "UNKNOWN")),
                candidate.side.value,
            )
            for candidate in self.candidates
            if is_candidate_population_event(candidate)
        )
        independent_counts = Counter(
            (episode.strategy.value, episode.symbol, episode.side.value)
            for episode in self.independent_episodes
        )
        strategy_independent = Counter(episode.strategy.value for episode in self.independent_episodes)
        strategy_symbols: dict[str, set[str]] = defaultdict(set)
        strategy_weeks: dict[str, set[str]] = defaultdict(set)
        for episode in self.independent_episodes:
            strategy_symbols[episode.strategy.value].add(episode.symbol)
            iso = episode.decision_at.isocalendar()
            strategy_weeks[episode.strategy.value].add(f"{iso.year}-W{iso.week:02d}")
        denominator = Counter(
            (
                candidate.strategy.value,
                str(dict(candidate.metadata).get("symbol", "UNKNOWN")),
                candidate.side.value,
            )
            for candidate in self.candidates
        )
        groups = sorted(denominator)
        candidate_snapshot_ids = {
            candidate.snapshot_id
            for candidate in self.candidates
            if is_candidate_population_event(candidate)
        }
        support = {
            strategy.value: {
                "independent_train_candidates": strategy_independent[strategy.value],
                "required_train_candidates": TRAIN_MINIMUM_PER_SPECIALIST,
                "symbols": len(strategy_symbols[strategy.value]),
                "required_symbols": MINIMUM_SYMBOLS,
                "weekly_blocks": len(strategy_weeks[strategy.value]),
                "required_weekly_blocks": MINIMUM_WEEKLY_BLOCKS,
                "train_support_met": (
                    strategy_independent[strategy.value] >= TRAIN_MINIMUM_PER_SPECIALIST
                    and len(strategy_symbols[strategy.value]) >= MINIMUM_SYMBOLS
                    and len(strategy_weeks[strategy.value]) >= MINIMUM_WEEKLY_BLOCKS
                ),
            }
            for strategy in Strategy
        }
        return {
            "schema": GENERAL_MARKET_SCHEMA,
            "initial_experiment_mode": "INDEPENDENT_STRATEGY_DISCOVERY",
            "aegis_signals_loaded": False,
            "anchor_cadence_minutes": ANCHOR_MINUTES,
            "overlap_horizon_minutes": int(OVERLAP_HORIZON.total_seconds() / 60),
            "snapshots": len(self.snapshots),
            "snapshots_by_symbol": dict(sorted(snapshot_counts.items())),
            "none_snapshots": len(self.snapshots) - len(candidate_snapshot_ids),
            "candidate_evaluations": len(self.candidates),
            "candidate_population_events": sum(population_counts.values()),
            "independent_candidate_episodes": len(self.independent_episodes),
            "suppressed_overlap_or_setup_duplicates": len(self.suppressed_candidates),
            "candidate_counts_by_strategy_symbol_side_status": {
                f"{strategy}:{symbol}:{side}:{status}": count
                for (strategy, symbol, side, status), count in sorted(evaluation_counts.items())
            },
            "candidate_event_rate_by_strategy_symbol_side": {
                f"{strategy}:{symbol}:{side}": population_counts[(strategy, symbol, side)] / denominator[(strategy, symbol, side)]
                for strategy, symbol, side in groups
            },
            "independent_counts_by_strategy_symbol_side": {
                f"{strategy}:{symbol}:{side}": count
                for (strategy, symbol, side), count in sorted(independent_counts.items())
            },
            "fresh_support": support,
            "fresh_data_sufficiency": (
                "MET" if any(value["train_support_met"] for value in support.values())
                else "NOT_YET_MET"
            ),
            "rejected_anchors": [
                {"anchor_id": anchor_id, "reason": reason}
                for anchor_id, reason in self.rejected_anchors
            ],
            "candle_coverage": [asdict(item) for item in self.candle_coverage],
            "outcomes_loaded": False,
            "edge_validation_performed": False,
        }


class GeneralMarketCandidatePipeline:
    def __init__(self, builder: object | None = None) -> None:
        self.builder = builder or DeterministicSnapshotBuilder()
        self.generators = CandidateGeneratorRegistry()

    def run(
        self,
        *,
        symbols: Iterable[str],
        start_at: datetime,
        end_at: datetime,
        candle_source: ParquetMinuteCandleSource,
    ) -> GeneralMarketPipelineResult:
        start = utc_datetime(start_at)
        end = utc_datetime(end_at)
        if end <= start:
            raise ValueError("general-market end_at must follow start_at")
        loaded: dict[str, pd.DataFrame] = {}
        coverages: dict[str, CandleCoverage] = {}
        anchors: list[tuple[datetime, str]] = []
        rejected: list[tuple[str, str]] = []
        for symbol in sorted(set(symbols)):
            try:
                frame, coverage = candle_source.load(symbol)
                loaded[symbol] = frame
                coverages[symbol] = coverage
                first_close = pd.to_datetime(frame.iloc[0].open_time_ms, unit="ms", utc=True) + pd.Timedelta(minutes=1)
                last_close = pd.to_datetime(frame.iloc[-1].open_time_ms, unit="ms", utc=True) + pd.Timedelta(minutes=1)
                lower = max(pd.Timestamp(start), first_close).ceil(f"{ANCHOR_MINUTES}min")
                upper = min(pd.Timestamp(end), last_close).floor(f"{ANCHOR_MINUTES}min")
                if upper < lower:
                    rejected.append((symbol, "NO_FULLY_CLOSED_GENERAL_MARKET_ANCHOR"))
                    continue
                anchors.extend((value.to_pydatetime(), symbol) for value in pd.date_range(
                    lower, upper, freq=f"{ANCHOR_MINUTES}min", tz="UTC"
                ))
            except (ValueError, OSError) as error:
                rejected.append((symbol, str(error)))

        snapshots: list[MarketSnapshot] = []
        candidates: list[CandidateEvaluation] = []
        replay_context = CandidateReplayContext()
        for decision_at, symbol in sorted(anchors, key=lambda item: (item[0], item[1])):
            anchor_id = f"{symbol}:{decision_at.isoformat()}"
            try:
                frame = loaded[symbol]
                latest_open_ms = int(decision_at.timestamp() * 1_000) - 60_000
                reference_rows = frame.loc[frame.open_time_ms.eq(latest_open_ms), "close"]
                if len(reference_rows) != 1:
                    raise ValueError("GENERAL_MARKET_REFERENCE_CANDLE_MISSING")
                source_hash_provider = getattr(self.builder, "causal_source_hash", None)
                source_hash = (
                    source_hash_provider(symbol, frame, decision_at)
                    if source_hash_provider is not None
                    else causal_candle_source_hash(frame, decision_at)
                )
                snapshot = self.builder.build(
                    symbol=symbol,
                    decision_at=decision_at,
                    built_at=decision_at,
                    reference_price=float(reference_rows.iloc[0]),
                    one_minute=frame,
                    proposed_side=None,
                    signal_id=None,
                    source_versions={
                        "general_market_pipeline": GENERAL_MARKET_SCHEMA,
                        "fresh_candle_source_hash": source_hash,
                    },
                )
                incomplete = [
                    state.timeframe.value
                    for state in snapshot.timeframes
                    if state.status is not DataStatus.AVAILABLE
                    or (state.structural is not None and state.structural.status is not DataStatus.AVAILABLE)
                ]
                if incomplete:
                    raise ValueError(f"INCOMPLETE_SNAPSHOT:{','.join(incomplete)}")
                snapshots.append(snapshot)
                for side in (Side.LONG, Side.SHORT):
                    candidates.extend(
                        self.generators.generate_all_replay(snapshot, side, replay_context)
                    )
            except ValueError as error:
                rejected.append((anchor_id, str(error)))

        independent, suppressed = select_independent_episodes(candidates)
        return GeneralMarketPipelineResult(
            snapshots=tuple(snapshots),
            candidates=tuple(candidates),
            independent_episodes=independent,
            suppressed_candidates=suppressed,
            candle_coverage=tuple(coverages[symbol] for symbol in sorted(coverages)),
            rejected_anchors=tuple(sorted(rejected)),
        )


def merge_general_market_results(
    results: Iterable[GeneralMarketPipelineResult],
) -> GeneralMarketPipelineResult:
    """Merge symbol partitions while preserving the serial replay ordering."""
    values = tuple(results)
    side_order = {side: index for index, side in enumerate(Side)}
    strategy_order = {strategy: index for index, strategy in enumerate(Strategy)}
    snapshots = tuple(sorted(
        (snapshot for result in values for snapshot in result.snapshots),
        key=lambda item: (item.decision_at, item.symbol, item.snapshot_id),
    ))
    candidates = tuple(sorted(
        (candidate for result in values for candidate in result.candidates),
        key=lambda item: (
            item.decision_at,
            str(dict(item.metadata).get("symbol", "UNKNOWN")),
            side_order[item.side],
            strategy_order[item.strategy],
            item.candidate_episode_id,
        ),
    ))
    independent, suppressed = select_independent_episodes(candidates)
    coverage_by_symbol = {
        coverage.symbol: coverage
        for result in values
        for coverage in result.candle_coverage
    }
    return GeneralMarketPipelineResult(
        snapshots=snapshots,
        candidates=candidates,
        independent_episodes=independent,
        suppressed_candidates=suppressed,
        candle_coverage=tuple(coverage_by_symbol[symbol] for symbol in sorted(coverage_by_symbol)),
        rejected_anchors=tuple(sorted(
            rejected for result in values for rejected in result.rejected_anchors
        )),
    )


def persist_general_market_result(result: GeneralMarketPipelineResult, output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    payloads = {
        "snapshots.jsonl": [snapshot.to_primitive() for snapshot in result.snapshots],
        "candidate_evaluations.jsonl": [candidate.to_primitive() for candidate in result.candidates],
        "independent_candidate_episodes.jsonl": [episode.to_primitive() for episode in result.independent_episodes],
        "suppressed_candidates.jsonl": [
            {"candidate_episode_id": candidate_id, "reason": reason}
            for candidate_id, reason in result.suppressed_candidates
        ],
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
