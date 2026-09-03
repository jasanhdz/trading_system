"""Fixed Phase 2 generator registry and episode-overlap controls."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

from aegis_strategy_router.candidates.contracts import CandidateEvaluation, CandidateSetup, Strategy
from aegis_strategy_router.candidates.generators import (
    BreakoutRetestGenerator,
    PullbackContinuationGenerator,
    RangeMeanReversionGenerator,
    RegimeTransitionGenerator,
    TrendContinuationGenerator,
)
from aegis_strategy_router.domain.types import MarketSnapshot, Side


class DuplicateCandidateEpisode(ValueError):
    pass


@dataclass(slots=True)
class CandidateReplayContext:
    """Explicit outcome-free setup state for deterministic chronological replay."""

    _setups: dict[tuple[Strategy, str, Side], CandidateSetup] = field(default_factory=dict)
    _last_boundary: dict[tuple[Strategy, str, Side], tuple[datetime, str]] = field(default_factory=dict)

    def setup(self, strategy: Strategy, symbol: str, side: Side) -> CandidateSetup | None:
        return self._setups.get((strategy, symbol, side))

    def apply(self, evaluation: CandidateEvaluation) -> None:
        metadata = dict(evaluation.metadata)
        symbol = str(metadata["symbol"])
        key = (evaluation.strategy, symbol, evaluation.side)
        boundary = (evaluation.decision_at, evaluation.snapshot_id)
        previous = self._last_boundary.get(key)
        if previous is not None and boundary < previous:
            raise ValueError("candidate replay snapshots must be chronological")
        self._last_boundary[key] = boundary
        if metadata.get("setup_clear"):
            self._setups.pop(key, None)
        if metadata.get("setup_active") and not metadata.get("setup_clear"):
            setup_id = str(metadata["setup_episode_id"])
            started_at = metadata["setup_started_at"]
            self._setups[key] = CandidateSetup(
                setup_episode_id=setup_id,
                strategy=evaluation.strategy,
                symbol=symbol,
                side=evaluation.side,
                started_at=started_at,
                metadata=tuple(
                    (name, value) for name, value in evaluation.metadata
                    if name not in {"symbol", "setup_clear"}
                ),
            )


@dataclass(slots=True)
class CandidateEpisodeIndex:
    _by_id: dict[str, CandidateEvaluation] = field(default_factory=dict)
    _by_overlap: dict[str, list[str]] = field(default_factory=dict)

    def add(self, evaluation: CandidateEvaluation) -> None:
        if evaluation.candidate_episode_id in self._by_id:
            raise DuplicateCandidateEpisode(evaluation.candidate_episode_id)
        self._by_id[evaluation.candidate_episode_id] = evaluation
        self._by_overlap.setdefault(evaluation.overlap_group_id, []).append(evaluation.candidate_episode_id)

    def overlap_ids(self, overlap_group_id: str) -> tuple[str, ...]:
        return tuple(sorted(self._by_overlap.get(overlap_group_id, ())))


class CandidateGeneratorRegistry:
    def __init__(self) -> None:
        generators = (
            TrendContinuationGenerator(),
            PullbackContinuationGenerator(),
            BreakoutRetestGenerator(),
            RangeMeanReversionGenerator(),
            RegimeTransitionGenerator(),
        )
        self.generators = {generator.strategy: generator for generator in generators}

    def generate_all(self, snapshot: MarketSnapshot, side: Side) -> tuple[CandidateEvaluation, ...]:
        return tuple(
            self.generators[strategy].generate(snapshot, side)
            for strategy in Strategy
        )

    def generate_all_replay(
        self,
        snapshot: MarketSnapshot,
        side: Side,
        context: CandidateReplayContext,
    ) -> tuple[CandidateEvaluation, ...]:
        evaluations = []
        for strategy in Strategy:
            generator = self.generators[strategy]
            evaluation = generator.generate(
                snapshot, side, context.setup(strategy, snapshot.symbol, side)
            )
            evaluations.append(evaluation)
            context.apply(evaluation)
        return tuple(evaluations)
