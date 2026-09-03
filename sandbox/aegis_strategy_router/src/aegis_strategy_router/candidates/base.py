"""Fail-closed base implementation for deterministic candidate generators."""

from __future__ import annotations

from abc import ABC, abstractmethod

from aegis_strategy_router.candidates.contracts import (
    CandidateEvaluation,
    CandidateStatus,
    CandidateSubstate,
    CandidateSetup,
    FrozenDecisionGap,
    RuleObservation,
    Strategy,
)
from aegis_strategy_router.domain.types import DataStatus, MarketSnapshot, Side


GENERATOR_VERSION = "aegis-strategy-router-candidates-v2-rule-freeze"


class CandidateGenerator(ABC):
    strategy: Strategy

    @property
    @abstractmethod
    def frozen_gaps(self) -> tuple[FrozenDecisionGap, ...]:
        """Return unresolved methodological decisions; never infer them from data."""

    def frozen_observations(
        self, snapshot: MarketSnapshot, side: Side
    ) -> tuple[RuleObservation, ...]:
        return ()

    def generate(
        self,
        snapshot: MarketSnapshot,
        side: Side,
        prior_setup: CandidateSetup | None = None,
    ) -> CandidateEvaluation:
        for timeframe in snapshot.timeframes:
            timeframe.assert_causal(snapshot.decision_at)
        unavailable = tuple(
            f"{state.timeframe.value}:{state.status.value}"
            for state in snapshot.timeframes
            if state.status is not DataStatus.AVAILABLE
        )
        if unavailable:
            return CandidateEvaluation.create(
                snapshot_id=snapshot.snapshot_id,
                signal_episode_id=snapshot.signal_id,
                strategy=self.strategy,
                side=side,
                decision_at=snapshot.decision_at,
                status=CandidateStatus.UNKNOWN,
                substate=None,
                reason_codes=("SNAPSHOT_DATA_UNAVAILABLE",),
                rules=(),
                frozen_gaps=(),
                generator_version=GENERATOR_VERSION,
                metadata={"unavailable_timeframes": unavailable},
            )
        gaps = self.frozen_gaps
        observations = self.frozen_observations(snapshot, side)
        if gaps:
            return CandidateEvaluation.create(
                snapshot_id=snapshot.snapshot_id,
                signal_episode_id=snapshot.signal_id,
                strategy=self.strategy,
                side=side,
                decision_at=snapshot.decision_at,
                status=CandidateStatus.BLOCKED_FROZEN_DECISION_GAP,
                substate=None,
                reason_codes=tuple(gap.code for gap in gaps),
                rules=observations,
                frozen_gaps=gaps,
                generator_version=GENERATOR_VERSION,
                metadata={"symbol": snapshot.symbol},
            )
        return self.evaluate(snapshot, side, prior_setup)

    @abstractmethod
    def evaluate(
        self,
        snapshot: MarketSnapshot,
        side: Side,
        prior_setup: CandidateSetup | None,
    ) -> CandidateEvaluation:
        """Evaluate only rules recorded in the Phase 2 rule freeze."""

    def evaluation(
        self,
        snapshot: MarketSnapshot,
        side: Side,
        *,
        status: CandidateStatus,
        substate: CandidateSubstate | None,
        reasons: tuple[str, ...],
        rules: tuple[RuleObservation, ...],
        metadata: dict[str, object] | tuple[tuple[str, object], ...] = (),
    ) -> CandidateEvaluation:
        common = {"symbol": snapshot.symbol}
        common.update(dict(metadata))
        return CandidateEvaluation.create(
            snapshot_id=snapshot.snapshot_id,
            signal_episode_id=snapshot.signal_id,
            strategy=self.strategy,
            side=side,
            decision_at=snapshot.decision_at,
            status=status,
            substate=substate,
            reason_codes=reasons,
            rules=rules,
            frozen_gaps=(),
            generator_version=GENERATOR_VERSION,
            metadata=common,
        )

    def require_frozen_rules(self) -> None:
        if self.frozen_gaps:
            raise self.frozen_gaps[0]
