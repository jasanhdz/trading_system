"""Immutable Phase 2 candidate identities, substates, and fail-closed gaps."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any, Iterable, Mapping

from aegis_strategy_router.domain.serialization import content_hash, frozen_pairs, utc_datetime
from aegis_strategy_router.domain.types import Side


class Strategy(str, Enum):
    TREND_CONTINUATION = "TREND_CONTINUATION"
    PULLBACK_CONTINUATION = "PULLBACK_CONTINUATION"
    BREAKOUT_RETEST = "BREAKOUT_RETEST"
    RANGE_MEAN_REVERSION = "RANGE_MEAN_REVERSION"
    REGIME_TRANSITION_REVERSAL = "REGIME_TRANSITION_REVERSAL"


class CandidateStatus(str, Enum):
    ELIGIBLE = "ELIGIBLE"
    INELIGIBLE = "INELIGIBLE"
    UNKNOWN = "UNKNOWN"
    BLOCKED_FROZEN_DECISION_GAP = "BLOCKED_FROZEN_DECISION_GAP"


class SubstateDisposition(str, Enum):
    CANDIDATE = "CANDIDATE"
    ENTERABLE = "ENTERABLE"
    TERMINAL_WAIT = "TERMINAL_WAIT"
    INVALIDATED = "INVALIDATED"


class CandidateSubstate(str, Enum):
    TREND_CANDIDATE = "TREND_CANDIDATE"
    TREND_CONFIRMED = "TREND_CONFIRMED"
    TREND_INVALIDATED = "TREND_INVALIDATED"
    PULLBACK_FORMING = "PULLBACK_FORMING"
    PULLBACK_CONFIRMED = "PULLBACK_CONFIRMED"
    PULLBACK_INVALIDATED = "PULLBACK_INVALIDATED"
    PULLBACK_TOO_LATE = "PULLBACK_TOO_LATE"
    BREAKOUT_CANDIDATE = "BREAKOUT_CANDIDATE"
    BREAKOUT_CONFIRMED = "BREAKOUT_CONFIRMED"
    RETEST_PENDING = "RETEST_PENDING"
    RETEST_CONFIRMED = "RETEST_CONFIRMED"
    FALSE_BREAKOUT = "FALSE_BREAKOUT"
    BREAKOUT_TOO_LATE = "BREAKOUT_TOO_LATE"
    RANGE_EDGE_CANDIDATE = "RANGE_EDGE_CANDIDATE"
    RANGE_REJECTION_CONFIRMED = "RANGE_REJECTION_CONFIRMED"
    RANGE_BROKEN = "RANGE_BROKEN"
    OLD_REGIME_DETERIORATING = "OLD_REGIME_DETERIORATING"
    TRANSITION_CANDIDATE = "TRANSITION_CANDIDATE"
    NEW_REGIME_CONFIRMED = "NEW_REGIME_CONFIRMED"
    TRANSITION_FAILED = "TRANSITION_FAILED"


SUBSTATE_DISPOSITIONS: Mapping[CandidateSubstate, SubstateDisposition] = {
    CandidateSubstate.TREND_CANDIDATE: SubstateDisposition.CANDIDATE,
    CandidateSubstate.TREND_CONFIRMED: SubstateDisposition.ENTERABLE,
    CandidateSubstate.TREND_INVALIDATED: SubstateDisposition.INVALIDATED,
    CandidateSubstate.PULLBACK_FORMING: SubstateDisposition.TERMINAL_WAIT,
    CandidateSubstate.PULLBACK_CONFIRMED: SubstateDisposition.ENTERABLE,
    CandidateSubstate.PULLBACK_INVALIDATED: SubstateDisposition.INVALIDATED,
    CandidateSubstate.PULLBACK_TOO_LATE: SubstateDisposition.INVALIDATED,
    CandidateSubstate.BREAKOUT_CANDIDATE: SubstateDisposition.CANDIDATE,
    CandidateSubstate.BREAKOUT_CONFIRMED: SubstateDisposition.ENTERABLE,
    CandidateSubstate.RETEST_PENDING: SubstateDisposition.TERMINAL_WAIT,
    CandidateSubstate.RETEST_CONFIRMED: SubstateDisposition.ENTERABLE,
    CandidateSubstate.FALSE_BREAKOUT: SubstateDisposition.INVALIDATED,
    CandidateSubstate.BREAKOUT_TOO_LATE: SubstateDisposition.INVALIDATED,
    CandidateSubstate.RANGE_EDGE_CANDIDATE: SubstateDisposition.CANDIDATE,
    CandidateSubstate.RANGE_REJECTION_CONFIRMED: SubstateDisposition.ENTERABLE,
    CandidateSubstate.RANGE_BROKEN: SubstateDisposition.INVALIDATED,
    CandidateSubstate.OLD_REGIME_DETERIORATING: SubstateDisposition.TERMINAL_WAIT,
    CandidateSubstate.TRANSITION_CANDIDATE: SubstateDisposition.CANDIDATE,
    CandidateSubstate.NEW_REGIME_CONFIRMED: SubstateDisposition.ENTERABLE,
    CandidateSubstate.TRANSITION_FAILED: SubstateDisposition.INVALIDATED,
}


@dataclass(frozen=True, slots=True)
class FrozenDecisionGap(RuntimeError):
    strategy: Strategy
    code: str
    description: str

    def __str__(self) -> str:
        return f"{self.strategy.value}:{self.code}:{self.description}"


@dataclass(frozen=True, slots=True)
class RuleObservation:
    code: str
    passed: bool | None
    available_at: datetime | None
    detail: str

    def __post_init__(self) -> None:
        if self.available_at is not None:
            object.__setattr__(self, "available_at", utc_datetime(self.available_at))

    def to_primitive(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "passed": self.passed,
            "available_at": self.available_at,
            "detail": self.detail,
        }


@dataclass(frozen=True, slots=True)
class CandidateEvaluation:
    candidate_episode_id: str
    overlap_group_id: str
    snapshot_id: str
    signal_episode_id: str | None
    strategy: Strategy
    side: Side
    decision_at: datetime
    status: CandidateStatus
    substate: CandidateSubstate | None
    disposition: SubstateDisposition | None
    reason_codes: tuple[str, ...]
    rules: tuple[RuleObservation, ...]
    frozen_gaps: tuple[FrozenDecisionGap, ...]
    generator_version: str
    metadata: tuple[tuple[str, Any], ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "decision_at", utc_datetime(self.decision_at))
        object.__setattr__(self, "reason_codes", tuple(sorted(set(self.reason_codes))))
        object.__setattr__(self, "rules", tuple(sorted(self.rules, key=lambda item: item.code)))
        object.__setattr__(self, "frozen_gaps", tuple(sorted(self.frozen_gaps, key=lambda item: item.code)))
        object.__setattr__(self, "metadata", frozen_pairs(self.metadata))
        if self.substate is None and self.disposition is not None:
            raise ValueError("disposition requires a substate")
        if self.substate is not None and SUBSTATE_DISPOSITIONS[self.substate] is not self.disposition:
            raise ValueError("substate disposition does not match frozen contract")
        if self.status is CandidateStatus.BLOCKED_FROZEN_DECISION_GAP and not self.frozen_gaps:
            raise ValueError("blocked evaluation requires FrozenDecisionGap")
        for rule in self.rules:
            if rule.available_at is not None and rule.available_at > self.decision_at:
                raise ValueError("candidate rule contains information after decision_at")
        expected = content_hash(self.identity_payload())
        if self.candidate_episode_id != expected:
            raise ValueError("candidate_episode_id does not match canonical identity")

    @classmethod
    def create(
        cls,
        *,
        snapshot_id: str,
        signal_episode_id: str | None,
        strategy: Strategy,
        side: Side,
        decision_at: datetime,
        status: CandidateStatus,
        substate: CandidateSubstate | None,
        reason_codes: Iterable[str],
        rules: Iterable[RuleObservation],
        frozen_gaps: Iterable[FrozenDecisionGap],
        generator_version: str,
        metadata: Mapping[str, Any] | tuple[tuple[str, Any], ...] = (),
    ) -> "CandidateEvaluation":
        boundary = utc_datetime(decision_at)
        primitive = {
            "snapshot_id": snapshot_id,
            "strategy": strategy,
            "side": side,
            "decision_at": boundary,
            "generator_version": generator_version,
        }
        return cls(
            candidate_episode_id=content_hash(primitive),
            overlap_group_id=snapshot_id,
            snapshot_id=snapshot_id,
            signal_episode_id=signal_episode_id,
            strategy=strategy,
            side=side,
            decision_at=boundary,
            status=status,
            substate=substate,
            disposition=SUBSTATE_DISPOSITIONS[substate] if substate else None,
            reason_codes=tuple(reason_codes),
            rules=tuple(rules),
            frozen_gaps=tuple(frozen_gaps),
            generator_version=generator_version,
            metadata=frozen_pairs(metadata),
        )

    def identity_payload(self) -> dict[str, Any]:
        return {
            "snapshot_id": self.snapshot_id,
            "strategy": self.strategy,
            "side": self.side,
            "decision_at": self.decision_at,
            "generator_version": self.generator_version,
        }

    def to_primitive(self) -> dict[str, Any]:
        return {
            "candidate_episode_id": self.candidate_episode_id,
            "overlap_group_id": self.overlap_group_id,
            "signal_episode_id": self.signal_episode_id,
            **self.identity_payload(),
            "status": self.status,
            "substate": self.substate,
            "disposition": self.disposition,
            "reason_codes": self.reason_codes,
            "rules": [item.to_primitive() for item in self.rules],
            "frozen_gaps": [str(item) for item in self.frozen_gaps],
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True, slots=True)
class CandidateSetup:
    """Outcome-free state linking candidate substates across causal snapshots."""

    setup_episode_id: str
    strategy: Strategy
    symbol: str
    side: Side
    started_at: datetime
    metadata: tuple[tuple[str, Any], ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "started_at", utc_datetime(self.started_at))
        object.__setattr__(self, "metadata", frozen_pairs(self.metadata))
        if not self.symbol or not self.setup_episode_id:
            raise ValueError("candidate setup identity cannot be empty")

    @classmethod
    def create(
        cls,
        *,
        strategy: Strategy,
        symbol: str,
        side: Side,
        started_at: datetime,
        identity: Mapping[str, Any],
        metadata: Mapping[str, Any] | tuple[tuple[str, Any], ...] = (),
    ) -> "CandidateSetup":
        return cls(
            setup_episode_id=content_hash(identity),
            strategy=strategy,
            symbol=symbol,
            side=side,
            started_at=started_at,
            metadata=frozen_pairs(metadata),
        )

    def to_primitive(self) -> dict[str, Any]:
        return {
            "setup_episode_id": self.setup_episode_id,
            "strategy": self.strategy,
            "symbol": self.symbol,
            "side": self.side,
            "started_at": self.started_at,
            "metadata": dict(self.metadata),
        }
