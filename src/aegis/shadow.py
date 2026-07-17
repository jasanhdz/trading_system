"""Non-operational shadow outcome resolution and deterministic replay."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Mapping, Sequence

from .domain import (
    Candle, DecisionOutcome, DecisionRequest, DecisionResponse, DecisionStatus,
    EvidenceMode, FillOutcome, OutcomeExecutionStatus, TradeSide,
)
from .runtime import BrainRuntime


class ShadowOutcomeError(ValueError):
    pass


def resolve_shadow_outcome(
    response: DecisionResponse,
    future_candles: Mapping[str, Sequence[Candle]],
    *,
    friction_fraction: float,
    mode: EvidenceMode = EvidenceMode.SHADOW,
) -> DecisionOutcome | None:
    """Resolve an H12 hypothesis using only final candles after the decision."""
    if response.status is DecisionStatus.NO_TRADE:
        return None
    if response.status is not DecisionStatus.SELECTED or len(response.selected) != 1:
        raise ShadowOutcomeError("shadow outcome requires exactly one selected candidate")
    if not math.isfinite(friction_fraction) or friction_fraction < 0:
        raise ShadowOutcomeError("friction must be finite and non-negative")
    if mode not in (EvidenceMode.SHADOW, EvidenceMode.REPLAY):
        raise ShadowOutcomeError("outcome resolution is restricted to shadow/replay")
    candidate = response.selected[0]
    candles = tuple(future_candles.get(candidate.symbol, ()))
    horizon = candidate.horizon_bars
    if len(candles) < horizon:
        raise ShadowOutcomeError("shadow outcome is not mature")
    evaluation = candles[:horizon]
    if any(not candle.is_closed or candle.open_time < response.generated_at for candle in evaluation):
        raise ShadowOutcomeError("shadow candles must be final and strictly forward")
    if any(evaluation[index].open_time <= evaluation[index - 1].open_time for index in range(1, len(evaluation))):
        raise ShadowOutcomeError("shadow candles are not ordered")
    if any(evaluation[index].open_time != evaluation[index - 1].close_time for index in range(1, len(evaluation))):
        raise ShadowOutcomeError("shadow candles contain a temporal gap")
    entry = evaluation[0].open
    exit_price = evaluation[-1].close
    stop_distance = candidate.risk_intent.stop_distance_fraction
    if candidate.side is TradeSide.LONG:
        favorable = max(candle.high for candle in evaluation) / entry - 1.0
        adverse = (entry - min(candle.low for candle in evaluation)) / entry
        horizon_return = exit_price / entry - 1.0
    elif candidate.side is TradeSide.SHORT:
        favorable = (entry - min(candle.low for candle in evaluation)) / entry
        adverse = max(candle.high for candle in evaluation) / entry - 1.0
        horizon_return = entry / exit_price - 1.0
    else:
        raise ShadowOutcomeError("NO_TRADE candidate cannot have a shadow outcome")
    invalidated = stop_distance is not None and adverse >= stop_distance
    gross_return = -stop_distance if invalidated and stop_distance is not None else horizon_return
    net_return = gross_return - friction_fraction
    details: dict[str, float | str | bool] = {
        "hypothetical_entry_price": entry, "hypothetical_exit_price": exit_price,
        "gross_return_fraction": gross_return, "net_return_fraction": net_return,
        "favorable_excursion_fraction": max(0.0, favorable),
        "adverse_excursion_fraction": max(0.0, adverse), "friction_fraction": friction_fraction,
        "horizon_bars": float(horizon), "invalidation_triggered": invalidated,
        "outcome_policy": "NEXT_BAR_OPEN_TO_H12_CLOSE_WITH_SCIENTIFIC_STOP",
    }
    return DecisionOutcome(
        decision_id=response.decision_id, decision_cycle_id=response.decision_cycle_id,
        candidate_hash=candidate.candidate_hash, accepted=False, executed=False,
        rejection_reason="SHADOW_NON_EXECUTING", fill=FillOutcome(OutcomeExecutionStatus.NOT_EXECUTED),
        closed_at=evaluation[-1].close_time, realized_pnl=None,
        close_reason="SCIENTIFIC_INVALIDATION" if invalidated else "H12_MATURITY",
        incidents=(), reconciled=True, occurred_at=evaluation[-1].close_time,
        execution_mode=mode, hypothetical_details=details,
    )


@dataclass(frozen=True)
class ShadowReplayCase:
    request: DecisionRequest
    future_candles: Mapping[str, Sequence[Candle]]


@dataclass(frozen=True)
class ShadowReplayResult:
    decisions: tuple[DecisionResponse, ...]
    outcomes: tuple[DecisionOutcome, ...]
    no_trade_count: int


class ShadowReplayEngine:
    def __init__(self, runtime: BrainRuntime, *, friction_fraction: float) -> None:
        self.runtime = runtime
        self.friction_fraction = friction_fraction

    def run(self, cases: Sequence[ShadowReplayCase]) -> ShadowReplayResult:
        ordered = sorted(cases, key=lambda item: (item.request.snapshot.closed_at, item.request.decision_cycle_id))
        if len({case.request.decision_cycle_id for case in ordered}) != len(ordered):
            raise ShadowOutcomeError("duplicate replay decision cycle")
        decisions: list[DecisionResponse] = []
        outcomes: list[DecisionOutcome] = []
        previous_time = None
        for case in ordered:
            if previous_time is not None and case.request.snapshot.closed_at <= previous_time:
                raise ShadowOutcomeError("replay cycles must be strictly chronological")
            previous_time = case.request.snapshot.closed_at
            setter = getattr(self.runtime.clock, "set", None)
            if setter is not None:
                setter(case.request.snapshot.closed_at)
            decision = self.runtime.evaluate(case.request)
            decisions.append(decision)
            outcome = resolve_shadow_outcome(decision, case.future_candles,
                                             friction_fraction=self.friction_fraction, mode=EvidenceMode.REPLAY)
            if outcome is not None:
                self.runtime.evidence.record_outcome(outcome)
                outcomes.append(outcome)
        return ShadowReplayResult(tuple(decisions), tuple(outcomes),
                                  sum(decision.status is DecisionStatus.NO_TRADE for decision in decisions))
