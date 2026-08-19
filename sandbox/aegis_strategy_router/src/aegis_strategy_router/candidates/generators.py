"""Five deterministic Phase 2 candidate generators under the frozen rules."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta

from aegis_strategy_router.candidates.base import CandidateGenerator
from aegis_strategy_router.candidates.contracts import (
    CandidateEvaluation,
    CandidateSetup,
    CandidateStatus,
    CandidateSubstate,
    FrozenDecisionGap,
    RuleObservation,
    Strategy,
)
from aegis_strategy_router.candidates.frozen_rules import (
    COMMON_TARGET_ATR,
    FrozenRuleInputUnavailable,
    at_range_edge,
    breakout_levels,
    closed_back_inside,
    common_target_space_available,
    direction,
    favorable_structural_space_atr,
    feature_observation,
    feature_value,
    flat_higher_timeframe_slope,
    higher_timeframe_alignment,
    low_directional_efficiency,
    new_structure,
    pullback_invalidation,
    pullback_opposition,
    pullback_realigned,
    range_geometry,
    range_rejection_confirmed,
    retest_touch_and_close_confirmed,
    stable_range,
    sustained_directional_move,
    timeframe_state,
    transition_slopes_confirm,
    trend_invalidation,
)
from aegis_strategy_router.domain.serialization import content_hash
from aegis_strategy_router.domain.types import MarketSnapshot, Side, Timeframe


def _rule(code: str, passed: bool | None, available_at, detail: str) -> RuleObservation:
    return RuleObservation(code, passed, available_at, detail)


def _space_observation(snapshot: MarketSnapshot, side: Side) -> RuleObservation:
    spaces = favorable_structural_space_atr(snapshot, side)
    passed = common_target_space_available(snapshot, side)
    detail = "NO_CAUSAL_FAVORABLE_OBSTRUCTION" if not spaces else ";".join(
        f"{timeframe.value}={distance:.12g}" for timeframe, distance in spaces
    )
    return _rule("COMMON_TARGET_STRUCTURAL_SPACE", passed, snapshot.decision_at, detail)


def _unknown(
    generator: CandidateGenerator,
    snapshot: MarketSnapshot,
    side: Side,
    error: FrozenRuleInputUnavailable,
    rules: tuple[RuleObservation, ...] = (),
) -> CandidateEvaluation:
    return generator.evaluation(
        snapshot,
        side,
        status=CandidateStatus.UNKNOWN,
        substate=None,
        reasons=("FROZEN_RULE_INPUT_UNAVAILABLE", str(error)),
        rules=rules,
    )


def _setup_metadata(
    *, setup_episode_id: str, started_at, kind: str, values: dict[str, object] | None = None
) -> dict[str, object]:
    return {
        "setup_active": True,
        "setup_episode_id": setup_episode_id,
        "setup_started_at": started_at,
        "setup_kind": kind,
        **(values or {}),
    }


def _prior_metadata(prior: CandidateSetup | None) -> dict[str, object]:
    return dict(prior.metadata) if prior is not None else {}


@dataclass(frozen=True, slots=True)
class TrendContinuationGenerator(CandidateGenerator):
    strategy: Strategy = Strategy.TREND_CONTINUATION

    @property
    def frozen_gaps(self) -> tuple[FrozenDecisionGap, ...]:
        return ()

    def evaluate(
        self, snapshot: MarketSnapshot, side: Side, prior_setup: CandidateSetup | None
    ) -> CandidateEvaluation:
        del prior_setup
        rules: list[RuleObservation] = []
        try:
            aligned = higher_timeframe_alignment(snapshot, side)
            invalidated = trend_invalidation(snapshot, side)
            sustained = sustained_directional_move(snapshot, side)
            space = common_target_space_available(snapshot, side)
            if space is None:
                raise FrozenRuleInputUnavailable("COMMON_TARGET_STRUCTURAL_SPACE:MISSING")
            rules.extend((
                _rule("TREND_HTF_ALIGNMENT", aligned, snapshot.decision_at, "1h_and_4h"),
                _rule("TREND_15M_NOT_INVALIDATED", not invalidated, snapshot.decision_at, "latest_confirmed_pivot"),
                _rule("SUSTAINED_DIRECTIONAL_MOVE", sustained, snapshot.decision_at, "5m_15m_return_and_unsigned_efficiency"),
                _space_observation(snapshot, side),
            ))
        except FrozenRuleInputUnavailable as error:
            return _unknown(self, snapshot, side, error, tuple(rules))
        if invalidated:
            return self.evaluation(snapshot, side, status=CandidateStatus.INELIGIBLE,
                                   substate=CandidateSubstate.TREND_INVALIDATED,
                                   reasons=("TREND_15M_INVALIDATED",), rules=tuple(rules))
        if aligned and sustained and space:
            return self.evaluation(snapshot, side, status=CandidateStatus.ELIGIBLE,
                                   substate=CandidateSubstate.TREND_CONFIRMED,
                                   reasons=("TREND_RULES_CONFIRMED",), rules=tuple(rules))
        return self.evaluation(snapshot, side, status=CandidateStatus.INELIGIBLE,
                               substate=CandidateSubstate.TREND_CANDIDATE,
                               reasons=("TREND_RULES_NOT_CONFIRMED",), rules=tuple(rules))


@dataclass(frozen=True, slots=True)
class PullbackContinuationGenerator(CandidateGenerator):
    strategy: Strategy = Strategy.PULLBACK_CONTINUATION

    @property
    def frozen_gaps(self) -> tuple[FrozenDecisionGap, ...]:
        return ()

    def evaluate(
        self, snapshot: MarketSnapshot, side: Side, prior_setup: CandidateSetup | None
    ) -> CandidateEvaluation:
        rules: list[RuleObservation] = []
        try:
            aligned = higher_timeframe_alignment(snapshot, side)
            invalidated = pullback_invalidation(snapshot, side)
            opposition = pullback_opposition(snapshot, side)
            realigned = pullback_realigned(snapshot, side)
            space = common_target_space_available(snapshot, side)
            if space is None:
                raise FrozenRuleInputUnavailable("COMMON_TARGET_STRUCTURAL_SPACE:MISSING")
            rules.extend((
                _rule("PULLBACK_HTF_ALIGNMENT", aligned, snapshot.decision_at, "reuses_trend_alignment"),
                _rule("PULLBACK_TEMPORARY_OPPOSITION", opposition, snapshot.decision_at, "1m_and_5m"),
                _rule("PULLBACK_NOT_INVALIDATED", not invalidated, snapshot.decision_at, "1h_pivot_vs_5m_close"),
                _rule("PULLBACK_REALIGNED", realigned, snapshot.decision_at, "1m_5m_return_and_1m_taker"),
                _space_observation(snapshot, side),
            ))
        except FrozenRuleInputUnavailable as error:
            return _unknown(self, snapshot, side, error, tuple(rules))
        prior_is_forming = prior_setup is not None and dict(prior_setup.metadata).get("setup_kind") == "PULLBACK_FORMING"
        if invalidated:
            return self.evaluation(snapshot, side, status=CandidateStatus.INELIGIBLE,
                                   substate=CandidateSubstate.PULLBACK_INVALIDATED,
                                   reasons=("PULLBACK_INVALIDATION_LEVEL_BROKEN",), rules=tuple(rules),
                                   metadata={"setup_clear": True})
        if not space:
            return self.evaluation(snapshot, side, status=CandidateStatus.INELIGIBLE,
                                   substate=CandidateSubstate.PULLBACK_TOO_LATE,
                                   reasons=("PULLBACK_COMMON_TARGET_SPACE_INSUFFICIENT",), rules=tuple(rules),
                                   metadata={"setup_clear": True})
        if prior_is_forming and aligned and realigned:
            return self.evaluation(snapshot, side, status=CandidateStatus.ELIGIBLE,
                                   substate=CandidateSubstate.PULLBACK_CONFIRMED,
                                   reasons=("PULLBACK_REALIGNMENT_CONFIRMED",), rules=tuple(rules),
                                   metadata={"setup_episode_id": prior_setup.setup_episode_id, "setup_clear": True})
        if aligned and (opposition or prior_is_forming):
            setup_id = prior_setup.setup_episode_id if prior_is_forming else content_hash({
                "strategy": self.strategy, "symbol": snapshot.symbol, "side": side,
                "forming_at": snapshot.decision_at,
            })
            metadata = _setup_metadata(
                setup_episode_id=setup_id,
                started_at=prior_setup.started_at if prior_is_forming else snapshot.decision_at,
                kind="PULLBACK_FORMING",
            )
            return self.evaluation(snapshot, side, status=CandidateStatus.ELIGIBLE,
                                   substate=CandidateSubstate.PULLBACK_FORMING,
                                   reasons=("PULLBACK_WAITING_FOR_REALIGNMENT",), rules=tuple(rules),
                                   metadata=metadata)
        return self.evaluation(snapshot, side, status=CandidateStatus.INELIGIBLE,
                               substate=None, reasons=("PULLBACK_SETUP_ABSENT",), rules=tuple(rules),
                               metadata={"setup_clear": prior_setup is not None})


@dataclass(frozen=True, slots=True)
class BreakoutRetestGenerator(CandidateGenerator):
    strategy: Strategy = Strategy.BREAKOUT_RETEST

    @property
    def frozen_gaps(self) -> tuple[FrozenDecisionGap, ...]:
        return ()

    def evaluate(
        self, snapshot: MarketSnapshot, side: Side, prior_setup: CandidateSetup | None
    ) -> CandidateEvaluation:
        rules: list[RuleObservation] = []
        try:
            state = timeframe_state(snapshot, Timeframe.M15)
            candle = state.latest_candle
            structural = state.structural
            if candle is None or structural is None or structural.atr14 is None:
                raise FrozenRuleInputUnavailable("15m:BREAKOUT_INPUT_UNAVAILABLE")
            prior = _prior_metadata(prior_setup)
            if prior_setup is not None and prior.get("setup_kind") == "BREAKOUT_RETEST":
                level_price = float(prior["level_price"])
                atr = float(prior["breakout_atr"])
                elapsed = snapshot.decision_at - prior_setup.started_at
                if snapshot.decision_at <= prior_setup.started_at:
                    raise FrozenRuleInputUnavailable("RETEST_SNAPSHOT_NOT_AFTER_BREAKOUT")
                confirmed = retest_touch_and_close_confirmed(
                    low=candle.low, high=candle.high, close=candle.close,
                    level_price=level_price, atr=atr, side=side,
                )
                failed = closed_back_inside(close=candle.close, level_price=level_price, side=side)
                rules.extend((
                    _rule("RETEST_WITHIN_60M", elapsed <= timedelta(minutes=60), candle.available_at, str(elapsed)),
                    _rule("RETEST_TOUCH_AND_CLOSE", confirmed, candle.available_at, str(prior["level_id"])),
                    _rule("RETEST_NOT_CLOSED_INSIDE", not failed, candle.available_at, str(prior["level_id"])),
                ))
                metadata = {"setup_episode_id": prior_setup.setup_episode_id, **prior}
                if elapsed > timedelta(minutes=60):
                    return self.evaluation(snapshot, side, status=CandidateStatus.INELIGIBLE,
                                           substate=None, reasons=("RETEST_WINDOW_EXPIRED",), rules=tuple(rules),
                                           metadata={**metadata, "setup_clear": True})
                if confirmed:
                    return self.evaluation(snapshot, side, status=CandidateStatus.ELIGIBLE,
                                           substate=CandidateSubstate.RETEST_CONFIRMED,
                                           reasons=("RETEST_CONFIRMED",), rules=tuple(rules),
                                           metadata={**metadata, "setup_clear": True})
                if failed:
                    return self.evaluation(snapshot, side, status=CandidateStatus.INELIGIBLE,
                                           substate=CandidateSubstate.FALSE_BREAKOUT,
                                           reasons=("CLOSED_BACK_INSIDE_BROKEN_LEVEL",), rules=tuple(rules),
                                           metadata={**metadata, "setup_clear": True})
                return self.evaluation(snapshot, side, status=CandidateStatus.ELIGIBLE,
                                       substate=CandidateSubstate.RETEST_PENDING,
                                       reasons=("RETEST_PENDING",), rules=tuple(rules),
                                       metadata=_setup_metadata(
                                           setup_episode_id=prior_setup.setup_episode_id,
                                           started_at=prior_setup.started_at,
                                           kind="BREAKOUT_RETEST",
                                           values=prior,
                                       ))

            levels = breakout_levels(snapshot, side)
            space = common_target_space_available(snapshot, side)
            if space is None:
                raise FrozenRuleInputUnavailable("COMMON_TARGET_STRUCTURAL_SPACE:MISSING")
            rules.extend((
                _rule("PRIOR_15M_LEVEL_BREAKOUT", bool(levels), candle.available_at, f"qualifying={len(levels)}"),
                _space_observation(snapshot, side),
            ))
            if not levels:
                return self.evaluation(snapshot, side, status=CandidateStatus.INELIGIBLE,
                                       substate=None, reasons=("BREAKOUT_SETUP_ABSENT",), rules=tuple(rules))
            level = levels[0]
            if not space:
                return self.evaluation(snapshot, side, status=CandidateStatus.INELIGIBLE,
                                       substate=CandidateSubstate.BREAKOUT_TOO_LATE,
                                       reasons=("BREAKOUT_COMMON_TARGET_SPACE_INSUFFICIENT",), rules=tuple(rules))
            identity = {
                "symbol": snapshot.symbol, "side": side, "timeframe": Timeframe.M15,
                "level_id": level.level_id, "breakout_close_at": candle.close_at,
            }
            setup_id = content_hash(identity)
            metadata = _setup_metadata(
                setup_episode_id=setup_id,
                started_at=candle.close_at,
                kind="BREAKOUT_RETEST",
                values={"level_id": level.level_id, "level_price": level.price,
                        "breakout_atr": structural.atr14, "breakout_close_at": candle.close_at},
            )
            return self.evaluation(snapshot, side, status=CandidateStatus.ELIGIBLE,
                                   substate=CandidateSubstate.BREAKOUT_CANDIDATE,
                                   reasons=("BREAKOUT_CANDIDATE_FROZEN_RULES_MET",), rules=tuple(rules),
                                   metadata=metadata)
        except FrozenRuleInputUnavailable as error:
            return _unknown(self, snapshot, side, error, tuple(rules))


@dataclass(frozen=True, slots=True)
class RangeMeanReversionGenerator(CandidateGenerator):
    strategy: Strategy = Strategy.RANGE_MEAN_REVERSION

    @property
    def frozen_gaps(self) -> tuple[FrozenDecisionGap, ...]:
        return ()

    def evaluate(
        self, snapshot: MarketSnapshot, side: Side, prior_setup: CandidateSetup | None
    ) -> CandidateEvaluation:
        rules: list[RuleObservation] = []
        try:
            candle = timeframe_state(snapshot, Timeframe.M15).latest_candle
            if candle is None:
                raise FrozenRuleInputUnavailable("15m:LATEST_CANDLE_MISSING")
            prior = _prior_metadata(prior_setup)
            if prior_setup is not None and prior.get("setup_kind") == "STABLE_RANGE":
                support = float(prior["support_price"])
                resistance = float(prior["resistance_price"])
                broken = candle.close < support if side is Side.LONG else candle.close > resistance
                if broken:
                    rules.append(_rule("RANGE_NOT_BROKEN", False, candle.available_at, f"{support}:{resistance}"))
                    return self.evaluation(snapshot, side, status=CandidateStatus.INELIGIBLE,
                                           substate=CandidateSubstate.RANGE_BROKEN,
                                           reasons=("RANGE_BOUNDARY_BROKEN",), rules=tuple(rules),
                                           metadata={"setup_episode_id": prior_setup.setup_episode_id, "setup_clear": True})
            low_efficiency = low_directional_efficiency(snapshot)
            flat = flat_higher_timeframe_slope(snapshot)
            is_stable, geometry = stable_range(snapshot)
            rules.extend((
                _rule("RANGE_LOW_EFFICIENCY", low_efficiency, snapshot.decision_at, "15m<=0.35"),
                _rule("RANGE_FLAT_HTF_SLOPE", flat, snapshot.decision_at, "1h_and_4h<=0.05"),
                _rule("RANGE_STABLE_GEOMETRY", is_stable, snapshot.decision_at,
                      "NO_BOUNDARIES" if geometry is None else f"width_atr={geometry.width_atr:.12g}"),
            ))
            if geometry is None or not is_stable:
                return self.evaluation(snapshot, side, status=CandidateStatus.INELIGIBLE,
                                       substate=None, reasons=("STABLE_RANGE_ABSENT",), rules=tuple(rules),
                                       metadata={"setup_clear": prior_setup is not None})
            edge = at_range_edge(geometry, side)
            rejection = range_rejection_confirmed(snapshot, geometry, side)
            rules.extend((
                _rule("RANGE_EDGE_WITHIN_0_20_ATR", edge, candle.available_at,
                      f"support={geometry.distance_to_support_atr:.12g};resistance={geometry.distance_to_resistance_atr:.12g}"),
                _rule("RANGE_REJECTION_CLOSE_INWARD", rejection, candle.available_at, f"midpoint={geometry.midpoint:.12g}"),
            ))
            setup_id = content_hash({
                "strategy": self.strategy, "symbol": snapshot.symbol,
                "support_id": geometry.support.level_id, "resistance_id": geometry.resistance.level_id,
            })
            metadata = _setup_metadata(
                setup_episode_id=setup_id,
                started_at=prior_setup.started_at if prior_setup is not None else snapshot.decision_at,
                kind="STABLE_RANGE",
                values={"support_price": geometry.support.price, "resistance_price": geometry.resistance.price,
                        "midpoint": geometry.midpoint},
            )
            if edge and rejection:
                return self.evaluation(snapshot, side, status=CandidateStatus.ELIGIBLE,
                                       substate=CandidateSubstate.RANGE_REJECTION_CONFIRMED,
                                       reasons=("RANGE_REJECTION_CONFIRMED",), rules=tuple(rules), metadata=metadata)
            if edge:
                return self.evaluation(snapshot, side, status=CandidateStatus.ELIGIBLE,
                                       substate=CandidateSubstate.RANGE_EDGE_CANDIDATE,
                                       reasons=("RANGE_EDGE_CANDIDATE",), rules=tuple(rules), metadata=metadata)
            return self.evaluation(snapshot, side, status=CandidateStatus.INELIGIBLE,
                                   substate=None, reasons=("PRICE_NOT_AT_RANGE_EDGE",), rules=tuple(rules), metadata=metadata)
        except FrozenRuleInputUnavailable as error:
            return _unknown(self, snapshot, side, error, tuple(rules))


@dataclass(frozen=True, slots=True)
class RegimeTransitionGenerator(CandidateGenerator):
    strategy: Strategy = Strategy.REGIME_TRANSITION_REVERSAL

    @property
    def frozen_gaps(self) -> tuple[FrozenDecisionGap, ...]:
        return ()

    def evaluate(
        self, snapshot: MarketSnapshot, side: Side, prior_setup: CandidateSetup | None
    ) -> CandidateEvaluation:
        rules: list[RuleObservation] = []
        try:
            old_side = Side.SHORT if side is Side.LONG else Side.LONG
            prior_trend = higher_timeframe_alignment(snapshot, old_side)
            prior_range, geometry = stable_range(snapshot)
            prior = _prior_metadata(prior_setup)
            prior_range_setup = prior_setup is not None and prior.get("setup_kind") == "PRIOR_RANGE"
            prior_transition = prior_setup is not None and prior.get("setup_kind") == "TRANSITION"
            old_signed = direction(old_side)
            trend_deterioration = (
                prior_trend
                and old_signed * feature_value(snapshot, Timeframe.M15, "ema25_slope_atr") <= 0
                and trend_invalidation(snapshot, old_side)
            )
            range_deterioration = False
            if prior_range_setup:
                candle = timeframe_state(snapshot, Timeframe.M15).latest_candle
                if candle is None:
                    raise FrozenRuleInputUnavailable("15m:LATEST_CANDLE_MISSING")
                boundary = float(prior["resistance_price"] if side is Side.LONG else prior["support_price"])
                range_deterioration = candle.close > boundary if side is Side.LONG else candle.close < boundary
            deteriorating = trend_deterioration or range_deterioration or prior_transition
            formed = new_structure(snapshot, side)
            confirmed = formed and transition_slopes_confirm(snapshot, side)
            rules.extend((
                _rule("PRIOR_OPPOSITE_TREND", prior_trend, snapshot.decision_at, old_side.value),
                _rule("PRIOR_STABLE_RANGE", prior_range or prior_range_setup, snapshot.decision_at, "frozen_range"),
                _rule("PRIOR_REGIME_DETERIORATING", deteriorating, snapshot.decision_at, "trend_or_range"),
                _rule("NEW_15M_STRUCTURE", formed, snapshot.decision_at, side.value),
                _rule("TRANSITION_SLOPES_CONFIRM", confirmed, snapshot.decision_at, "15m_and_1h"),
            ))
            if deteriorating:
                setup_id = prior_setup.setup_episode_id if (prior_range_setup or prior_transition) else content_hash({
                    "strategy": self.strategy, "symbol": snapshot.symbol, "side": side,
                    "deterioration_at": snapshot.decision_at,
                })
                metadata = _setup_metadata(
                    setup_episode_id=setup_id,
                    started_at=prior_setup.started_at if (prior_range_setup or prior_transition) else snapshot.decision_at,
                    kind="TRANSITION",
                )
                if confirmed:
                    return self.evaluation(snapshot, side, status=CandidateStatus.ELIGIBLE,
                                           substate=CandidateSubstate.NEW_REGIME_CONFIRMED,
                                           reasons=("NEW_REGIME_CONFIRMED",), rules=tuple(rules),
                                           metadata={**metadata, "setup_clear": True})
                if formed:
                    return self.evaluation(snapshot, side, status=CandidateStatus.ELIGIBLE,
                                           substate=CandidateSubstate.TRANSITION_CANDIDATE,
                                           reasons=("TRANSITION_STRUCTURE_FORMING",), rules=tuple(rules), metadata=metadata)
                return self.evaluation(snapshot, side, status=CandidateStatus.ELIGIBLE,
                                       substate=CandidateSubstate.OLD_REGIME_DETERIORATING,
                                       reasons=("OLD_REGIME_DETERIORATING",), rules=tuple(rules), metadata=metadata)
            if prior_range and geometry is not None:
                setup_id = content_hash({
                    "strategy": self.strategy, "symbol": snapshot.symbol,
                    "support_id": geometry.support.level_id, "resistance_id": geometry.resistance.level_id,
                })
                metadata = _setup_metadata(
                    setup_episode_id=setup_id, started_at=snapshot.decision_at, kind="PRIOR_RANGE",
                    values={"support_price": geometry.support.price, "resistance_price": geometry.resistance.price},
                )
                return self.evaluation(snapshot, side, status=CandidateStatus.INELIGIBLE,
                                       substate=None, reasons=("PRIOR_RANGE_NOT_DETERIORATING",),
                                       rules=tuple(rules), metadata=metadata)
            if not prior_trend and not prior_range_setup:
                return self.evaluation(snapshot, side, status=CandidateStatus.UNKNOWN,
                                       substate=None, reasons=("PRIOR_REGIME_UNKNOWN",), rules=tuple(rules),
                                       metadata={"setup_clear": prior_setup is not None})
            return self.evaluation(snapshot, side, status=CandidateStatus.INELIGIBLE,
                                   substate=None, reasons=("PRIOR_REGIME_NOT_DETERIORATING",), rules=tuple(rules))
        except FrozenRuleInputUnavailable as error:
            return _unknown(self, snapshot, side, error, tuple(rules))
