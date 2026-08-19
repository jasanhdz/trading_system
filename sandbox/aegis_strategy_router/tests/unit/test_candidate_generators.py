from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timedelta, timezone

import pytest

from aegis_strategy_router.candidates.base import GENERATOR_VERSION
from aegis_strategy_router.candidates.contracts import (
    CandidateEvaluation,
    CandidateStatus,
    CandidateSubstate,
    RuleObservation,
    SUBSTATE_DISPOSITIONS,
    Strategy,
    SubstateDisposition,
)
from aegis_strategy_router.candidates.frozen_rules import (
    COMMON_TARGET_ATR,
    breakout_close_confirmed,
    breakout_too_late,
    retest_touch_and_close_confirmed,
)
from aegis_strategy_router.candidates.registry import (
    CandidateEpisodeIndex,
    CandidateGeneratorRegistry,
    CandidateReplayContext,
    DuplicateCandidateEpisode,
)
from aegis_strategy_router.domain.serialization import content_hash
from aegis_strategy_router.domain.types import (
    Candle,
    ConfirmedPivot,
    DataStatus,
    FeatureObservation,
    FeatureSet,
    LevelDistance,
    MarketSnapshot,
    PivotKind,
    Side,
    StructuralContext,
    StructuralLevel,
    Timeframe,
    TimeframeSnapshot,
)


NOW = datetime(2026, 8, 17, 22, tzinfo=timezone.utc)


def _feature(timeframe: Timeframe, name: str, value: float, at: datetime) -> FeatureObservation:
    return FeatureObservation(
        f"tf{timeframe.value}__{name}", value, at, at, "fixture", "fixture-v1", DataStatus.AVAILABLE
    )


def _level(kind: PivotKind, price: float, timeframe: Timeframe, at: datetime) -> StructuralLevel:
    return StructuralLevel(
        level_id=content_hash({"kind": kind, "price": price, "timeframe": timeframe}),
        timeframe=timeframe,
        kind=kind,
        price=price,
        touch_count=2,
        pivot_indices=(1, 4),
        pivot_prices=(price, price),
        first_touch_at=at - timeframe.duration * 6,
        last_touch_at=at - timeframe.duration * 3,
        available_at=at - timeframe.duration,
    )


def _pivots(side: Side, at: datetime) -> tuple[ConfirmedPivot, ...]:
    if side is Side.LONG:
        high_prices, low_prices = (100.0, 105.0), (90.0, 95.0)
    else:
        high_prices, low_prices = (110.0, 105.0), (100.0, 95.0)
    values = []
    for kind, prices in ((PivotKind.HIGH, high_prices), (PivotKind.LOW, low_prices)):
        for offset, price in zip((3, 2), prices):
            values.append(ConfirmedPivot(
                kind, price, offset * 3, at - timedelta(hours=offset * 3),
                at - timedelta(hours=offset * 3 - 1),
            ))
    return tuple(values)


def market(
    side: Side,
    *,
    at: datetime = NOW,
    one_bar_sign: float | None = None,
    close: float | None = None,
    m15_levels: tuple[StructuralLevel, ...] = (),
    m15_atr: float = 10.0,
    range_features: bool = False,
) -> MarketSnapshot:
    signed = 1.0 if side is Side.LONG else -1.0
    one_bar = signed if one_bar_sign is None else one_bar_sign
    last_close = 102.0 if side is Side.LONG else 98.0
    if close is not None:
        last_close = close
    states = []
    for timeframe in Timeframe:
        candle = Candle(
            open_at=at - timeframe.duration,
            close_at=at,
            open=last_close - signed * 0.25,
            high=max(last_close, last_close - signed * 0.25) + 0.5,
            low=min(last_close, last_close - signed * 0.25) - 0.5,
            close=last_close,
            volume=100.0,
            taker_buy_volume=55.0,
            available_at=at,
            source_id=f"fixture:{timeframe.value}:{at.isoformat()}",
        )
        values = {
            "return_1_bps": one_bar,
            "return_3_bps": signed,
            "path_efficiency_6": 0.2 if range_features else 0.8,
            "ema25_slope_atr": 0.01 if range_features and timeframe in (Timeframe.H1, Timeframe.H4) else signed * 0.1,
            "taker_imbalance": signed * 0.2,
        }
        features = FeatureSet(
            tuple(_feature(timeframe, name, value, at) for name, value in values.items()),
            "fixture-schema",
        )
        structural = None
        if timeframe.structural_lookback is not None:
            levels = m15_levels if timeframe is Timeframe.M15 else ()
            structural = StructuralContext(
                status=DataStatus.AVAILABLE,
                pivots=_pivots(side, at),
                levels=levels,
                atr14=m15_atr,
                cluster_tolerance=2.0,
                reference_price=last_close,
                nearest_below=LevelDistance("below", last_close - 10.0, 1.0, 100.0),
                nearest_above=LevelDistance("above", last_close + 10.0, 1.0, 100.0),
            )
        states.append(TimeframeSnapshot(
            timeframe=timeframe,
            status=DataStatus.AVAILABLE,
            candle_count=120,
            required_warmup_bars=99,
            latest_closed_at=at,
            latest_candle=candle,
            features=features,
            structural=structural,
        ))
    return MarketSnapshot.create(
        schema_version="fixture-v1",
        schema_hash="fixture-schema",
        symbol="TESTUSDT",
        decision_at=at,
        built_at=at,
        proposed_side=side,
        signal_id=f"signal:{side.value}:{at.isoformat()}",
        reference_price=last_close,
        timeframes=states,
        source_versions={"fixture": "v1"},
    )


def _replace_candle(snapshot: MarketSnapshot, timeframe: Timeframe, candle: Candle) -> MarketSnapshot:
    states = tuple(
        replace(state, latest_closed_at=candle.close_at, latest_candle=candle)
        if state.timeframe is timeframe else state
        for state in snapshot.timeframes
    )
    return MarketSnapshot.create(
        schema_version=snapshot.schema_version,
        schema_hash=snapshot.schema_hash,
        symbol=snapshot.symbol,
        decision_at=candle.close_at,
        built_at=candle.close_at,
        proposed_side=snapshot.proposed_side,
        signal_id=f"later:{candle.close_at.isoformat()}",
        reference_price=candle.close,
        timeframes=states,
        source_versions=dict(snapshot.source_versions),
    )


def test_registry_contains_exactly_five_generators_and_no_frozen_gaps() -> None:
    evaluations = CandidateGeneratorRegistry().generate_all(market(Side.LONG), Side.LONG)
    assert {item.strategy for item in evaluations} == set(Strategy)
    assert all(item.status is not CandidateStatus.BLOCKED_FROZEN_DECISION_GAP for item in evaluations)
    assert all(not item.frozen_gaps for item in evaluations)


def test_trend_rule_execution_is_deterministic_and_mirrored() -> None:
    registry = CandidateGeneratorRegistry()
    long_first = registry.generate_all(market(Side.LONG), Side.LONG)
    long_second = registry.generate_all(market(Side.LONG), Side.LONG)
    short = registry.generate_all(market(Side.SHORT), Side.SHORT)
    assert long_first == long_second
    long_trend = next(item for item in long_first if item.strategy is Strategy.TREND_CONTINUATION)
    short_trend = next(item for item in short if item.strategy is Strategy.TREND_CONTINUATION)
    assert long_trend.status is short_trend.status is CandidateStatus.ELIGIBLE
    assert long_trend.substate is short_trend.substate is CandidateSubstate.TREND_CONFIRMED
    assert [rule.passed for rule in long_trend.rules] == [rule.passed for rule in short_trend.rules]


def test_pullback_forming_then_later_realigns_without_reusing_old_identity() -> None:
    context = CandidateReplayContext()
    registry = CandidateGeneratorRegistry()
    first_market = market(Side.LONG, one_bar_sign=-1.0)
    first = registry.generate_all_replay(first_market, Side.LONG, context)
    forming = next(item for item in first if item.strategy is Strategy.PULLBACK_CONTINUATION)
    assert forming.substate is CandidateSubstate.PULLBACK_FORMING
    later_market = market(Side.LONG, at=NOW + timedelta(minutes=1), one_bar_sign=1.0)
    later = registry.generate_all_replay(later_market, Side.LONG, context)
    confirmed = next(item for item in later if item.strategy is Strategy.PULLBACK_CONTINUATION)
    assert confirmed.substate is CandidateSubstate.PULLBACK_CONFIRMED
    assert confirmed.status is CandidateStatus.ELIGIBLE
    assert confirmed.candidate_episode_id != forming.candidate_episode_id
    assert dict(confirmed.metadata)["setup_episode_id"] == dict(forming.metadata)["setup_episode_id"]


def test_breakout_retest_sequence_uses_prior_level_and_later_snapshot() -> None:
    level = _level(PivotKind.HIGH, 100.0, Timeframe.M15, NOW)
    first_market = market(Side.LONG, close=102.0, m15_levels=(level,))
    context = CandidateReplayContext()
    registry = CandidateGeneratorRegistry()
    first = registry.generate_all_replay(first_market, Side.LONG, context)
    breakout = next(item for item in first if item.strategy is Strategy.BREAKOUT_RETEST)
    assert breakout.substate is CandidateSubstate.BREAKOUT_CANDIDATE
    later_at = NOW + timedelta(minutes=15)
    later_candle = Candle(
        open_at=NOW, close_at=later_at, open=102.0, high=102.5, low=99.5, close=101.0,
        volume=100.0, taker_buy_volume=55.0, available_at=later_at, source_id="retest",
    )
    later_market = _replace_candle(market(Side.LONG, at=later_at, close=101.0, m15_levels=(level,)), Timeframe.M15, later_candle)
    later = registry.generate_all_replay(later_market, Side.LONG, context)
    retest = next(item for item in later if item.strategy is Strategy.BREAKOUT_RETEST)
    assert retest.substate is CandidateSubstate.RETEST_CONFIRMED
    assert dict(retest.metadata)["setup_episode_id"] == dict(breakout.metadata)["setup_episode_id"]
    assert context.setup(Strategy.BREAKOUT_RETEST, "TESTUSDT", Side.LONG) is None


def test_breakout_close_back_inside_is_invalidated_before_any_later_confirmation() -> None:
    level = _level(PivotKind.HIGH, 100.0, Timeframe.M15, NOW)
    context = CandidateReplayContext()
    registry = CandidateGeneratorRegistry()
    first = registry.generate_all_replay(
        market(Side.LONG, close=102.0, m15_levels=(level,)), Side.LONG, context
    )
    assert next(item for item in first if item.strategy is Strategy.BREAKOUT_RETEST).substate is CandidateSubstate.BREAKOUT_CANDIDATE
    later_at = NOW + timedelta(minutes=15)
    failed_candle = Candle(
        open_at=NOW, close_at=later_at, open=102.0, high=102.2, low=98.5, close=99.0,
        volume=100.0, taker_buy_volume=45.0, available_at=later_at, source_id="false-break",
    )
    later_market = _replace_candle(
        market(Side.LONG, at=later_at, close=99.0, m15_levels=(level,)),
        Timeframe.M15,
        failed_candle,
    )
    later = registry.generate_all_replay(later_market, Side.LONG, context)
    failed = next(item for item in later if item.strategy is Strategy.BREAKOUT_RETEST)
    assert failed.status is CandidateStatus.INELIGIBLE
    assert failed.substate is CandidateSubstate.FALSE_BREAKOUT


def test_sequence_state_is_isolated_by_symbol_and_side() -> None:
    context = CandidateReplayContext()
    registry = CandidateGeneratorRegistry()
    registry.generate_all_replay(market(Side.LONG, one_bar_sign=-1.0), Side.LONG, context)
    short_later = market(Side.SHORT, at=NOW + timedelta(minutes=1), one_bar_sign=-1.0)
    evaluations = registry.generate_all_replay(short_later, Side.SHORT, context)
    pullback = next(item for item in evaluations if item.strategy is Strategy.PULLBACK_CONTINUATION)
    assert pullback.substate is not CandidateSubstate.PULLBACK_CONFIRMED


def test_replay_rejects_retroactive_snapshot_order() -> None:
    context = CandidateReplayContext()
    registry = CandidateGeneratorRegistry()
    registry.generate_all_replay(market(Side.LONG, at=NOW + timedelta(minutes=1)), Side.LONG, context)
    with pytest.raises(ValueError, match="chronological"):
        registry.generate_all_replay(market(Side.LONG, at=NOW), Side.LONG, context)


@pytest.mark.parametrize("side,close,kind", [
    (Side.LONG, 91.0, PivotKind.LOW),
    (Side.SHORT, 109.0, PivotKind.HIGH),
])
def test_range_edge_and_rejection_are_long_short_symmetric(side: Side, close: float, kind: PivotKind) -> None:
    del kind
    support = _level(PivotKind.LOW, 90.0, Timeframe.M15, NOW)
    resistance = _level(PivotKind.HIGH, 110.0, Timeframe.M15, NOW)
    snapshot = market(side, close=close, m15_levels=(support, resistance), range_features=True)
    evaluation = CandidateGeneratorRegistry().generators[Strategy.RANGE_MEAN_REVERSION].generate(snapshot, side)
    assert evaluation.status is CandidateStatus.ELIGIBLE
    assert evaluation.substate is CandidateSubstate.RANGE_REJECTION_CONFIRMED


def test_missing_snapshot_data_is_unknown_before_strategy_rules() -> None:
    base = market(Side.SHORT)
    states = tuple(replace(state, status=DataStatus.UNKNOWN, reason="NO_DATA") for state in base.timeframes)
    unavailable = MarketSnapshot.create(
        schema_version=base.schema_version, schema_hash=base.schema_hash, symbol=base.symbol,
        decision_at=base.decision_at, built_at=base.built_at, proposed_side=base.proposed_side,
        signal_id=base.signal_id, reference_price=base.reference_price, timeframes=states,
        source_versions=dict(base.source_versions),
    )
    evaluations = CandidateGeneratorRegistry().generate_all(unavailable, Side.SHORT)
    assert all(item.status is CandidateStatus.UNKNOWN for item in evaluations)
    assert all(item.reason_codes == ("SNAPSHOT_DATA_UNAVAILABLE",) for item in evaluations)


def test_episode_identity_and_overlap_are_snapshot_scoped() -> None:
    evaluations = CandidateGeneratorRegistry().generate_all(market(Side.LONG), Side.LONG)
    index = CandidateEpisodeIndex()
    for item in evaluations:
        index.add(item)
    assert len(index.overlap_ids(evaluations[0].snapshot_id)) == 5
    with pytest.raises(DuplicateCandidateEpisode):
        index.add(evaluations[0])


def test_all_frozen_substates_have_explicit_dispositions() -> None:
    assert set(SUBSTATE_DISPOSITIONS) == set(CandidateSubstate)
    assert SUBSTATE_DISPOSITIONS[CandidateSubstate.PULLBACK_FORMING] is SubstateDisposition.TERMINAL_WAIT
    assert SUBSTATE_DISPOSITIONS[CandidateSubstate.RETEST_PENDING] is SubstateDisposition.TERMINAL_WAIT
    assert SUBSTATE_DISPOSITIONS[CandidateSubstate.BREAKOUT_CONFIRMED] is SubstateDisposition.ENTERABLE
    assert SUBSTATE_DISPOSITIONS[CandidateSubstate.TRANSITION_FAILED] is SubstateDisposition.INVALIDATED


def test_rule_observation_after_decision_is_rejected() -> None:
    future = NOW + timedelta(seconds=1)
    with pytest.raises(ValueError, match="after decision_at"):
        CandidateEvaluation.create(
            snapshot_id="snapshot", signal_episode_id=None,
            strategy=Strategy.TREND_CONTINUATION, side=Side.LONG, decision_at=NOW,
            status=CandidateStatus.INELIGIBLE, substate=CandidateSubstate.TREND_INVALIDATED,
            reason_codes=("TEST",), rules=(RuleObservation("TEST", False, future, "future"),),
            frozen_gaps=(), generator_version=GENERATOR_VERSION,
        )


def test_frozen_breakout_rules_are_directionally_symmetric() -> None:
    assert breakout_close_confirmed(close=101.0, level_price=100.0, atr=10.0, side=Side.LONG)
    assert breakout_close_confirmed(close=99.0, level_price=100.0, atr=10.0, side=Side.SHORT)
    assert not breakout_close_confirmed(close=100.9, level_price=100.0, atr=10.0, side=Side.LONG)
    assert not breakout_close_confirmed(close=99.1, level_price=100.0, atr=10.0, side=Side.SHORT)
    assert retest_touch_and_close_confirmed(
        low=99.5, high=101.0, close=100.1, level_price=100.0, atr=10.0, side=Side.LONG
    )
    assert retest_touch_and_close_confirmed(
        low=99.0, high=100.5, close=99.9, level_price=100.0, atr=10.0, side=Side.SHORT
    )
    assert breakout_too_late(remaining_space_atr=0.49)
    assert not breakout_too_late(remaining_space_atr=COMMON_TARGET_ATR)
