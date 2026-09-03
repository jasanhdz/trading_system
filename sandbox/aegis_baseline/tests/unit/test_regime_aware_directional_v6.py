from __future__ import annotations

from datetime import datetime, timedelta, timezone

from aegis.domain import Candle
from aegis.research.hybrid_ts_protection_replay import TsProtectionConfig
from aegis.research.regime_aware_directional_v6 import (
    CommitteeObservation,
    DirectionalPathContract,
    DirectionalRole,
    ExitEyeReplayConfig,
    classify_regime_axes,
    directional_interactions,
    directional_path_outcome,
    directional_role,
    realized_global_regime,
    regime_aware_feature_vector,
    regime_router_feature_vector,
    replay_full_lifecycle,
)
from aegis.research.long_entry_v21_shadow import LONG_V21_FEATURE_NAMES
from aegis.training.hybrid_directional import DirectionalSide

NOW = datetime(2026, 1, 1, tzinfo=timezone.utc)


def candle(index: int, open_: float, high: float, low: float, close: float) -> Candle:
    opened = NOW + timedelta(minutes=5 * index)
    return Candle(
        opened,
        opened + timedelta(minutes=5),
        open_,
        high,
        low,
        close,
        100.0,
        True,
        "TEST",
    )


def path_contract() -> DirectionalPathContract:
    return DirectionalPathContract(
        leverage=10.0,
        roe_checkpoints=(0.05, 0.10, 0.20, 0.50),
        primary_protectable_roe=0.05,
        favorable_atr_multiple=1.0,
        adverse_atr_multiple=0.75,
        favorable_floor_fraction=0.002,
        adverse_floor_fraction=0.002,
        favorable_ceiling_fraction=0.015,
        adverse_ceiling_fraction=0.012,
        fast_success_bars=3,
        early_reversal_bars=2,
        round_trip_cost_fraction=0.001,
    )


def exit_eye() -> ExitEyeReplayConfig:
    return ExitEyeReplayConfig(
        enabled=True,
        min_roe_to_protect=0.08,
        min_peak_roe_to_protect=0.12,
        min_giveback_from_peak_roe=0.04,
        neutral_votes_to_protect=2,
        opposite_votes_to_close=2,
        min_roe_to_close_on_opposite=0.06,
        min_peak_roe_to_close_on_opposite=0.10,
        close_on_neutral_decay=True,
        neutral_close_votes=3,
        min_roe_to_close_on_neutral=0.08,
        min_peak_roe_to_close_on_neutral=0.12,
        min_giveback_to_close_on_neutral=0.04,
        require_consecutive_neutral_close=2,
        require_consecutive_neutral=2,
        require_consecutive_opposite=1,
        min_minutes_in_trade=3,
    )


def test_directional_path_labels_long_and_short_symmetrically() -> None:
    signal = candle(0, 100, 100.2, 99.8, 100)
    long_future = (
        candle(1, 100, 100.7, 99.9, 100.5),
        candle(2, 100.5, 101.2, 100.4, 101.0),
    )
    short_future = (
        candle(1, 100, 100.1, 99.3, 99.5),
        candle(2, 99.5, 99.6, 98.8, 99.0),
    )
    long = directional_path_outcome(
        signal=signal,
        future=long_future,
        atr_fraction=0.005,
        side=DirectionalSide.LONG,
        contract=path_contract(),
    )
    short = directional_path_outcome(
        signal=signal,
        future=short_future,
        atr_fraction=0.005,
        side=DirectionalSide.SHORT,
        contract=path_contract(),
    )
    assert long["target_before_stop"] is True
    assert short["target_before_stop"] is True
    assert long["protectable_advantage"] is True
    assert short["protectable_advantage"] is True


def test_same_bar_target_and_stop_fails_closed() -> None:
    result = directional_path_outcome(
        signal=candle(0, 100, 100, 100, 100),
        future=(candle(1, 100, 101, 99, 100),),
        atr_fraction=0.005,
        side=DirectionalSide.LONG,
        contract=path_contract(),
    )
    assert result["same_bar_ambiguity"] is True
    assert result["target_before_stop"] is False


def test_regime_and_roles_are_asymmetric() -> None:
    base = {
        "market_breadth_6": 0.2,
        "high_vol_regime_proxy": 0.0,
        "low_vol_regime_proxy": 0.0,
        "ret_1": 0.001,
        "ret_3": -0.004,
        "ret_12": -0.012,
        "lower_wick_fraction": 0.2,
        "upper_wick_fraction": 0.2,
        "volume_ratio_6_24": 0.9,
    }
    context = {
        "1h_ret_3": -0.02,
        "1h_trend_stack_long": 0.0,
        "1h_trend_stack_short": 1.0,
        "15m_volatility_ratio_6_24": 1.0,
        "1h_chop_12": 0.4,
        "1h_trend_strength_12": 2.0,
    }
    regime = classify_regime_axes(base, context)
    assert regime["direction"] == "BEARISH"
    assert regime["phase"] == "PULLBACK"
    assert (
        directional_role(DirectionalSide.SHORT, regime["direction"])
        is DirectionalRole.PRIMARY_TREND
    )
    assert (
        directional_role(DirectionalSide.LONG, regime["direction"])
        is DirectionalRole.TACTICAL_COUNTERTREND
    )


def test_probabilistic_router_inputs_are_causal_cross_sectional_context() -> None:
    features = {
        symbol: {
            "ret_1": offset * 0.001,
            "ret_3": offset * 0.002,
            "ret_12": offset * 0.003,
            "atr_12": 0.01,
            "volume_ratio_6_24": 1.0,
            "trend_strength_12": 2.0,
            "high_vol_regime_proxy": 0.0,
            "low_vol_regime_proxy": 1.0,
        }
        for offset, symbol in enumerate(("BTCUSDT", "ETHUSDT", "SOLUSDT"), start=1)
    }
    vector = regime_router_feature_vector(features)
    assert len(vector) == 16
    assert vector[0] == 0.001
    assert vector[6] == 0.002
    assert vector[9] == 1.0


def test_realized_global_regime_requires_btc_and_cross_section_agreement() -> None:
    assert (
        realized_global_regime(
            {"BTCUSDT": 0.01, "ETHUSDT": 0.02, "SOLUSDT": 0.03},
            btc_direction_threshold_fraction=0.0025,
            cross_section_breadth_threshold=0.55,
        )
        == "BULLISH"
    )
    assert (
        realized_global_regime(
            {"BTCUSDT": -0.01, "ETHUSDT": -0.02, "SOLUSDT": 0.001},
            btc_direction_threshold_fraction=0.0025,
            cross_section_breadth_threshold=0.55,
        )
        == "BEARISH"
    )
    assert (
        realized_global_regime(
            {"BTCUSDT": 0.01, "ETHUSDT": -0.02, "SOLUSDT": -0.01},
            btc_direction_threshold_fraction=0.0025,
            cross_section_breadth_threshold=0.55,
        )
        == "NEUTRAL"
    )


def test_directional_interactions_mirror_market_evidence() -> None:
    base = {
        "ret_1": -0.01,
        "ret_3": -0.02,
        "ret_12": -0.03,
        "market_direction_6": -0.01,
        "market_breadth_6": 0.2,
        "volume_ratio_6_24": 1.5,
        "close_position_in_range": 0.2,
        "lower_wick_fraction": 0.1,
        "upper_wick_fraction": 0.4,
        "trend_stack_long": 0.0,
        "trend_stack_short": 1.0,
    }
    long = directional_interactions(base, DirectionalSide.LONG)
    short = directional_interactions(base, DirectionalSide.SHORT)
    assert long[0] < 0 < short[0]
    assert long[4] == 0.2
    assert short[4] == 0.8
    assert short[-1] == 1.0


def test_regime_aware_vector_has_stable_shared_schema() -> None:
    base = {
        "ret_1": -0.01,
        "ret_3": -0.02,
        "ret_12": -0.03,
        "market_direction_6": -0.01,
        "market_breadth_6": 0.2,
        "volume_ratio_6_24": 1.5,
        "close_position_in_range": 0.2,
        "lower_wick_fraction": 0.1,
        "upper_wick_fraction": 0.4,
        "trend_stack_long": 0.0,
        "trend_stack_short": 1.0,
    }
    vector = regime_aware_feature_vector(
        multitimeframe_features=(0.0,) * len(LONG_V21_FEATURE_NAMES),
        base_features=base,
        side=DirectionalSide.SHORT,
        regime={
            "direction": "BEARISH",
            "volatility": "NORMAL",
            "structure": "TREND",
            "phase": "CONTINUATION",
        },
    )
    assert len(vector) == len(LONG_V21_FEATURE_NAMES) + 26
    assert all(isinstance(value, float) for value in vector)


def test_exit_eye_closes_profitable_long_on_opposite_signal() -> None:
    history = tuple(candle(index - 15, 100, 100.2, 99.8, 100) for index in range(15))
    future = (
        candle(1, 100, 101.5, 99.9, 101.3),
        candle(2, 101.3, 101.4, 100.7, 100.9),
    )
    observations = (
        CommitteeObservation("LONG", 2, 0, 1),
        CommitteeObservation("SHORT", 0, 2, 1),
    )
    result = replay_full_lifecycle(
        side=DirectionalSide.LONG,
        history=history,
        future=future,
        observations=observations,
        protection=TsProtectionConfig(
            leverage=10,
            hard_stop_roe=-0.40,
            take_profit_roe=0.50,
            break_even_trigger_roe=0.20,
            trailing_activation_roe=0.30,
            use_atr_trailing=False,
        ),
        exit_eye=exit_eye(),
    )
    assert result["exit_eye_close_bar"] == 2
    assert result["exit_eye_close_reason"] == "EXIT_EYE_OPPOSITE_SIGNAL"
    assert result["full_lifecycle_worst_net_return"] > 0


def test_exit_eye_never_closes_losing_trade() -> None:
    history = tuple(candle(index - 15, 100, 100.2, 99.8, 100) for index in range(15))
    future = (
        candle(1, 100, 100.1, 99.0, 99.2),
        candle(2, 99.2, 99.4, 98.8, 99.0),
    )
    observations = (
        CommitteeObservation("SHORT", 0, 3, 0),
        CommitteeObservation("SHORT", 0, 3, 0),
    )
    result = replay_full_lifecycle(
        side=DirectionalSide.LONG,
        history=history,
        future=future,
        observations=observations,
        protection=TsProtectionConfig(leverage=10, use_atr_trailing=False),
        exit_eye=exit_eye(),
    )
    assert result["exit_eye_close_bar"] is None
    assert result["full_lifecycle_worst_net_return"] < 0
