from datetime import datetime, timedelta, timezone

import pytest

from aegis.research.regime_v2 import (
    DirectionRegime,
    FactorizedRegimeAnalyzer,
    RegimeV2Observation,
    RegimeV2Settings,
    StructureRegime,
    VolatilityRegime,
)


def _settings(minimum_state_bars: int = 1) -> RegimeV2Settings:
    return RegimeV2Settings(
        schema_version="aegis-regime-v2-research-settings-v1",
        history_window=8,
        minimum_history=3,
        low_volatility_quantile=0.25,
        high_volatility_quantile=0.75,
        trend_enter_fraction=0.003,
        trend_exit_fraction=0.001,
        trend_strength_enter=0.8,
        trend_strength_exit=0.5,
        chop_enter_fraction=0.70,
        chop_exit_fraction=0.60,
        high_expansion_ratio=0.50,
        low_expansion_ratio=-0.20,
        minimum_state_bars=minimum_state_bars,
    )


def _observation(
    index: int,
    *,
    direction: float = -0.01,
    range_mean: float = 0.01,
    expansion: float = 0.0,
    chop: float = 0.2,
    strength: float = 1.2,
) -> RegimeV2Observation:
    return RegimeV2Observation(
        symbol="ETHUSDT",
        timestamp=datetime(2026, 7, 24, tzinfo=timezone.utc) + timedelta(minutes=5 * index),
        market_direction_6=direction,
        range_mean_24=range_mean,
        range_expansion=expansion,
        chop_12=chop,
        trend_strength_12=strength,
    )


def test_regime_v2_preserves_direction_structure_and_volatility_axes() -> None:
    analyzer = FactorizedRegimeAnalyzer(_settings())
    analyzer.observe(_observation(0, range_mean=0.005))
    analyzer.observe(_observation(1, range_mean=0.01))
    result = analyzer.observe(_observation(2, range_mean=0.03, expansion=0.8))

    assert result.evidence_ready
    assert result.direction is DirectionRegime.BEARISH
    assert result.structure is StructureRegime.TREND
    assert result.volatility is VolatilityRegime.HIGH
    assert result.short_context


def test_regime_v2_uses_hysteresis_and_minimum_state_duration() -> None:
    analyzer = FactorizedRegimeAnalyzer(_settings(minimum_state_bars=2))
    first = analyzer.observe(_observation(0))
    assert first.direction is DirectionRegime.BEARISH

    pending = analyzer.observe(_observation(1, direction=0.01))
    assert pending.direction is DirectionRegime.BEARISH
    changed = analyzer.observe(_observation(2, direction=0.01))
    assert changed.direction is DirectionRegime.BULLISH


def test_regime_v2_does_not_call_range_a_direction() -> None:
    analyzer = FactorizedRegimeAnalyzer(_settings())
    result = analyzer.observe(_observation(0, direction=-0.01, chop=0.9))

    assert result.direction is DirectionRegime.BEARISH
    assert result.structure is StructureRegime.RANGE


def test_regime_v2_rejects_non_chronological_symbol_updates() -> None:
    analyzer = FactorizedRegimeAnalyzer(_settings())
    analyzer.observe(_observation(1))
    with pytest.raises(ValueError, match="strictly chronological"):
        analyzer.observe(_observation(1))

