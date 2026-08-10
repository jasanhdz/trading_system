from __future__ import annotations

import math
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
import yaml

from aegis.domain import Candle
from aegis.research.long_entry_v22_shadow import LONG_V22_FEATURE_NAMES
from aegis.research.long_entry_v3_shadow import (
    LONG_V3_FEATURE_NAMES,
    MICROSTRUCTURE_FEATURE_NAMES,
    HardNegativeType,
    LongCandidateFamily,
    MicrostructureBar,
    classify_hard_negative,
    classify_long_v3_candidate,
    long_v3_feature_vector,
    microstructure_feature_vector,
)


def _candles(count: int = 40) -> tuple[Candle, ...]:
    start = datetime(2026, 1, 1, tzinfo=timezone.utc)
    result = []
    for index in range(count):
        close = 100.0 + index * 0.1
        result.append(
            Candle(
                open_time=start + timedelta(minutes=5 * index),
                close_time=start + timedelta(minutes=5 * (index + 1)),
                open=close - 0.05,
                high=close + 0.1,
                low=close - 0.1,
                close=close,
                volume=1000.0,
                is_closed=True,
                source="TEST",
            )
        )
    return tuple(result)


def _candidate_config() -> dict[str, object]:
    root = Path(__file__).parents[2]
    config = yaml.safe_load(
        (root / "config/experiments/aegis_long_entry_v3_shadow.yaml").read_text()
    )
    return config["candidate_builder"]


def _base(**overrides: float) -> dict[str, float]:
    values = {
        "atr_12": 0.005,
        "distance_to_rolling_high_12": 0.0,
        "ret_1": 0.002,
        "ret_3": 0.006,
        "ret_12": 0.02,
        "close_position_in_range": 0.9,
        "body_to_range": 0.7,
        "volume_ratio_6_24": 1.5,
        "trend_stack_long": 1.0,
        "close_vs_ema_12": 0.001,
        "close_vs_ema_6": 0.001,
        "lower_wick_fraction": 0.3,
        "upper_wick_fraction": 0.1,
    }
    values.update(overrides)
    return values


def _context(**overrides: float) -> dict[str, float]:
    values = {"1h_trend_stack_long": 1.0, "15m_ret_3": 0.01}
    values.update(overrides)
    return values


def _micro(**overrides: float) -> dict[str, float]:
    values = {
        "taker_buy_ratio_3": 0.60,
        "taker_imbalance_acceleration_3_12": 0.05,
        "trade_intensity_ratio_3_24": 1.2,
    }
    values.update(overrides)
    return values


def test_microstructure_features_are_causal_finite_and_exact_length() -> None:
    bars = tuple(
        MicrostructureBar(
            quote_volume=1000.0 + index,
            trade_count=100 + index,
            taker_buy_base=55.0 + index / 100.0,
            base_volume=100.0,
        )
        for index in range(24)
    )
    vector = microstructure_feature_vector(
        bars,
        return_1=0.002,
        atr_fraction=0.005,
        funding_rate_last=0.0001,
        funding_rate_previous=0.00005,
    )
    assert len(vector) == len(MICROSTRUCTURE_FEATURE_NAMES)
    assert all(math.isfinite(value) for value in vector)
    assert vector[-1] == pytest.approx(0.00005)
    combined = long_v3_feature_vector(
        (0.0,) * len(LONG_V22_FEATURE_NAMES), vector
    )
    assert len(combined) == len(LONG_V3_FEATURE_NAMES)


def test_candidate_builder_accepts_convincing_breakout_and_rejects_ordinary_bar() -> None:
    accepted = classify_long_v3_candidate(
        base=_base(),
        context=_context(),
        micro=_micro(),
        history=_candles(),
        config=_candidate_config(),
    )
    assert accepted["family"] == LongCandidateFamily.BREAKOUT_EXPANSION.value
    assert accepted["is_candidate"] is True

    rejected = classify_long_v3_candidate(
        base=_base(
            distance_to_rolling_high_12=0.05,
            ret_1=-0.01,
            ret_3=-0.02,
            ret_12=-0.03,
            close_position_in_range=0.2,
            body_to_range=0.1,
            volume_ratio_6_24=0.7,
            trend_stack_long=0.0,
            close_vs_ema_6=-0.01,
            lower_wick_fraction=0.1,
            upper_wick_fraction=0.5,
        ),
        context=_context(**{"1h_trend_stack_long": 0.0, "15m_ret_3": -0.01}),
        micro=_micro(
            taker_buy_ratio_3=0.45,
            taker_imbalance_acceleration_3_12=-0.1,
            trade_intensity_ratio_3_24=0.7,
        ),
        history=_candles(),
        config=_candidate_config(),
    )
    assert rejected["family"] == LongCandidateFamily.NONE.value
    assert rejected["is_candidate"] is False


@pytest.mark.parametrize(
    ("family", "outcome", "expected"),
    [
        (
            LongCandidateFamily.BREAKOUT_EXPANSION.value,
            {
                "barrier_order": "ADVERSE_FIRST",
                "mae_fraction": 0.004,
                "atr_fraction": 0.005,
                "target_before_stop": False,
                "clean_fast_success": False,
            },
            HardNegativeType.FALSE_BREAKOUT,
        ),
        (
            LongCandidateFamily.CONFIRMED_REVERSAL.value,
            {
                "barrier_order": "ADVERSE_FIRST",
                "mae_fraction": 0.010,
                "atr_fraction": 0.005,
                "target_before_stop": False,
                "clean_fast_success": False,
            },
            HardNegativeType.FALLING_KNIFE,
        ),
        (
            LongCandidateFamily.TREND_PULLBACK_RECLAIM.value,
            {
                "barrier_order": "FAVORABLE_FIRST",
                "mae_fraction": 0.001,
                "atr_fraction": 0.005,
                "target_before_stop": True,
                "clean_fast_success": False,
            },
            HardNegativeType.LATE_RECOVERY,
        ),
    ],
)
def test_hard_negative_taxonomy(
    family: str, outcome: dict[str, object], expected: HardNegativeType
) -> None:
    assert classify_hard_negative(family, outcome) is expected


def test_v3_preregistration_is_shadow_only_and_public_get_only() -> None:
    root = Path(__file__).parents[2]
    payload = yaml.safe_load(
        (root / "config/experiments/aegis_long_entry_v3_shadow.yaml").read_text()
    )
    assert payload["mode"] == "SHADOW"
    assert payload["selection_effect"] == "NONE"
    assert payload["automatic_live_promotion"] is False
    assert payload["public_data_contract"]["method"] == "GET"
    assert payload["public_data_contract"]["authenticated"] is False
    assert payload["deployment"]["live_runtime"] == "PROHIBITED"
    assert payload["deployment"]["exchange_mutations"] == 0


def test_public_collector_has_no_credentials_or_mutating_methods() -> None:
    root = Path(__file__).parents[2]
    source = (root / "scripts/refresh_long_v3_public_microstructure.py").read_text()
    forbidden = (
        'method="POST"',
        'method="PUT"',
        'method="DELETE"',
        "BINANCE_API_KEY",
        "BINANCE_API_SECRET",
        "/fapi/v1/order",
    )
    assert not any(token in source for token in forbidden)
    assert 'method="GET"' in source
