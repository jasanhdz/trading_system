from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
import yaml

from aegis.domain import Candle
from aegis.research.long_entry_v4_hybrid_shadow import (
    EXIT_STATE_FEATURE_NAMES,
    LongTechnicalSetup,
    clean_entry_label,
    exit_now_preferred_label,
    exit_state_feature_vector,
    hybrid_score,
    technical_setup,
)


def _candle(index: int, close: float) -> Candle:
    start = datetime(2026, 1, 1, tzinfo=timezone.utc) + timedelta(minutes=5 * index)
    return Candle(
        open_time=start,
        close_time=start + timedelta(minutes=5),
        open=close - 0.1,
        high=close + 0.2,
        low=close - 0.2,
        close=close,
        volume=1000.0,
        is_closed=True,
        source="TEST",
    )


def test_technical_generator_keeps_three_setups_and_drops_expansion() -> None:
    assert technical_setup("TREND_PULLBACK_RECLAIM") is LongTechnicalSetup.TREND_CONTINUATION
    assert technical_setup("BREAKOUT_RETEST") is LongTechnicalSetup.BREAKOUT_RETEST
    assert technical_setup("CONFIRMED_REVERSAL") is LongTechnicalSetup.CAPITULATION_REVERSAL
    assert technical_setup("BREAKOUT_EXPANSION") is None


def test_clean_entry_requires_direction_mae_speed_and_underwater_quality() -> None:
    row = {
        "target_before_stop": True,
        "clean_fast_success": True,
        "mae_fraction": 0.002,
        "adverse_barrier_fraction": 0.003,
        "time_underwater_bars": 4,
    }
    assert clean_entry_label(row, horizon_bars=12) is True
    assert clean_entry_label({**row, "mae_fraction": 0.004}, horizon_bars=12) is False
    assert clean_entry_label({**row, "time_underwater_bars": 8}, horizon_bars=12) is False


def test_exit_features_use_only_observed_closed_bars() -> None:
    observed = tuple(_candle(index, 100.0 + index * 0.2) for index in range(3))
    features = exit_state_feature_vector(
        entry_price=100.0,
        observed=observed,
        horizon_bars=12,
        atr_fraction=0.005,
        round_trip_cost_fraction=0.001,
    )
    assert len(features) == len(EXIT_STATE_FEATURE_NAMES)
    assert features[0] == pytest.approx(0.25)


def test_exit_label_compares_current_exit_to_frozen_continue_replay() -> None:
    assert exit_now_preferred_label(
        current_net_return=0.002, continue_worst_protected_net=-0.001
    )
    assert not exit_now_preferred_label(
        current_net_return=-0.002, continue_worst_protected_net=0.001
    )


def test_hybrid_score_does_not_fabricate_specialist_values() -> None:
    assert hybrid_score(0.8, 0.5, 0.25, 0.2) == pytest.approx(0.24)
    with pytest.raises(ValueError):
        hybrid_score(0.8, 1.1, 0.25, 0.2)


def test_v4_is_shadow_only_and_has_no_trade_quota() -> None:
    root = Path(__file__).parents[2]
    payload = yaml.safe_load(
        (root / "config/experiments/aegis_long_entry_v4_shadow.yaml").read_text()
    )
    assert payload["mode"] == "SHADOW"
    assert payload["selection_effect"] == "NONE"
    assert payload["automatic_live_promotion"] is False
    assert payload["tournament"]["no_trade_quota"] is True
    assert payload["deployment"]["live_runtime"] == "PROHIBITED"
    assert payload["deployment"]["exit_runtime_authority"] == "NONE"
    assert payload["deployment"]["exchange_mutations"] == 0
