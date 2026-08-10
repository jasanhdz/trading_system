from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest
import yaml

from aegis.research.long_entry_v21_shadow import LONG_V21_FEATURE_NAMES
from aegis.research.long_entry_v22_shadow import (
    LONG_V22_FEATURE_NAMES,
    long_v22_feature_vector,
    select_cross_section,
    specialist_committee_score,
)


def base_vector() -> tuple[float, ...]:
    values = {name: 0.0 for name in LONG_V21_FEATURE_NAMES}
    values.update(
        {
            "atr_12": 0.005,
            "volume_ratio_6_24": 1.4,
            "close_position_in_range": 0.8,
            "body_to_range": 0.6,
            "market_breadth_6": 0.7,
            "trend_stack_long": 1.0,
            "15m_trend_stack_long": 1.0,
            "1h_trend_stack_long": 1.0,
            "1h_chop_12": 0.4,
        }
    )
    return tuple(values[name] for name in LONG_V21_FEATURE_NAMES)


def test_v22_features_are_causal_finite_and_regime_explicit() -> None:
    result = long_v22_feature_vector(
        base_vector(),
        {"direction": "BULLISH", "volatility": "NORMAL", "structure": "TREND"},
    )
    assert len(result) == len(LONG_V22_FEATURE_NAMES)
    assert sum(result[-27:]) == 1.0
    assert not any(
        token in name.lower()
        for name in LONG_V22_FEATURE_NAMES
        for token in ("future", "outcome", "pnl", "mfe", "mae")
    )


def test_specialist_score_preserves_each_probability_without_votes() -> None:
    assert specialist_committee_score(0.8, 0.5, 0.25) == pytest.approx(0.3)
    with pytest.raises(ValueError):
        specialist_committee_score(1.1, 0.5, 0.25)


def test_cross_section_selects_best_symbols_at_same_timestamp() -> None:
    timestamp = datetime(2026, 8, 9, tzinfo=timezone.utc)
    rows = [
        {
            "timestamp": timestamp,
            "symbol": symbol,
            "committee_score": score,
            "path_risk_probability": risk,
            "timing_probability": timing,
        }
        for symbol, score, risk, timing in (
            ("BTCUSDT", 0.30, 0.20, 0.70),
            ("ETHUSDT", 0.40, 0.30, 0.80),
            ("SOLUSDT", 0.35, 0.10, 0.60),
        )
    ]
    selected = select_cross_section(
        rows,
        minimum_score=0.25,
        maximum_path_risk=0.35,
        maximum_selected_per_timestamp=2,
    )
    assert selected == (False, True, True)


def test_v22_preregistration_is_shadow_only_and_excludes_unaligned_data() -> None:
    root = Path(__file__).parents[2]
    payload = yaml.safe_load(
        (root / "config/experiments/aegis_long_entry_v22_shadow.yaml").read_text()
    )
    assert payload["mode"] == "SHADOW"
    assert payload["selection_effect"] == "NONE"
    assert payload["automatic_live_promotion"] is False
    assert payload["models"]["specialists"] == [
        "DIRECTION",
        "ENTRY_TIMING",
        "PATH_RISK",
    ]
    availability = payload["features"]["availability_audit"]
    assert availability["ohlcv_causal_interactions"] == "INCLUDED"
    assert all(
        value.startswith("EXCLUDED_")
        for key, value in availability.items()
        if key != "ohlcv_causal_interactions"
    )
    assert payload["deployment"]["live_runtime"] == "PROHIBITED"
    assert payload["deployment"]["exchange_mutations"] == 0
