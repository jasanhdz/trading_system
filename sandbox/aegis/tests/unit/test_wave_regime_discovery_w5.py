from __future__ import annotations

import pandas as pd
import pytest

from aegis.research.wave_regime_discovery_w5 import (
    benjamini_hochberg,
    correlation_cluster_id,
    economic_summary,
    resolve_wave_path,
    stable_wave_id,
)


def test_wave_identity_is_stable_and_side_specific() -> None:
    value = stable_wave_id("BTCUSDT", "LONG", 100)
    assert value == stable_wave_id("BTCUSDT", "LONG", 100)
    assert value != stable_wave_id("BTCUSDT", "SHORT", 100)


def test_good_wave_requires_clean_favorable_path() -> None:
    result = resolve_wave_path(
        entry=100, atr=1, direction=1,
        highs=[100.2, 100.6], lows=[99.95, 100.1], closes=[100.15, 100.5],
    )
    assert result["barrier_outcome"] == "FAVORABLE"
    assert result["wave_label"] == "GOOD_WAVE"


def test_adverse_barrier_wins_same_minute() -> None:
    result = resolve_wave_path(
        entry=100, atr=1, direction=-1,
        highs=[100.3], lows=[99.4], closes=[99.8],
    )
    assert result["barrier_outcome"] == "ADVERSE"
    assert result["wave_label"] == "BAD_WAVE"


def test_cluster_is_fifteen_minutes() -> None:
    assert correlation_cluster_id(899_999) == 0
    assert correlation_cluster_id(900_000) == 1


def test_benjamini_hochberg_controls_family() -> None:
    result = benjamini_hochberg({"a": 0.001, "b": 0.02, "c": 0.9})
    assert result == {"a": True, "b": True, "c": False}


def test_economic_summary_reports_cluster_and_symbol_concentration() -> None:
    frame = pd.DataFrame({
        "net_return_bps": [2.0, -1.0], "correlation_cluster_id": [1, 1],
        "wave_label": ["GOOD_WAVE", "BAD_WAVE"], "mfe_atr": [1, 0],
        "mae_atr": [0, 1], "symbol": ["BTCUSDT", "ETHUSDT"],
    })
    result = economic_summary(frame)
    assert result["net_expectancy_bps"] == pytest.approx(0.5)
    assert result["independent_clusters"] == 1
    assert result["maximum_symbol_share"] == pytest.approx(0.5)
