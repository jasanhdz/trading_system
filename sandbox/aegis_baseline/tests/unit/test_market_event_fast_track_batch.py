import pandas as pd
import pytest

from aegis.research.market_event_fast_track_batch import (
    DISCOVERY_END_MS,
    VALIDATION_END_MS,
    bootstrap_expectancy,
    build_hourly_regime,
    collapse_events,
    evaluate_events,
    partition_name,
    summarize_batch,
)


def test_partitions_are_frozen_disjoint_and_pseudo_is_not_fresh_holdout() -> None:
    assert partition_name(DISCOVERY_END_MS - 1) == "DISCOVERY"
    assert partition_name(DISCOVERY_END_MS) == "VALIDATION"
    assert partition_name(VALIDATION_END_MS) == "PSEUDO_HOLDOUT"


def test_event_collapse_applies_symbol_spacing_then_market_cluster() -> None:
    frame = pd.DataFrame(
        [
            {
                "pattern": "P",
                "side": "LONG",
                "symbol": "ADAUSDT",
                "timestamp_ms": 1_000_000,
            },
            {
                "pattern": "P",
                "side": "LONG",
                "symbol": "BTCUSDT",
                "timestamp_ms": 1_300_000,
            },
            {
                "pattern": "P",
                "side": "LONG",
                "symbol": "ADAUSDT",
                "timestamp_ms": 5_000_000,
            },
        ]
    )
    result = collapse_events(frame)
    assert list(result["symbol"]) == ["ADAUSDT", "ADAUSDT"]


def test_batch_summary_reports_net_economics_and_concentration() -> None:
    frame = pd.DataFrame(
        [
            {
                "net_return_fraction": 0.02,
                "mae_fraction": 0.01,
                "mfe_fraction": 0.03,
                "symbol": "ADAUSDT",
            },
            {
                "net_return_fraction": -0.01,
                "mae_fraction": 0.02,
                "mfe_fraction": 0.01,
                "symbol": "BTCUSDT",
            },
        ]
    )
    result = summarize_batch(frame)
    assert result.expectancy == pytest.approx(0.005)
    assert result.profit_factor == pytest.approx(2.0)
    assert result.symbol_share_maximum == 0.5


def test_hourly_regime_uses_completed_bars_without_duplicate_timestamp() -> None:
    rows = 60 * 24 * 10
    frame = pd.DataFrame(
        {
            "open_time": [DISCOVERY_END_MS + minute * 60_000 for minute in range(rows)],
            "close": [100.0 + minute / 10_000 for minute in range(rows)],
            "quote_volume": [1_000.0 + minute for minute in range(rows)],
        }
    )
    result = build_hourly_regime(frame)
    assert result["timestamp"].is_unique
    assert result["timestamp_ms"].is_monotonic_increasing
    assert result["timestamp_ms"].min() >= DISCOVERY_END_MS + 24 * 60 * 60_000


def test_day_block_bootstrap_is_reproducible() -> None:
    frame = pd.DataFrame(
        [
            {
                "timestamp_ms": DISCOVERY_END_MS + index * 86_400_000,
                "net_return_fraction": value,
            }
            for index, value in enumerate((0.01, -0.005, 0.02, -0.002, 0.015))
        ]
    )
    first = bootstrap_expectancy(frame, repetitions=100)
    second = bootstrap_expectancy(frame, repetitions=100)
    assert first == second
    assert first["expectancy_median"] > 0.0


def test_vectorized_event_evaluation_uses_next_bar_and_exact_path() -> None:
    start = DISCOVERY_END_MS
    frame = pd.DataFrame(
        {
            "open_time": [start + minute * 60_000 for minute in range(4)],
            "open": [100.0, 101.0, 102.0, 103.0],
            "high": [101.0, 104.0, 103.0, 105.0],
            "low": [99.0, 100.0, 98.0, 102.0],
            "close": [100.5, 102.0, 100.0, 104.0],
        }
    )
    events = pd.DataFrame(
        [
            {
                "pattern": "P",
                "side": "LONG",
                "symbol": "ADAUSDT",
                "timestamp_ms": start + 59_999,
                "regime_direction": "BULL",
                "regime_volatility": "NORMAL",
            }
        ]
    )
    result = evaluate_events(events, {"ADAUSDT": frame}, horizon=3).iloc[0]
    assert result["entry_timestamp_ms"] == start + 60_000
    assert result["gross_return_fraction"] == pytest.approx((104.0 - 101.0) / 101.0)
    assert result["mae_fraction"] == pytest.approx((101.0 - 98.0) / 101.0)
    assert result["mfe_fraction"] == pytest.approx((105.0 - 101.0) / 101.0)
