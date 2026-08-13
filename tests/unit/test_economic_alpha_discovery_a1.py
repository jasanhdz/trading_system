import numpy as np
import pandas as pd

from aegis.research.economic_alpha_discovery_a1 import (
    RobustScale,
    aggregate_completed_15m,
    cross_sectional_winners,
    daily_space,
    deterministic_random_symbol,
    side_components,
)


def test_aggregate_rejects_incomplete_fifteen_minute_bar():
    minute = pd.DataFrame(
        {
            "open_time": np.arange(29, dtype=np.int64) * 60_000,
            "open": 100.0,
            "high": 101.0,
            "low": 99.0,
            "close": 100.5,
            "quote_volume": 10.0,
            "taker_buy_quote": 6.0,
            "spot_close": 100.4,
        }
    )
    mark = minute[["open_time"]].assign(mark_close=100.45)
    result = aggregate_completed_15m(minute, mark)
    assert result["timestamp_ms"].tolist() == [0]
    assert result.iloc[0]["state_close_ms"] == 899_999


def test_side_components_are_directionally_symmetric():
    panel = pd.DataFrame(
        {
            "return_4h": [0.04], "return_24h": [0.08],
            "volume_persistence_1h": [1.2], "taker_flow_1h": [0.3],
            "prior_taker_flow_1h": [-0.1], "breakout_acceptance_long": [0.5],
            "breakout_acceptance_short": [-0.5], "relative_strength_btc_4h": [0.02],
            "extension_z_24h": [2.0], "return_1h": [0.01],
            "btc_return_4h": [0.02], "wick_rejection_long": [0.01],
            "wick_rejection_short": [-0.01], "basis_z_7d": [-2.0],
            "funding_rate": [-0.0001], "basis_convergence_1h": [0.001],
        }
    )
    long = side_components(panel, "LONG")
    short = side_components(panel, "SHORT")
    assert long.iloc[0]["trend_return_4h"] == -short.iloc[0]["trend_return_4h"]
    assert long.iloc[0]["carry_funding"] == -short.iloc[0]["carry_funding"]
    assert long.iloc[0]["reversal_extension"] == -short.iloc[0]["reversal_extension"]


def test_cross_sectional_winner_and_daily_spacing_are_deterministic():
    rows = pd.DataFrame(
        {
            "timestamp_ms": [0, 0, 900_000, 86_400_000],
            "symbol": ["BTCUSDT", "ETHUSDT", "ETHUSDT", "ETHUSDT"],
            "score": [1.0, 2.0, 3.0, 4.0],
        }
    )
    winners = cross_sectional_winners(rows)
    assert winners["symbol"].tolist() == ["ETHUSDT", "ETHUSDT", "ETHUSDT"]
    spaced = daily_space(winners)
    assert spaced["timestamp_ms"].tolist() == [0, 86_400_000]


def test_robust_scale_and_random_control_are_reproducible():
    scale = RobustScale(median=2.0, iqr=2.0, lower=1.0, upper=3.0)
    assert scale.apply(pd.Series([0.0, 2.0, 4.0])).tolist() == [-0.5, 0.0, 0.5]
    rows = pd.DataFrame({"symbol": ["ETHUSDT", "BTCUSDT", "ADAUSDT"]})
    first = deterministic_random_symbol(rows, "frozen-event")
    assert first == deterministic_random_symbol(rows, "frozen-event")
