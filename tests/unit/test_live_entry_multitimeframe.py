import pandas as pd

from aegis.research.live_entry_multitimeframe import add_directional_context, aggregate_klines, attach_features


def candles(rows: int = 240) -> pd.DataFrame:
    return pd.DataFrame({
        "open_time_ms": [1_700_000_100_000 + index * 60_000 for index in range(rows)],
        "open": [100 + index * 0.01 for index in range(rows)],
        "high": [100.1 + index * 0.01 for index in range(rows)],
        "low": [99.9 + index * 0.01 for index in range(rows)],
        "close": [100.05 + index * 0.01 for index in range(rows)],
        "volume": [10 + index % 5 for index in range(rows)],
        "taker_buy_volume": [5 + index % 3 for index in range(rows)],
    })


def test_aggregation_requires_complete_bars() -> None:
    result = aggregate_klines(candles(11), 5)
    assert len(result) == 2
    assert result["bar_count"].eq(5).all()


def test_features_use_last_closed_bar_only() -> None:
    source = candles()
    first_open = pd.to_datetime(source.iloc[0]["open_time_ms"], unit="ms", utc=True)
    entry = first_open + pd.Timedelta(minutes=201, seconds=30)
    entries = pd.DataFrame({"symbol": ["BTCUSDT"], "opened_at": [entry.isoformat()]})
    result = attach_features(entries, {"BTCUSDT": source}, [5])
    expected_close = float(source.iloc[199]["close"])
    prior = float(source.iloc[194]["close"])
    assert abs(result.iloc[0]["tf5m__return_1_bps"] - (expected_close / prior - 1) * 10_000) < 1e-9


def test_directional_context_is_symmetric() -> None:
    base = pd.DataFrame({
        "side": ["LONG", "SHORT"], "tf5m__return_1_bps": [10.0, 10.0],
        "tf5m__return_3_bps": [10.0, 10.0], "tf5m__return_6_bps": [10.0, 10.0],
        "tf5m__ema7_extension_atr": [1.0, 1.0], "tf5m__ema25_extension_atr": [1.0, 1.0],
        "tf5m__ema99_extension_atr": [1.0, 1.0], "tf5m__ema7_slope_atr": [1.0, 1.0],
        "tf5m__ema25_slope_atr": [1.0, 1.0], "tf5m__trend_age": [2.0, 2.0],
        "tf5m__prior_move_6_atr": [1.0, 1.0], "tf5m__taker_imbalance": [0.5, 0.5],
        "tf5m__rsi6": [60.0, 60.0], "tf5m__rsi12": [60.0, 60.0], "tf5m__rsi24": [60.0, 60.0],
        "tf5m__distance_recent_high_atr": [3.0, 3.0], "tf5m__distance_recent_low_atr": [2.0, 2.0],
        "tf5m__breakout_up": [1.0, 1.0], "tf5m__breakout_down": [0.0, 0.0], "tf5m__clv": [0.8, 0.8],
    })
    result = add_directional_context(base, [5])
    assert result.loc[0, "dir5m__return_1_bps"] == 10.0
    assert result.loc[1, "dir5m__return_1_bps"] == -10.0
    assert result.loc[0, "dir5m__favorable_space_atr"] == 3.0
    assert result.loc[1, "dir5m__favorable_space_atr"] == 2.0
