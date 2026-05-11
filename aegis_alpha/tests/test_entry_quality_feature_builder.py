from __future__ import annotations

import numpy as np
import pandas as pd

from aegis_alpha.entry_quality.runtime_feature_cache import (
    add_base_features,
    add_mtf_features,
    build_features_from_candles,
)


def synthetic_candles(rows: int = 240) -> pd.DataFrame:
    index = pd.date_range("2026-01-01", periods=rows, freq="5min")
    base = 100.0 + np.linspace(0, 5, rows)
    close = base + np.sin(np.arange(rows) / 8.0)
    open_ = close - 0.05
    high = np.maximum(open_, close) + 0.2
    low = np.minimum(open_, close) - 0.2
    volume = 1000.0 + np.arange(rows) % 37
    return pd.DataFrame(
        {
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
        },
        index=index,
    )


def test_builds_5m_features_from_synthetic_candles() -> None:
    result = build_features_from_candles("BTCUSDT", synthetic_candles())
    assert result.values["ret_1"] == result.values["ret_1"]
    assert result.values["momentum_12"] == result.values["momentum_12"]


def test_computes_ema_features() -> None:
    values = build_features_from_candles("BTCUSDT", synthetic_candles()).values
    assert values["ema_9"] > 0
    assert values["ema_21"] > 0
    assert values["ema_50"] > 0
    assert np.isfinite(values["price_to_ema_9"])


def test_computes_atr_and_vol_features() -> None:
    values = build_features_from_candles("BTCUSDT", synthetic_candles()).values
    assert values["atr_14"] > 0
    assert values["atr_pct"] > 0
    assert np.isfinite(values["realized_vol_12"])
    assert np.isfinite(values["realized_vol_36"])


def test_computes_candle_body_and_wicks() -> None:
    values = build_features_from_candles("BTCUSDT", synthetic_candles()).values
    assert np.isfinite(values["candle_body_pct"])
    assert 0 <= values["upper_wick_pct"] <= 1
    assert 0 <= values["lower_wick_pct"] <= 1
    assert values["green_candle_count_3"] >= 0
    assert values["red_candle_count_3"] >= 0


def test_computes_15m_mtf_features_from_5m() -> None:
    values = build_features_from_candles("BTCUSDT", synthetic_candles()).values
    assert np.isfinite(values["mtf_15m_ret_1"])
    assert np.isfinite(values["mtf_15m_ema_9_slope"])
    assert values["mtf_15m_trend_direction"] in {-1.0, 0.0, 1.0}


def test_computes_1h_mtf_features_from_5m() -> None:
    values = build_features_from_candles("BTCUSDT", synthetic_candles()).values
    assert np.isfinite(values["mtf_1h_ret_1"])
    assert np.isfinite(values["mtf_1h_ema_9_slope"])
    assert values["mtf_1h_trend_direction"] in {-1.0, 0.0, 1.0}


def test_no_future_leakage_in_resample() -> None:
    candles = synthetic_candles()
    partial = candles.iloc[:-1]
    full_features = add_mtf_features(add_base_features(candles)[0], candles)
    partial_features = add_mtf_features(add_base_features(partial)[0], partial)
    common_ts = partial_features.index[-1]
    assert full_features.loc[common_ts, "mtf_1h_ret_1"] == partial_features.loc[common_ts, "mtf_1h_ret_1"]


def test_aligns_expected_runtime_columns() -> None:
    values = build_features_from_candles("BTCUSDT", synthetic_candles()).values
    expected = {
        "ret_1",
        "ema_9",
        "atr_14",
        "quote_volume",
        "mtf_15m_ret_1",
        "mtf_1h_ret_1",
        "long_mtf_agreement",
        "short_mtf_agreement",
    }
    assert expected.issubset(values.keys())


def test_missing_candles_returns_no_crash() -> None:
    result = build_features_from_candles("BTCUSDT", pd.DataFrame())
    assert result.values == {}
    assert "no_recent_candles" in result.warnings


def test_quote_volume_approximated_if_absent() -> None:
    result = build_features_from_candles("BTCUSDT", synthetic_candles())
    assert "quote_volume" in result.approximated_features
    assert result.values["quote_volume"] > 0


def test_cache_equivalent_inputs_have_same_timestamp() -> None:
    candles = synthetic_candles()
    first = build_features_from_candles("BTCUSDT", candles)
    second = build_features_from_candles("BTCUSDT", candles)
    assert first.feature_timestamp == second.feature_timestamp
    assert first.values["ret_1"] == second.values["ret_1"]
