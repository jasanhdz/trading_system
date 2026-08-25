from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pandas.testing as pdt

from aegis_ephemeral_regime_w11.data import (
    aggregate_complete_5m_bars,
    build_data_panel,
    load_frozen_config,
)


PROJECT = Path(__file__).resolve().parents[1]


def candles(symbol: str, periods: int = 240) -> pd.DataFrame:
    time = pd.date_range("2023-05-01", periods=periods, freq="1min", tz="UTC")
    offset = {"BTCUSDT": 0.0, "ETHUSDT": 20.0, "ADAUSDT": 40.0}[symbol]
    close = 100.0 + offset + np.arange(periods) * 0.02 + np.sin(np.arange(periods) / 9)
    open_ = close - 0.01
    return pd.DataFrame(
        {
            "open_time_ms": time.astype("int64") // 1_000_000,
            "open": open_,
            "high": close + 0.1,
            "low": open_ - 0.1,
            "close": close,
            "volume": 10.0 + np.arange(periods) % 7,
            "close_time_ms": time.astype("int64") // 1_000_000 + 59_999,
            "quote_volume": (10.0 + np.arange(periods) % 7) * close,
            "trade_count": np.full(periods, 5),
            "taker_buy_volume": 4.0 + np.arange(periods) % 5,
            "taker_buy_quote_volume": (4.0 + np.arange(periods) % 5) * close,
        }
    )


def synthetic_config() -> dict:
    config = load_frozen_config(PROJECT / "config" / "w11_frozen.json")
    config["source"]["symbols"] = ["ADAUSDT", "BTCUSDT", "ETHUSDT"]
    return config


def test_future_mutation_does_not_change_past_features() -> None:
    raw = {symbol: candles(symbol) for symbol in synthetic_config()["source"]["symbols"]}
    original = build_data_panel(raw, synthetic_config())
    cutoff = pd.Timestamp("2023-05-01 02:30", tz="UTC")
    mutated = {symbol: frame.copy() for symbol, frame in raw.items()}
    for frame in mutated.values():
        future = frame["open_time_ms"] >= int(cutoff.timestamp() * 1000)
        frame.loc[future, ["open", "high", "low", "close"]] *= 3.0
        frame.loc[future, ["volume", "taker_buy_volume"]] *= 7.0
    changed = build_data_panel(mutated, synthetic_config())
    names = original.attrs["feature_names"]
    pdt.assert_frame_equal(original.loc[:cutoff, names], changed.loc[:cutoff, names])


def test_target_enters_at_next_bar_open_and_exits_at_horizon_close() -> None:
    raw = {symbol: candles(symbol) for symbol in synthetic_config()["source"]["symbols"]}
    bars = aggregate_complete_5m_bars(raw)
    panel = build_data_panel(raw, synthetic_config())
    decision = pd.Timestamp("2023-05-01 01:00", tz="UTC")
    symbol_bars = bars[bars["symbol"].eq("ADAUSDT")].set_index("close_time")
    entry = symbol_bars.loc[decision + pd.Timedelta(minutes=5), "open"]
    expected_5 = (symbol_bars.loc[decision + pd.Timedelta(minutes=5), "close"] / entry - 1) * 10_000
    expected_15 = (symbol_bars.loc[decision + pd.Timedelta(minutes=15), "close"] / entry - 1) * 10_000
    row = panel.loc[(decision, "ADAUSDT")]
    assert np.isclose(row["gross_target_5m_bps"], expected_5)
    assert np.isclose(row["gross_target_15m_bps"], expected_15)
    assert row["outcome_available_at"] == decision + pd.Timedelta(minutes=60)


def test_gap_rejects_incomplete_five_minute_bar() -> None:
    raw = candles("BTCUSDT", periods=20).drop(index=7).reset_index(drop=True)
    bars = aggregate_complete_5m_bars({"BTCUSDT": raw})
    rejected_close = pd.Timestamp("2023-05-01 00:10", tz="UTC")
    assert rejected_close not in set(bars["close_time"])
    assert list(bars["close_time"]) == [
        pd.Timestamp("2023-05-01 00:05", tz="UTC"),
        pd.Timestamp("2023-05-01 00:15", tz="UTC"),
        pd.Timestamp("2023-05-01 00:20", tz="UTC"),
    ]


def test_feature_names_exactly_match_frozen_config() -> None:
    config_path = PROJECT / "config" / "w11_frozen.json"
    with config_path.open(encoding="utf-8") as handle:
        expected = json.load(handle)["features"]["names"]
    raw = {symbol: candles(symbol) for symbol in synthetic_config()["source"]["symbols"]}
    panel = build_data_panel(raw, synthetic_config())
    assert panel.attrs["feature_names"] == expected
    assert [column for column in panel.columns if column in expected] == expected
    assert len(expected) == 20
    assert (panel["feature_available_at"] <= panel.index.get_level_values("decision_at")).all()
