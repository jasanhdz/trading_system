from __future__ import annotations

from build_feature_information_v14_dataset import _database_symbol, _timestamp


def test_database_symbol_and_timestamp_are_canonical() -> None:
    assert _database_symbol("BTCUSDT") == "BTC/USDT"
    assert _timestamp("2026-01-01T00:00:00+00:00").utcoffset().total_seconds() == 0
