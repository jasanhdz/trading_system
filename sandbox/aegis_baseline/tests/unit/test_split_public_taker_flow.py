from __future__ import annotations

import json
import sqlite3

import pytest

from aegis.features import CANONICAL_SYMBOLS
from aegis.research.split_public_taker_flow import split_public_flow_lookup


def test_split_public_flow_is_causal_complete_and_deterministic(tmp_path):
    entry = 10_000_000
    delta = tmp_path / "candles.jsonl"
    database = tmp_path / "micro.db"
    connection = sqlite3.connect(database)
    connection.execute(
        "CREATE TABLE kline_microstructure (symbol TEXT,open_time_ms INTEGER,"
        "quote_volume REAL,trade_count INTEGER,taker_buy_base REAL,"
        "taker_buy_quote REAL,PRIMARY KEY(symbol,open_time_ms))"
    )
    lines = []
    for symbol in CANONICAL_SYMBOLS:
        for offset in range(24, 0, -1):
            timestamp = entry - offset * 300_000
            lines.append(json.dumps({
                "symbol": symbol,
                "open_time_ms": timestamp,
                "volume": 100.0,
            }))
            connection.execute(
                "INSERT INTO kline_microstructure VALUES(?,?,?,?,?,?)",
                (symbol, timestamp, 1.0, 1, 60.0, 1.0),
            )
    connection.commit()
    connection.close()
    delta.write_text("\n".join(lines) + "\n", encoding="utf-8")
    first, inventory = split_public_flow_lookup(
        candle_delta=delta,
        microstructure_database=database,
        required_entry_timestamps_ms=[entry],
    )
    second, _ = split_public_flow_lookup(
        candle_delta=delta,
        microstructure_database=database,
        required_entry_timestamps_ms=[entry],
    )
    assert first == second
    assert len(first) == 11
    assert inventory["complete_eleven_symbol_timestamps"] == 1
    assert inventory["strict_pre_entry_only"] is True
    assert first[(entry, "BTCUSDT")][0] == pytest.approx(0.2)


def test_split_public_flow_omits_entire_timestamp_when_one_symbol_is_incomplete(tmp_path):
    entry = 10_000_000
    delta = tmp_path / "candles.jsonl"
    database = tmp_path / "micro.db"
    connection = sqlite3.connect(database)
    connection.execute(
        "CREATE TABLE kline_microstructure (symbol TEXT,open_time_ms INTEGER,"
        "quote_volume REAL,trade_count INTEGER,taker_buy_base REAL,"
        "taker_buy_quote REAL,PRIMARY KEY(symbol,open_time_ms))"
    )
    lines = []
    for symbol in CANONICAL_SYMBOLS:
        for offset in range(24, 0, -1):
            if symbol == "BTCUSDT" and offset == 1:
                continue
            timestamp = entry - offset * 300_000
            lines.append(json.dumps({"symbol": symbol, "open_time_ms": timestamp, "volume": 100.0}))
            connection.execute(
                "INSERT INTO kline_microstructure VALUES(?,?,?,?,?,?)",
                (symbol, timestamp, 1.0, 1, 60.0, 1.0),
            )
    connection.commit()
    connection.close()
    delta.write_text("\n".join(lines) + "\n", encoding="utf-8")
    lookup, inventory = split_public_flow_lookup(
        candle_delta=delta,
        microstructure_database=database,
        required_entry_timestamps_ms=[entry],
    )
    assert lookup == {}
    assert inventory["complete_eleven_symbol_timestamps"] == 0

