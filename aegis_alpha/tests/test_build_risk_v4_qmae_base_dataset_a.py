#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sqlite3
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from aegis_alpha.tools.build_risk_v4_qmae_base_dataset_a import build_samples, run_builder, sqlite_row_limit, timeframe_minutes


def fixture_candles(n: int = 80) -> list[dict]:
    rows = []
    price = 100.0
    for i in range(n):
        if i == 25:
            high = price * 1.006
            low = price * 0.995
        else:
            high = price * 1.002
            low = price * 0.997
        close = price * (0.999 if i % 5 else 1.001)
        rows.append({"timestamp": f"2026-07-09 00:{i:02d}:00", "open": price, "high": high, "low": low, "close": close, "volume": 1.0, "buy_volume": 0.5})
        price = close
    return rows


def test_no_db_status() -> None:
    with tempfile.TemporaryDirectory() as td:
        result = run_builder(argparse.Namespace(symbols="ADAUSDT", timeframes="5m", horizons="6", lookback_days=1, out_dir=td, db_path=str(Path(td) / "missing.db")))
        assert result["status"] == "INSUFFICIENT_OHLCV_DATA"
        assert result["sample_count"] == 0


def test_fixture_labels() -> None:
    rows = build_samples(fixture_candles(), "ADAUSDT", "5m", [6])
    assert rows
    assert {"tail_risk_v4", "early_mae_v4", "qmae_target"} <= set(rows[0].keys())
    assert all(float(r["qmae_target"]) >= 0 for r in rows)


def test_sqlite_builder_serializes() -> None:
    with tempfile.TemporaryDirectory() as td:
        db = Path(td) / "candles.db"
        con = sqlite3.connect(db)
        con.execute("create table ohlcv_data (symbol text, timeframe text, timestamp text, open real, high real, low real, close real, volume real, buy_volume real)")
        con.executemany(
            "insert into ohlcv_data values (?,?,?,?,?,?,?,?,?)",
            [("ADA/USDT", "5m", r["timestamp"], r["open"], r["high"], r["low"], r["close"], r["volume"], r["buy_volume"]) for r in fixture_candles(90)],
        )
        con.commit()
        con.close()
        result = run_builder(argparse.Namespace(symbols="ADAUSDT", timeframes="5m", horizons="6", lookback_days=1, out_dir=td, db_path=str(db)))
        assert result["status"] == "OK"
        assert Path(result["outputs"]["samples_csv"]).exists()


def test_no_train() -> None:
    with tempfile.TemporaryDirectory() as td:
        result = run_builder(argparse.Namespace(symbols="ADAUSDT", timeframes="5m", horizons="6", lookback_days=1, out_dir=td, db_path=str(Path(td) / "missing.db")))
        assert result["no_train"] is True


def test_lookback_days_drives_sqlite_limit() -> None:
    assert timeframe_minutes("5m") == 5
    assert timeframe_minutes("15m") == 15
    assert timeframe_minutes("1h") == 60
    # 365 days of 5m candles is ~105k rows, far beyond the old fixed 5000 limit.
    assert sqlite_row_limit("5m", 365) == 365 * 288 + 64
    assert sqlite_row_limit("5m", 17) == 17 * 288 + 64
    assert sqlite_row_limit("5m", 10_000) == 250_000  # capped
    assert sqlite_row_limit("15m", 30) == 30 * 96 + 64


if __name__ == "__main__":
    test_no_db_status()
    test_fixture_labels()
    test_sqlite_builder_serializes()
    test_no_train()
    test_lookback_days_drives_sqlite_limit()
    print("test_build_risk_v4_qmae_base_dataset_a: OK")
