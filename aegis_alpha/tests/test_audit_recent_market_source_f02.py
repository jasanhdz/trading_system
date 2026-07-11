#!/usr/bin/env python3
from __future__ import annotations

import sqlite3
import sys
import tempfile
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import aegis_alpha.tools.audit_recent_market_source_f02 as mod


def make_candles(start: str, n: int) -> pd.DataFrame:
    ts = pd.date_range(start, periods=n, freq="5min", tz="UTC")
    return pd.DataFrame({
        "timestamp": ts,
        "open": [100.0 + i for i in range(n)],
        "high": [101.0 + i for i in range(n)],
        "low": [99.0 + i for i in range(n)],
        "close": [100.5 + i for i in range(n)],
        "volume": [10.0 + i for i in range(n)],
        "buy_volume": [5.0 + i for i in range(n)],
    })


def test_local_coverage_and_public_snapshot_fallback() -> None:
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        db = root / "candles.db"
        con = sqlite3.connect(db)
        con.execute("create table ohlcv_data (symbol text, timeframe text, timestamp text, open real, high real, low real, close real, volume real, buy_volume real)")
        con.execute("insert into ohlcv_data values (?,?,?,?,?,?,?,?,?)", ("BTC/USDT", "5m", "2026-06-01 00:00:00", 1, 2, 1, 2, 10, 5))
        con.commit()
        con.close()
        cov = mod.local_coverage(db, ["BTCUSDT"], pd.Timestamp("2026-06-01T00:00:00Z"), pd.Timestamp("2026-06-01T00:10:00Z"), "5m")
        assert cov[0]["complete"] is False

        def fake_fetch(symbol: str, start: pd.Timestamp, end: pd.Timestamp, retries: int = 3) -> pd.DataFrame:
            assert symbol in {"BTCUSDT", "ETHUSDT"}
            return make_candles(str(start), 3)

        old = mod.fetch_binance_klines
        mod.fetch_binance_klines = fake_fetch
        try:
            payload = mod.run_audit(
                mod.parse_args([
                    "--db-path", str(db),
                    "--output-root", str(root / "snapshots"),
                    "--symbols", "BTCUSDT,ETHUSDT",
                    "--start", "2026-06-01T00:00:00Z",
                    "--until", "2026-06-01T00:10:00Z",
                    "--freeze-time", "2026-06-01T00:20:00Z",
                ])
            )
        finally:
            mod.fetch_binance_klines = old
        assert payload["source_selected"] == "binance_futures_public_klines"
        assert payload["decision"] == "RECENT_MARKET_SOURCE_READY"
        assert payload["read_only"] is True
        assert payload["uses_api_key"] is False
        assert Path(payload["snapshot_dir"], "snapshot_manifest.json").exists()


if __name__ == "__main__":
    test_local_coverage_and_public_snapshot_fallback()
    print("test_audit_recent_market_source_f02: OK")
