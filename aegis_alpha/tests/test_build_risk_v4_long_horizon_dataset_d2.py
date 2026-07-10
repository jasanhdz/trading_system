#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sqlite3
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pandas as pd

from aegis_alpha.tools.build_risk_v4_long_horizon_dataset_d2 import load_period, run_builder, uniform_steps


def make_db(path: Path, days: int = 210) -> None:
    con = sqlite3.connect(path)
    con.execute("create table ohlcv_data (symbol text, timeframe text, timestamp text, open real, high real, low real, close real, volume real, buy_volume real)")
    rows = []
    periods = days * 288
    start = pd.Timestamp("2025-01-01")
    for sym in ("ADA/USDT", "BTC/USDT", "ETH/USDT"):
        price = 1.0 if sym == "ADA/USDT" else 100.0
        for i in range(periods):
            ts = start + pd.Timedelta(minutes=5 * i)
            open_ = price
            close = price * (1 + (0.001 if i % 17 == 0 else -0.0002))
            high = max(open_, close) * 1.002
            low = min(open_, close) * 0.998
            volume = 1000 + (i % 50)
            rows.append((sym, "5m", str(ts), open_, high, low, close, volume, volume * 0.5))
            price = close
    con.executemany("insert into ohlcv_data values (?,?,?,?,?,?,?,?,?)", rows)
    con.commit()
    con.close()


def test_lookback_days_changes_period() -> None:
    with tempfile.TemporaryDirectory() as td:
        db = Path(td) / "candles.db"
        make_db(db)
        a = load_period(db, "ADAUSDT", "5m", 30)
        b = load_period(db, "ADAUSDT", "5m", 180)
        assert len(b) > len(a) * 4
        assert b["timestamp"].min() < a["timestamp"].min()


def test_uniform_sampling_covers_start_middle_end() -> None:
    steps = uniform_steps(pd.Series(range(1000)).to_numpy(), 30, 42)
    assert steps.min() < 20
    assert 450 < steps[len(steps) // 2] < 550
    assert steps.max() > 970


def test_builder_outputs_dense_and_strided() -> None:
    with tempfile.TemporaryDirectory() as td:
        db = Path(td) / "candles.db"
        make_db(db)
        result = run_builder(argparse.Namespace(db_path=str(db), output_dir=td, lookback_days=180, symbols="ADAUSDT", horizons="6 12 24", timeframe_minutes=5, max_rows_per_combo=200, sampling_mode="uniform_time", seed=7, write_report="true"))
        assert Path(result["artifacts"]["dense_csv"]).exists()
        assert Path(result["artifacts"]["causal_csv"]).exists()
        assert Path(result["artifacts"]["strided_csv"]).exists()
        assert result["rows_dense"] > result["rows_strided"]
        assert result["actual_days"] >= 170
        assert result["sampling_manifest"]["rows_before"] > result["sampling_manifest"]["rows_after"]
        df = pd.read_csv(result["artifacts"]["causal_csv"])
        assert df["id.timestamp"].is_monotonic_increasing
        assert result["safety_confirmations"]["no_trrm_training"] is True
        assert result["safety_confirmations"]["no_ts_touched"] is True


if __name__ == "__main__":
    test_lookback_days_changes_period()
    test_uniform_sampling_covers_start_middle_end()
    test_builder_outputs_dense_and_strided()
    print("test_build_risk_v4_long_horizon_dataset_d2: OK")
