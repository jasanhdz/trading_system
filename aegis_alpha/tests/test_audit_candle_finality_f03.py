#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sqlite3
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import pandas as pd

from aegis_alpha.tools.audit_candle_finality_f03 import future_max_high, run_audit


def _final_candles(n: int = 400, seed: int = 3) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    ts = pd.date_range("2026-07-01", periods=n, freq="5min")
    close = 100 + np.cumsum(rng.normal(0, 0.3, n))
    high = close + rng.random(n) * 0.8
    low = close - rng.random(n) * 0.8
    open_ = np.roll(close, 1)
    open_[0] = close[0]
    volume = rng.random(n) * 100 + 50
    return pd.DataFrame({"timestamp": ts, "open": open_, "high": high, "low": low, "close": close, "volume": volume, "buy_volume": volume * 0.5})


def _write_fixture(td: Path, partial: bool) -> argparse.Namespace:
    final = _final_candles()
    local = final.copy()
    if partial:
        # every 3rd candle captured mid-bar: strict subset of the final candle
        idx = np.arange(0, len(local), 3)
        local.loc[idx, "high"] = local.loc[idx, "high"] - 0.5
        local.loc[idx, "low"] = local.loc[idx, "low"] + 0.2
        local.loc[idx, "close"] = local.loc[idx, "close"] - 0.4
        local.loc[idx, "volume"] = local.loc[idx, "volume"] * 0.3
    db_path = td / "candles.db"
    with sqlite3.connect(db_path) as con:
        rows = local.copy()
        rows.insert(0, "timeframe", "5m")
        rows.insert(0, "symbol", "BTC/USDT")
        rows["timestamp"] = rows["timestamp"].astype(str)
        rows.to_sql("ohlcv_data", con, index=False)
    snap_dir = td / "snap"
    snap_dir.mkdir()
    final.to_csv(snap_dir / "BTCUSDT_5m.csv", index=False)
    return argparse.Namespace(db_path=str(db_path), snapshot_dir=str(snap_dir), symbols="BTCUSDT", timeframe="5m", horizons="12,24", output_dir=str(td))


def test_future_max_high() -> None:
    high = np.array([1.0, 5.0, 2.0, 4.0, 3.0])
    out = future_max_high(high, 2)
    assert out[0] == 5.0 and out[1] == 4.0 and out[2] == 4.0
    assert np.isnan(out[-1]) and np.isnan(out[-2])


def test_clean_store_passes() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        result = run_audit(_write_fixture(Path(tmp), partial=False))
        assert result["decision"] == "CANDLES_FINAL_OK"
        assert result["aggregate"]["max_partial_candle_rate"] <= 0.001


def test_partial_candles_detected_with_signature() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        result = run_audit(_write_fixture(Path(tmp), partial=True))
        assert result["decision"] == "PARTIAL_CANDLES_DETECTED"
        sym = result["per_symbol"][0]
        assert 0.30 <= sym["partial_candle_rate"] <= 0.36
        assert sym["open_always_exact"] is True
        assert sym["partial_is_strict_subset_of_final"] is True
        flips = {f["horizon"]: f for f in sym["label_flip_by_horizon"]}
        assert flips[12]["rows"] > 0
        assert Path(result["outputs"]["md"]).exists()


if __name__ == "__main__":
    test_future_max_high()
    test_clean_store_passes()
    test_partial_candles_detected_with_signature()
    print("test_audit_candle_finality_f03: OK")
