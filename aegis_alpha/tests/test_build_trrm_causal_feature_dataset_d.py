#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import sqlite3
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pandas as pd

from aegis_alpha.tools.build_trrm_causal_feature_dataset_d import (
    build_dataset,
    compute_causal_features,
    is_leakage_column,
    load_ohlcv,
)


FIELDS = [
    "timestamp",
    "symbol",
    "timeframe",
    "horizon",
    "close",
    "future_mfe_roe_proxy",
    "future_mae_roe_proxy",
    "time_to_mfe",
    "time_to_mae",
    "mfe_before_mae",
    "mfe_mae_ratio",
    "net_quality_after_costs",
    "clean_entry_v4",
    "bad_entry_v4",
    "premium_allowed_v4",
    "management_dependent_v4",
    "no_trade_v4",
    "tail_risk_v4",
    "early_mae_v4",
    "qmae_target",
    "q95_mae_target",
]


def fixture_rows(n: int = 90) -> list[dict]:
    out = []
    for i in range(n):
        tail = i % 5 == 0
        out.append({
            "timestamp": f"2026-07-01 {i // 12:02d}:{(i % 12) * 5:02d}:00",
            "symbol": "ADAUSDT",
            "timeframe": "5m",
            "horizon": "6",
            "close": 1.0 + i * 0.001,
            "future_mfe_roe_proxy": 0.08,
            "future_mae_roe_proxy": 0.12 if tail else 0.02,
            "time_to_mfe": 4,
            "time_to_mae": 2 if tail else 5,
            "mfe_before_mae": not tail,
            "mfe_mae_ratio": 0.8 if tail else 2.0,
            "net_quality_after_costs": -0.04 if tail else 0.04,
            "clean_entry_v4": 0 if tail else 1,
            "bad_entry_v4": 1 if tail else 0,
            "premium_allowed_v4": 0,
            "management_dependent_v4": 0,
            "no_trade_v4": 1 if tail else 0,
            "tail_risk_v4": 1 if tail else 0,
            "early_mae_v4": 1 if tail else 0,
            "qmae_target": 0.12 if tail else 0.02,
            "q95_mae_target": 0.12,
        })
    return out


def write_fixture_csv(path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(fixture_rows())


def write_fixture_db(path: Path, n: int = 130) -> None:
    con = sqlite3.connect(path)
    con.execute("create table ohlcv_data (symbol text, timeframe text, timestamp text, open real, high real, low real, close real, volume real, buy_volume real)")
    rows = []
    price = 1.0
    for symbol in ("ADA/USDT", "BTC/USDT", "ETH/USDT"):
        price = 1.0 if symbol == "ADA/USDT" else 100.0
        for i in range(n):
            ts = f"2026-07-01 {i // 12:02d}:{(i % 12) * 5:02d}:00"
            open_ = price
            close = price * (0.999 if i % 7 else 1.002)
            high = max(open_, close) * 1.003
            low = min(open_, close) * 0.997
            volume = 1000 + i * 3
            rows.append((symbol, "5m", ts, open_, high, low, close, volume, volume * 0.5))
            price = close
    con.executemany("insert into ohlcv_data values (?,?,?,?,?,?,?,?,?)", rows)
    con.commit()
    con.close()


def test_leakage_rules_and_close_override() -> None:
    assert is_leakage_column("feature.future_mae_roe_proxy")[0] is True
    assert is_leakage_column("feature.close_vs_ema_12")[0] is False
    assert is_leakage_column("feature.close_position_in_range")[0] is False


def test_compute_features_no_negative_shift_shape() -> None:
    candles = pd.DataFrame({
        "timestamp": pd.date_range("2026-07-01", periods=40, freq="5min"),
        "open": [1 + i * 0.001 for i in range(40)],
        "high": [1.004 + i * 0.001 for i in range(40)],
        "low": [0.996 + i * 0.001 for i in range(40)],
        "close": [1.001 + i * 0.001 for i in range(40)],
        "volume": [100 + i for i in range(40)],
    })
    feats = compute_causal_features(candles)
    assert "ret_1" in feats
    assert "breakdown_proxy_12" in feats
    assert feats["ret_1"].iloc[0] != feats["ret_1"].iloc[0]


def test_script_runs_with_fixture_and_reports() -> None:
    with tempfile.TemporaryDirectory() as td:
        csv_path = Path(td) / "samples.csv"
        db_path = Path(td) / "candles.db"
        write_fixture_csv(csv_path)
        write_fixture_db(db_path)
        import aegis_alpha.tools.build_trrm_causal_feature_dataset_d as mod
        old_db = mod.DB_PATH
        mod.DB_PATH = db_path
        try:
            result = build_dataset(argparse.Namespace(input_csv=str(csv_path), output_dir=td, max_rows=0, symbols="", horizons="", strict_causal="true", write_report="true"))
        finally:
            mod.DB_PATH = old_db
        assert Path(result["artifacts"]["csv"]).exists()
        assert Path(result["artifacts"]["md"]).exists()
        assert Path(result["artifacts"]["json"]).exists()
        assert result["decision"] in {"CAUSAL_FEATURE_DATASET_READY", "CAUSAL_FEATURE_DATASET_PARTIAL", "DATASET_NOT_USABLE", "LEAKAGE_RISK_TOO_HIGH"}
        assert result["feature_columns"]
        assert result["target_columns"]
        assert all(not c.startswith("target.") for c in result["feature_columns"])
        assert result["safety_confirmations"]["no_active_manifest"] is True
        assert result["safety_confirmations"]["no_yaml"] is True
        assert result["safety_confirmations"]["no_ts_touched"] is True
        assert result["safety_confirmations"]["no_model_training"] is True
        assert result["unavailable_features"] is not None


if __name__ == "__main__":
    test_leakage_rules_and_close_override()
    test_compute_features_no_negative_shift_shape()
    test_script_runs_with_fixture_and_reports()
    print("test_build_trrm_causal_feature_dataset_d: OK")
