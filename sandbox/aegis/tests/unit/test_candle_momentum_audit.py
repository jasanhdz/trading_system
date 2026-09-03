from __future__ import annotations

import sqlite3
from pathlib import Path

import pandas as pd

from aegis.research.candle_momentum_audit import (
    _prepare_events,
    load_candle_momentum_audit_config,
    render_candle_momentum_markdown,
    run_candle_momentum_audit,
)


ROOT = Path(__file__).resolve().parents[2]


def _config():
    return load_candle_momentum_audit_config(
        ROOT / "config/experiments/aegis_candle_momentum_audit_v1.yaml",
        repo_root=ROOT,
    )


def test_exact_two_pattern_tracks_third_and_fourth_without_doji() -> None:
    config = _config()
    timestamp = pd.date_range("2026-01-01", periods=30, freq="5min", tz="UTC")
    closes = [100.0 + (index % 2) * 0.1 for index in range(24)]
    closes.extend([100.0, 101.0, 102.0, 103.0, 104.0, 103.0])
    opens = [value - 0.1 if index % 2 else value + 0.1 for index, value in enumerate(closes[:24])]
    opens.extend([101.0, 100.0, 101.0, 102.0, 103.0, 104.0])
    frame = pd.DataFrame(
        {
            "timestamp": timestamp,
            "open": opens,
            "high": [max(left, right) + 0.1 for left, right in zip(opens, closes)],
            "low": [min(left, right) - 0.1 for left, right in zip(opens, closes)],
            "close": closes,
            "volume": [100.0 + index for index in range(30)],
        }
    )

    events, _ = _prepare_events(frame, config, run_length=2)
    event = events.loc[events["timestamp"] == timestamp[26]].iloc[0]

    assert event["direction"] == "GREEN"
    assert bool(event["third_same"])
    assert bool(event["fourth_same"])
    assert bool(event["run_reaches_four"])
    assert not bool(event["next_opposite"])
    assert not bool(event["next_doji"])


def test_exact_three_and_four_patterns_use_first_following_candle() -> None:
    config = _config()
    timestamp = pd.date_range("2026-01-01", periods=40, freq="5min", tz="UTC")
    colors = [1 if index % 2 else -1 for index in range(30)]
    colors.extend([-1, 1, 1, 1, 1, 1, -1, -1, -1, -1])
    opens = [100.0] * len(colors)
    closes = [101.0 if color > 0 else 99.0 for color in colors]
    frame = pd.DataFrame(
        {
            "timestamp": timestamp,
            "open": opens,
            "high": [102.0] * len(colors),
            "low": [98.0] * len(colors),
            "close": closes,
            "volume": [100.0 + index for index in range(len(colors))],
        }
    )

    run_three, _ = _prepare_events(frame, config, run_length=3)
    run_four, _ = _prepare_events(frame, config, run_length=4)

    three = run_three.loc[run_three["timestamp"] == timestamp[33]].iloc[0]
    four = run_four.loc[run_four["timestamp"] == timestamp[34]].iloc[0]
    assert bool(three["third_same"])
    assert bool(three["fourth_same"])
    assert bool(four["third_same"])


def test_config_is_research_only_and_has_no_runtime_authority() -> None:
    config = _config()
    source = (
        ROOT / "src/aegis/research/candle_momentum_audit.py"
    ).read_text(encoding="utf-8").lower()

    assert config.symbols
    assert config.exact_run_lengths == (2, 3, 4)
    assert config.timeframe == "5m"
    for forbidden in ("create_order", "cancel_order", "pm2", "api_secret"):
        assert forbidden not in source


def test_audit_reads_sqlite_in_read_only_mode(tmp_path: Path) -> None:
    config = _config()
    database = tmp_path / "candles.db"
    connection = sqlite3.connect(database)
    connection.execute(
        "CREATE TABLE ohlcv_data ("
        "timestamp TEXT, symbol TEXT, timeframe TEXT, open REAL, high REAL, "
        "low REAL, close REAL, volume REAL)"
    )
    timestamp = pd.date_range("2024-01-01", periods=40, freq="5min")
    for symbol in config.symbols:
        rows = []
        for index, stamp in enumerate(timestamp):
            open_price = 100.0
            close_price = 101.0 if index % 3 else 99.0
            rows.append(
                (
                    stamp.isoformat(),
                    f"{symbol[:-4]}/USDT",
                    "5m",
                    open_price,
                    102.0,
                    98.0,
                    close_price,
                    100.0 + index,
                )
            )
        connection.executemany(
            "INSERT INTO ohlcv_data VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            rows,
        )
    connection.commit()
    connection.close()
    altered = config.__class__(
        **{
            **config.__dict__,
            "database": database,
            "lookback_days": 730,
            "minimum_validation_events": 1,
        }
    )

    report = run_candle_momentum_audit(altered)

    assert report["exchange_calls"] == 0
    assert report["exchange_mutations"] == 0
    assert report["live_changes_authorized"] is False
    assert set(report["run_lengths"]) == {"2", "3", "4"}
    assert "Aegis 5m" in render_candle_momentum_markdown(report)
