import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path

import yaml

from aegis.config import CANONICAL_SYMBOLS
from aegis.training.experiment import run_experiment
from aegis.models import load_model_bundle


def _create_local_database(path: Path) -> None:
    connection = sqlite3.connect(path)
    connection.execute("CREATE TABLE ohlcv_data (id INTEGER PRIMARY KEY, symbol TEXT, timeframe TEXT, timestamp TEXT, open REAL, high REAL, low REAL, close REAL, volume REAL)")
    start = datetime(2025, 12, 31, 18, 0, tzinfo=timezone.utc)
    rows = []
    identifier = 1
    for symbol_index, symbol in enumerate(CANONICAL_SYMBOLS):
        database_symbol = symbol[:-4] + "/USDT"
        for bar_index in range(1700):
            timestamp = start + timedelta(minutes=5 * bar_index)
            drift = (symbol_index - 5) * 0.00003
            wave = ((bar_index % 24) - 12) * 0.00001
            open_price = (10 + symbol_index) * (1 + drift * bar_index)
            close = open_price * (1 + drift + wave)
            rows.append((identifier, database_symbol, "5m", timestamp.replace(tzinfo=None).strftime("%Y-%m-%d %H:%M:%S.%f"),
                         open_price, max(open_price, close) * 1.001, min(open_price, close) * 0.999, close, 1000 + bar_index))
            identifier += 1
    connection.executemany("INSERT INTO ohlcv_data VALUES (?,?,?,?,?,?,?,?,?)", rows)
    connection.execute("CREATE INDEX idx_symbol_timeframe_timestamp ON ohlcv_data(symbol,timeframe,timestamp)")
    connection.commit(); connection.close()


def _config(tmp_path: Path) -> Path:
    config_dir = tmp_path / "config"; data_dir = tmp_path / "data"
    config_dir.mkdir(); data_dir.mkdir(); _create_local_database(data_dir / "candles.db")
    payload = {
        "schema_version": "aegis-candidate-experiment-v1", "experiment_id": "fixture-experiment",
        "data": {"source": "data/candles.db", "timeframe": "5m", "start": "2026-01-01T00:00:00Z",
                 "end": "2026-01-06T12:00:00Z", "history_bars": 60, "horizon_bars": 12, "sample_every_bars": 12},
        "protocol": {"seed": 7, "train_fraction": 0.6, "validation_fraction": 0.2, "test_fraction": 0.2,
                     "embargo_minutes": 120, "friction_fraction": 0.0014, "direction_threshold": 0.1, "fold_count": 2},
        "promotion": {"mandatory": {"minimum_test_signals": 10, "minimum_positive_folds": 1,
                       "minimum_profit_factor": 1.0, "minimum_net_expectancy": 0.0,
                       "beat_best_directional_baseline_expectancy": True, "maximum_symbol_signal_fraction": 0.5,
                       "require_no_known_leakage": True}},
    }
    path = config_dir / "candidate.yaml"; path.write_text(yaml.safe_dump(payload), encoding="utf-8")
    return path


def test_candidate_experiment_is_read_only_temporal_and_reproducible(tmp_path: Path) -> None:
    config_path = _config(tmp_path)
    before = (tmp_path / "data" / "candles.db").stat().st_size
    first = run_experiment(config_path)
    second = run_experiment(config_path)
    assert first.dataset_hash == second.dataset_hash
    assert first.artifact_hash == second.artifact_hash
    assert first.baselines == second.baselines
    assert first.source_audit.read_only is True
    assert first.source_audit.accepted_cycles >= 100
    assert first.partition.train_window[1] < first.partition.validation_window[0] < first.partition.test_window[0]
    assert {"no_trade", "random", "momentum", "mean_reversion", "last_candle", "model_no_layers", "model_full_layers"} == set(first.baselines)
    assert first.classification in {"REJECTED", "APPROVED_FOR_SHADOW"}
    assert all(metric.profit_factor < float("inf") for metric in first.baselines.values())
    bundle_path = tmp_path / "candidate.json"
    import json
    bundle_path.write_text(json.dumps(first.candidate_bundle), encoding="utf-8")
    assert load_model_bundle(bundle_path).metadata.trained is True
    assert (tmp_path / "data" / "candles.db").stat().st_size == before
