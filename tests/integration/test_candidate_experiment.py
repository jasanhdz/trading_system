import csv
import hashlib
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import yaml

from aegis.config import CANONICAL_SYMBOLS
from aegis.training.experiment import run_experiment
from aegis.models import load_model_bundle


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _create_canonical_series(path: Path) -> str:
    path.mkdir()
    start = datetime(2025, 12, 31, 18, 0, tzinfo=timezone.utc)
    included = {}
    for symbol_index, symbol in enumerate(CANONICAL_SYMBOLS):
        file_path = path / f"{symbol}_5m.csv"
        with file_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle); writer.writerow(("timestamp", "open", "high", "low", "close", "volume"))
            for bar_index in range(1700):
                timestamp = start + timedelta(minutes=5 * bar_index)
                drift = (symbol_index - 5) * 0.00003
                wave = ((bar_index % 24) - 12) * 0.00001
                open_price = (10 + symbol_index) * (1 + drift * bar_index)
                close = open_price * (1 + drift + wave)
                writer.writerow((timestamp.replace(tzinfo=None).isoformat(sep=" "), open_price,
                                 max(open_price, close) * 1.001, min(open_price, close) * 0.999,
                                 close, 1000 + bar_index))
        included[symbol] = {"sha256": _sha(file_path), "rows": 1700}
    passes = [{"symbol": symbol, "passes": True} for symbol in CANONICAL_SYMBOLS]
    manifest = {
        "schema": "gen2_d3_series_v1", "status": "OK", "artifact_id": "fixture-canonical-v1",
        "excluded_symbols": [], "included_symbols": included,
        "gates": {"g4_gaps": passes, "g5_coverage": passes},
    }
    manifest_path = path / "series_manifest.json"
    manifest_path.write_text(json.dumps(manifest, sort_keys=True), encoding="utf-8")
    lines = [f"{included[symbol]['sha256']}  {symbol}_5m.csv" for symbol in CANONICAL_SYMBOLS]
    lines.append(f"{_sha(manifest_path)}  series_manifest.json")
    (path / "series_manifest.sha256").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return _sha(manifest_path)


def _config(tmp_path: Path) -> Path:
    config_dir = tmp_path / "config"; data_dir = tmp_path / "data"
    config_dir.mkdir(); data_dir.mkdir(); manifest_hash = _create_canonical_series(data_dir / "canonical")
    payload = {
        "schema_version": "aegis-candidate-experiment-v1", "experiment_id": "fixture-experiment",
        "data": {"source_kind": "canonical_d3_series", "source": "data/canonical", "manifest_sha256": manifest_hash,
                 "timeframe": "5m", "start": "2026-01-01T00:00:00Z",
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
    before = {path.name: _sha(path) for path in (tmp_path / "data" / "canonical").iterdir()}
    first = run_experiment(config_path)
    second = run_experiment(config_path)
    assert first.dataset_hash == second.dataset_hash
    assert first.artifact_hash == second.artifact_hash
    assert first.baselines == second.baselines
    assert first.source_audit.read_only is True
    assert first.source_audit.accepted_cycles >= 100
    assert first.partition.train_window[1] < first.partition.validation_window[0] < first.partition.test_window[0]
    assert {"no_trade", "random", "momentum", "mean_reversion", "last_candle", "model_no_layers", "model_full_layers"} == set(first.baselines)
    assert first.classification in {"EXPERIMENTAL_SMOKE_REJECTED", "EXPERIMENTAL_SMOKE_CRITERIA_MET"}
    assert all(metric.profit_factor < float("inf") for metric in first.baselines.values())
    bundle_path = tmp_path / "candidate.json"
    bundle_path.write_text(json.dumps(first.candidate_bundle), encoding="utf-8")
    assert load_model_bundle(bundle_path).metadata.trained is True
    after = {path.name: _sha(path) for path in (tmp_path / "data" / "canonical").iterdir()}
    assert after == before
