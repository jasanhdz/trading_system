from __future__ import annotations

import json
import os
from dataclasses import replace
from pathlib import Path

import joblib
import numpy as np

from aegis_alpha.turbo import turbo_signal


class ConstantEstimator:
    def __init__(self, value: float):
        self.value = value

    def predict(self, x):
        return np.full((len(x),), self.value, dtype=np.float32)


def _touch_newer(path: Path) -> None:
    now = os.stat(path).st_mtime_ns + 10_000_000
    os.utime(path, ns=(now, now))


def _write_model(path: Path, value: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"estimator": ConstantEstimator(value), "metadata": {"sample_count": 2017}}, path)


def _write_manifest(symbol_dir: Path, version: str) -> None:
    active_dir = symbol_dir / "active"
    manifest = {
        "symbol": "ETHUSDT",
        "version": version,
        "created_at": version,
        "promoted_at": version,
        "validation_status": "passed",
        "windows": [7],
        "model_paths": {
            "long_7d": str(active_dir / "turbo_long_edge_7d_v010.joblib"),
            "short_7d": str(active_dir / "turbo_short_edge_7d_v010.joblib"),
        },
    }
    path = symbol_dir / "active_manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    _touch_newer(path)


def _write_active_set(symbol_dir: Path, long_value: float, short_value: float) -> None:
    active_dir = symbol_dir / "active"
    _write_model(active_dir / "turbo_long_edge_7d_v010.joblib", long_value)
    _write_model(active_dir / "turbo_short_edge_7d_v010.joblib", short_value)


def test_turbo_model_set_hot_reloads_only_after_manifest_changes(monkeypatch, tmp_path):
    symbol_dir = tmp_path / "ETHUSDT"
    monkeypatch.setattr(turbo_signal, "turbo_symbol_model_dir", lambda symbol: symbol_dir)
    monkeypatch.setattr(
        turbo_signal,
        "DEFAULT_TURBO_CONFIG",
        replace(turbo_signal.DEFAULT_TURBO_CONFIG, lookback_days=(7,)),
    )
    turbo_signal._MODEL_CACHE.clear()
    turbo_signal._MODEL_SET_CACHE.clear()

    x = np.zeros((1, 3), dtype=np.float32)
    _write_active_set(symbol_dir, long_value=1.0, short_value=0.1)
    _write_manifest(symbol_dir, "v1")

    scores, votes = turbo_signal._score_models(x, "ETHUSDT")
    assert scores["long_7d"] == 1.0
    assert abs(float(scores["short_7d"]) - 0.1) < 1e-6
    assert votes["long"] == 1

    _write_active_set(symbol_dir, long_value=0.2, short_value=2.0)
    scores, votes = turbo_signal._score_models(x, "ETHUSDT")
    assert scores["long_7d"] == 1.0
    assert abs(float(scores["short_7d"]) - 0.1) < 1e-6
    assert votes["long"] == 1

    _write_manifest(symbol_dir, "v2")
    scores, votes = turbo_signal._score_models(x, "ETHUSDT")
    assert abs(float(scores["long_7d"]) - 0.2) < 1e-6
    assert scores["short_7d"] == 2.0
    assert votes["short"] == 1

    turbo_signal._MODEL_CACHE.clear()
    turbo_signal._MODEL_SET_CACHE.clear()
