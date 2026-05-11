from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pytest
from fastapi.testclient import TestClient

from aegis_alpha.entry_quality import model_loader
from aegis_alpha.entry_quality.entry_quality_shadow import evaluate_entry_quality_shadow


class ConstantClassifier:
    def __init__(self, score: float):
        self.score = float(score)

    def predict_proba(self, x):
        return np.tile(np.asarray([[1.0 - self.score, self.score]], dtype=np.float32), (len(x), 1))


class Passthrough:
    def transform(self, x):
        return x


@pytest.fixture(autouse=True)
def clear_cache_after_test():
    yield
    model_loader.clear_entry_quality_model_cache()


def test_missing_manifest_or_models_does_not_crash(monkeypatch, tmp_path):
    monkeypatch.setattr(model_loader, "MODEL_DIR", tmp_path)
    monkeypatch.setattr(model_loader, "MANIFEST_PATH", tmp_path / "model_manifest.json")
    monkeypatch.setattr(model_loader, "FEATURE_COLUMNS_PATH", tmp_path / "feature_columns.json")
    model_loader.clear_entry_quality_model_cache()

    result = evaluate_entry_quality_shadow("BTCUSDT")

    assert result["mode"] == "SHADOW"
    assert result["execute"] is False
    assert result["production_allowed"] is False
    assert result["recommendation"] == "INSUFFICIENT_DATA"
    assert result["model_scope"] == "none"


def test_model_load_with_v020_artifacts_returns_shadow_dict():
    model_loader.clear_entry_quality_model_cache()
    result = evaluate_entry_quality_shadow(
        "BTCUSDT",
        {
            "turbo": {
                "raw": {
                    "action": "LONG",
                    "turbo_score": 0.72,
                    "votes": {"long": 2, "short": 1, "neutral": 0},
                    "recent_scores": {
                        "long_7d": 0.001,
                        "long_14d": 0.002,
                        "long_30d": 0.0015,
                        "short_7d": -0.001,
                        "short_14d": 0.0,
                        "short_30d": -0.002,
                    },
                }
            }
        },
    )

    assert result["mode"] == "SHADOW"
    assert result["execute"] is False
    assert result["production_allowed"] is False
    assert result["status"] == "RESEARCH_CANDIDATE_NOT_LIVE"
    assert "recommendation" in result
    assert result["model_scope"] in {"symbol", "global", "none"}


def test_feature_columns_aligned_for_real_artifacts():
    cache = model_loader.load_entry_quality_models(force_reload=True)
    assert len(cache.feature_columns) == 73
    assert cache.feature_columns[0] == "ret_1"


def test_symbol_model_preferred_when_available():
    model_loader.clear_entry_quality_model_cache()
    pair = model_loader.get_model_pair("BTCUSDT")
    assert pair.scope == "symbol"


def test_global_fallback_if_symbol_model_missing(tmp_path, monkeypatch):
    features = ["ret_1"]
    (tmp_path / "model_manifest.json").write_text(json.dumps({"model_version": "v020"}), encoding="utf-8")
    (tmp_path / "feature_columns.json").write_text(
        json.dumps({"feature_columns": features, "symbol_encoding": {"BTCUSDT": 0}}),
        encoding="utf-8",
    )
    joblib.dump({"estimator": ConstantClassifier(0.7), "preprocessor": Passthrough()}, tmp_path / "global_entry_quality_model.joblib")
    joblib.dump({"estimator": ConstantClassifier(0.2), "preprocessor": Passthrough()}, tmp_path / "global_tail_risk_model.joblib")
    monkeypatch.setattr(model_loader, "MODEL_DIR", tmp_path)
    monkeypatch.setattr(model_loader, "MANIFEST_PATH", tmp_path / "model_manifest.json")
    monkeypatch.setattr(model_loader, "FEATURE_COLUMNS_PATH", tmp_path / "feature_columns.json")
    model_loader.clear_entry_quality_model_cache()

    pair = model_loader.get_model_pair("BTCUSDT")

    assert pair.scope == "global"


def test_ml_v2_predict_includes_entry_quality_model():
    from aegis_alpha.inference.server import app

    client = TestClient(app)
    response = client.post("/ml-v2/predict", json={"symbol": "BTCUSDT"})

    assert response.status_code == 200
    payload = response.json()
    eq = payload["aegis"]["entry_quality_model"]
    assert payload["aegis"]["turbo"]
    assert eq["mode"] == "SHADOW"
    assert eq["execute"] is False
    assert eq["production_allowed"] is False
    assert "recommendation" in eq


def test_model_error_does_not_break_predict(monkeypatch):
    import aegis_alpha.inference.server as server

    def boom(symbol, turbo_context=None):
        raise RuntimeError("forced")

    monkeypatch.setattr(server, "evaluate_entry_quality_shadow", boom)
    client = TestClient(server.app)
    response = client.post("/ml-v2/predict", json={"symbol": "BTCUSDT"})

    assert response.status_code == 200
    eq = response.json()["aegis"]["entry_quality_model"]
    assert eq["recommendation"] == "MODEL_ERROR"
    assert eq["execute"] is False
