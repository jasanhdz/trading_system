from __future__ import annotations

from pathlib import Path

import numpy as np

from aegis_alpha.decision_brain.decision_brain_shadow import evaluate_decision_brain_shadow
from aegis_alpha.decision_brain.feature_builder import build_decision_brain_features
from aegis_alpha.decision_brain.model_loader import load_decision_brain_artifacts


def _turbo_context() -> dict:
    return {
        "action": "LONG",
        "turbo_score": 0.72,
        "votes": {"long": 2, "short": 0, "neutral": 1},
        "recent_scores": {
            "long_7d": 0.001,
            "long_14d": 0.002,
            "long_30d": 0.001,
            "short_7d": -0.001,
            "short_14d": -0.002,
            "short_30d": -0.001,
        },
        "freshness": {"feature_timestamp": "2026-05-15T00:00:00+00:00"},
        "raw": {
            "action": "LONG",
            "turbo_score": 0.72,
            "votes": {"long": 2, "short": 0, "neutral": 1},
            "recent_scores": {
                "long_7d": 0.001,
                "long_14d": 0.002,
                "long_30d": 0.001,
                "short_7d": -0.001,
                "short_14d": -0.002,
                "short_30d": -0.001,
            },
        },
    }


def _entry_quality() -> dict:
    return {
        "mode": "SHADOW",
        "entry_quality_score": 0.64,
        "tail_risk_score": 0.37,
        "recommendation": "ALLOW_SHADOW",
        "feature_status": "ok",
    }


def _event_risk_auto() -> dict:
    return {
        "mode": "SHADOW",
        "suggested_mode": "CAUTION",
        "confidence": 0.72,
        "reasons": ["btc_weak_or_hold"],
        "btc_context": {"turbo_action": "HOLD", "turbo_score": 0.28, "tail_risk_score": 0.0},
        "eth_context": {"turbo_action": "HOLD", "turbo_score": 0.0, "tail_risk_score": 0.0},
    }


def test_missing_artifacts_no_crash(tmp_path: Path):
    artifacts = load_decision_brain_artifacts(model_dir=tmp_path, force_reload=True)
    assert artifacts is None


def test_loads_model_v010_successfully():
    artifacts = load_decision_brain_artifacts(force_reload=True)
    assert artifacts is not None
    assert artifacts.model_version == "v010"
    assert len(artifacts.feature_columns) == 112


def test_output_is_shadow_and_never_executable():
    row = evaluate_decision_brain_shadow("ETHUSDT", "LONG", _turbo_context(), _entry_quality(), _event_risk_auto())
    assert row["mode"] == "SHADOW"
    assert row["execute"] is False
    assert row["production_allowed"] is False
    assert row["status"] == "RESEARCH_CANDIDATE_NOT_LIVE"


def test_probabilities_sum_to_one():
    row = evaluate_decision_brain_shadow("ETHUSDT", "LONG", _turbo_context(), _entry_quality(), _event_risk_auto())
    total = row["enter_now_prob"] + row["wait_confirmation_prob"] + row["manual_only_prob"] + row["do_not_enter_prob"]
    assert abs(total - 1.0) < 0.02


def test_latest_news_missing_does_not_crash(monkeypatch):
    monkeypatch.setattr("aegis_alpha.decision_brain.feature_builder.LATEST_NEWS_PATH", Path("/tmp/missing-news.json"))
    row = evaluate_decision_brain_shadow("BTCUSDT", "LONG", _turbo_context(), _entry_quality(), _event_risk_auto())
    assert row["mode"] == "SHADOW"
    assert row["execute"] is False


def test_feature_alignment_to_112_columns():
    artifacts = load_decision_brain_artifacts()
    assert artifacts is not None
    vector, meta = build_decision_brain_features(
        symbol="SUIUSDT",
        side="LONG",
        turbo_context=_turbo_context(),
        entry_quality_model=_entry_quality(),
        event_risk_auto=_event_risk_auto(),
        news_sentiment={"suggested_mode": "NORMAL", "risk_score": 0.1, "confidence": 0.7},
        feature_columns=artifacts.feature_columns,
    )
    assert vector.shape == (1, 112)
    assert np.isfinite(vector[:, artifacts.feature_columns.index("eq_turbo_score")]).all()
    assert meta["missing_features_count"] >= 0
