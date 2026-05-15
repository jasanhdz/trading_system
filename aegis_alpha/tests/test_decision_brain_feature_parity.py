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
            "long_7d": 0.10,
            "long_14d": 0.12,
            "long_30d": 0.14,
            "short_7d": 0.02,
            "short_14d": 0.03,
            "short_30d": 0.04,
        },
        "freshness": {"feature_timestamp": "2026-05-15T00:00:00+00:00"},
        "raw": {
            "action": "LONG",
            "turbo_score": 0.72,
            "votes": {"long": 2, "short": 0, "neutral": 1},
            "recent_scores": {
                "long_7d": 0.10,
                "long_14d": 0.12,
                "long_30d": 0.14,
                "short_7d": 0.02,
                "short_14d": 0.03,
                "short_30d": 0.04,
            },
        },
    }


def _entry_quality() -> dict:
    return {
        "mode": "SHADOW",
        "entry_quality_score": 0.67,
        "tail_risk_score": 0.31,
        "recommendation": "ALLOW_SHADOW",
        "feature_status": "ok",
        "feature_parity_pct": 100.0,
    }


def _event_risk_auto() -> dict:
    return {
        "mode": "SHADOW",
        "suggested_mode": "CAUTION",
        "confidence": 0.72,
        "reasons": ["btc_weak_or_hold", "btc_eth_high_volatility"],
        "btc_context": {"turbo_action": "LONG", "turbo_score": 0.58, "tail_risk_score": 0.11},
        "eth_context": {"turbo_action": "LONG", "turbo_score": 0.61, "tail_risk_score": 0.10},
    }


def _news() -> dict:
    return {
        "suggested_mode": "CAUTION",
        "risk_score": 0.44,
        "confidence": 0.81,
        "top_events": [{"title": "Fear & Greed Index 52"}],
        "status": "ok",
    }


def _build():
    artifacts = load_decision_brain_artifacts()
    assert artifacts is not None
    return build_decision_brain_features(
        symbol="BTCUSDT",
        side="LONG",
        turbo_context=_turbo_context(),
        entry_quality_model=_entry_quality(),
        event_risk_auto=_event_risk_auto(),
        news_sentiment=_news(),
        feature_columns=artifacts.feature_columns,
    )


def test_feature_builder_aligns_to_112_columns():
    vector, meta = _build()
    assert vector.shape == (1, 112)
    assert meta["feature_parity_pct"] >= 75.0


def test_maps_entry_quality_model_fields():
    artifacts = load_decision_brain_artifacts()
    assert artifacts is not None
    vector, _meta = _build()
    row = vector[0]
    assert np.isclose(row[artifacts.feature_columns.index("entry_quality_score")], 0.67)
    assert np.isclose(row[artifacts.feature_columns.index("tail_risk_score")], 0.31)


def test_maps_event_risk_auto_fields():
    artifacts = load_decision_brain_artifacts()
    assert artifacts is not None
    vector, _meta = _build()
    row = vector[0]
    assert np.isclose(row[artifacts.feature_columns.index("event_risk_auto_confidence")], 0.72)
    assert np.isclose(row[artifacts.feature_columns.index("event_risk_auto_reason_count")], 2.0)


def test_maps_news_sentiment_latest():
    artifacts = load_decision_brain_artifacts()
    assert artifacts is not None
    vector, _meta = _build()
    row = vector[0]
    assert np.isclose(row[artifacts.feature_columns.index("news_sentiment_risk_score")], 0.44)
    assert np.isclose(row[artifacts.feature_columns.index("fear_greed_value")], 52.0)


def test_handles_missing_news_latest(monkeypatch):
    monkeypatch.setattr("aegis_alpha.decision_brain.feature_builder.LATEST_NEWS_PATH", Path("/tmp/aegis-missing-news.json"))
    artifacts = load_decision_brain_artifacts()
    assert artifacts is not None
    vector, meta = build_decision_brain_features(
        symbol="BTCUSDT",
        side="LONG",
        turbo_context=_turbo_context(),
        entry_quality_model=_entry_quality(),
        event_risk_auto=_event_risk_auto(),
        news_sentiment=None,
        feature_columns=artifacts.feature_columns,
    )
    assert vector.shape == (1, 112)
    assert meta["missing_features_count"] >= 0


def test_symbol_side_encoding_stable():
    artifacts = load_decision_brain_artifacts()
    assert artifacts is not None
    vector, _meta = _build()
    assert np.isclose(vector[0, artifacts.feature_columns.index("side_score")], 1.0)


def test_portfolio_missing_does_not_crash_and_is_marked_unavailable():
    artifacts = load_decision_brain_artifacts()
    assert artifacts is not None
    vector, _meta = _build()
    assert np.isclose(vector[0, artifacts.feature_columns.index("portfolio_context_available")], 0.0)


def test_feature_status_improves_when_required_groups_present():
    _vector, meta = _build()
    assert meta["feature_status"] in {"partial", "ok"}
    assert "market_mtf" in meta["available_feature_groups"]


def test_execute_false_and_production_allowed_false_always():
    row = evaluate_decision_brain_shadow("BTCUSDT", "LONG", _turbo_context(), _entry_quality(), _event_risk_auto(), _news())
    assert row["mode"] == "SHADOW"
    assert row["execute"] is False
    assert row["production_allowed"] is False
