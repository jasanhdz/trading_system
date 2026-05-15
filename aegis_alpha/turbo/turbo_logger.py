from __future__ import annotations

import json
import logging
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from aegis_alpha.turbo.config import DEFAULT_TURBO_CONFIG


LOGGER = logging.getLogger(__name__)
_LOG_LOCK = threading.Lock()


def log_turbo_shadow(signal: dict[str, Any], log_dir: Path | None = None) -> Path:
    target_dir = log_dir or DEFAULT_TURBO_CONFIG.log_dir
    target_dir.mkdir(parents=True, exist_ok=True)
    day = datetime.now(timezone.utc).strftime("%Y%m%d")
    path = target_dir / f"turbo_shadow_{day}.jsonl"
    raw = signal.get("raw") or {}
    gated = signal.get("gated") or {}
    entry_quality_model = signal.get("entry_quality_model")
    event_risk_auto = signal.get("event_risk_auto")
    decision_brain = signal.get("decision_brain")
    row = {
        "logged_at": datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
        "timestamp": signal.get("timestamp"),
        "symbol": signal.get("symbol"),
        "mode": "TURBO_SHADOW",
        "action": signal.get("action"),
        "would_execute": bool(signal.get("would_execute", False)),
        "execute": False,
        "live_enabled": False,
        "production_allowed": False,
        "turbo_score": float(signal.get("turbo_score", 0.0) or 0.0),
        "confidence": signal.get("confidence"),
        "leverage_suggestion": float(signal.get("leverage_suggestion", 0.0) or 0.0),
        "position_fraction": float(signal.get("position_fraction", 0.0) or 0.0),
        "reason": signal.get("reason"),
        "votes": signal.get("votes", {}),
        "recent_scores": signal.get("recent_scores", {}),
        "safe_context": signal.get("safe_context", {}),
        "risk_guard": signal.get("risk_guard", {}),
        "freshness": signal.get("freshness", {}),
        "entry_quality_model": {
            "mode": entry_quality_model.get("mode"),
            "model_version": entry_quality_model.get("model_version"),
            "model_scope": entry_quality_model.get("model_scope"),
            "entry_quality_score": entry_quality_model.get("entry_quality_score"),
            "tail_risk_score": entry_quality_model.get("tail_risk_score"),
            "recommendation": entry_quality_model.get("recommendation"),
            "reason": entry_quality_model.get("reason"),
            "feature_status": entry_quality_model.get("feature_status"),
            "feature_parity_pct": entry_quality_model.get("feature_parity_pct"),
            "missing_features_count": entry_quality_model.get("missing_features_count"),
            "approximated_features": entry_quality_model.get("approximated_features"),
            "critical_missing_groups": entry_quality_model.get("critical_missing_groups"),
            "feature_build_latency_ms": entry_quality_model.get("feature_build_latency_ms"),
            "model_latency_ms": entry_quality_model.get("model_latency_ms"),
            "total_latency_ms": entry_quality_model.get("total_latency_ms"),
            "latency_ms": entry_quality_model.get("latency_ms"),
        } if isinstance(entry_quality_model, dict) else None,
        "event_risk_auto": {
            "mode": event_risk_auto.get("mode"),
            "suggested_mode": event_risk_auto.get("suggested_mode"),
            "confidence": event_risk_auto.get("confidence"),
            "reasons": event_risk_auto.get("reasons"),
            "btc_context": event_risk_auto.get("btc_context"),
            "eth_context": event_risk_auto.get("eth_context"),
            "market_context": event_risk_auto.get("market_context"),
            "execute": event_risk_auto.get("execute"),
            "production_allowed": event_risk_auto.get("production_allowed"),
            "does_not_change_event_risk_mode": event_risk_auto.get("does_not_change_event_risk_mode"),
            "latency_ms": event_risk_auto.get("latency_ms"),
            "last_update": event_risk_auto.get("last_update"),
        } if isinstance(event_risk_auto, dict) else None,
        "decision_brain": {
            "mode": decision_brain.get("mode"),
            "status": decision_brain.get("status"),
            "model_version": decision_brain.get("model_version"),
            "decision": decision_brain.get("decision"),
            "enter_now_prob": decision_brain.get("enter_now_prob"),
            "wait_confirmation_prob": decision_brain.get("wait_confirmation_prob"),
            "manual_only_prob": decision_brain.get("manual_only_prob"),
            "do_not_enter_prob": decision_brain.get("do_not_enter_prob"),
            "recommendation": decision_brain.get("recommendation"),
            "reason": decision_brain.get("reason"),
            "feature_status": decision_brain.get("feature_status"),
            "feature_parity_pct": decision_brain.get("feature_parity_pct"),
            "missing_features_count": decision_brain.get("missing_features_count"),
            "critical_missing_groups": decision_brain.get("critical_missing_groups"),
            "available_feature_groups": decision_brain.get("available_feature_groups"),
            "approximated_features": decision_brain.get("approximated_features"),
            "feature_group_coverage_pct": decision_brain.get("feature_group_coverage_pct"),
            "feature_build_latency_ms": decision_brain.get("feature_build_latency_ms"),
            "model_latency_ms": decision_brain.get("model_latency_ms"),
            "total_latency_ms": decision_brain.get("total_latency_ms"),
            "latency_ms": decision_brain.get("latency_ms"),
            "execute": decision_brain.get("execute"),
            "production_allowed": decision_brain.get("production_allowed"),
        } if isinstance(decision_brain, dict) else None,
        "stale": not bool((signal.get("freshness") or {}).get("is_fresh", True)),
        "raw": raw,
        "gated": gated,
        "raw_action": raw.get("action"),
        "raw_would_execute": bool(raw.get("would_execute", False)),
        "raw_reason": raw.get("reason"),
        "raw_turbo_score": float(raw.get("turbo_score", 0.0) or 0.0),
        "raw_confidence": raw.get("confidence"),
        "raw_leverage_suggestion": float(raw.get("leverage_suggestion", 0.0) or 0.0),
        "raw_position_fraction": float(raw.get("position_fraction", 0.0) or 0.0),
        "raw_votes": raw.get("votes", {}),
        "raw_recent_scores": raw.get("recent_scores", {}),
        "gated_action": gated.get("action"),
        "gated_would_execute": bool(gated.get("would_execute", False)),
        "gated_reason": gated.get("reason"),
        "gated_blocked_by": gated.get("blocked_by"),
    }
    line = json.dumps(row, separators=(",", ":"), ensure_ascii=False) + "\n"
    with _LOG_LOCK:
        with path.open("a", encoding="utf-8") as f:
            f.write(line)
    return path


def safe_log_turbo_shadow(signal: dict[str, Any]) -> tuple[Path | None, str | None]:
    try:
        return log_turbo_shadow(signal), None
    except Exception as exc:  # pragma: no cover - logging must not break inference
        LOGGER.exception("turbo shadow logging failed")
        return None, repr(exc)
