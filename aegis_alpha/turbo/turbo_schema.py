from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from aegis_alpha.turbo.config import DEFAULT_TURBO_CONFIG, TURBO_MODE


def build_turbo_signal(
    *,
    symbol: str,
    action: str = "HOLD",
    would_execute: bool = False,
    reason: str = "not_evaluated",
    turbo_score: float = 0.0,
    confidence: str = "blocked",
    leverage_suggestion: float = 0.0,
    position_fraction: float = 0.0,
    votes: dict[str, int] | None = None,
    recent_scores: dict[str, float | None] | None = None,
    safe_context: dict[str, Any] | None = None,
    risk_guard: dict[str, Any] | None = None,
    raw: dict[str, Any] | None = None,
    gated: dict[str, Any] | None = None,
    enabled: bool = True,
    timestamp: str | None = None,
) -> dict[str, Any]:
    cfg = DEFAULT_TURBO_CONFIG
    action = str(action).upper()
    if action not in {"HOLD", "LONG", "SHORT"}:
        action = "HOLD"
    execute = False
    live_enabled = False
    production_allowed = False
    blocked = confidence == "blocked" or action == "HOLD"
    default_votes = votes or {"long": 0, "short": 0, "neutral": 0}
    default_scores = recent_scores or {
        "long_7d": None,
        "short_7d": None,
        "long_14d": None,
        "short_14d": None,
        "long_30d": None,
        "short_30d": None,
    }
    raw_payload = raw or {
        "action": action,
        "would_execute": bool(would_execute and action in {"LONG", "SHORT"}),
        "reason": str(reason),
        "turbo_score": float(turbo_score),
        "confidence": str(confidence),
        "leverage_suggestion": float(leverage_suggestion),
        "position_fraction": float(position_fraction),
        "votes": default_votes,
        "recent_scores": default_scores,
    }
    gated_payload = gated or {
        "action": action,
        "would_execute": bool(would_execute and action in {"LONG", "SHORT"}),
        "reason": str(reason),
        "blocked_by": None,
    }
    payload = {
        "mode": TURBO_MODE,
        "enabled": bool(enabled),
        "execute": execute,
        "live_enabled": live_enabled,
        "production_allowed": production_allowed,
        "symbol": symbol,
        "timestamp": timestamp or datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
        "action": action,
        "would_execute": bool(would_execute and action in {"LONG", "SHORT"}),
        "reason": str(reason),
        "turbo_score": float(turbo_score),
        "confidence": str(confidence if not blocked else "blocked"),
        "leverage_suggestion": float(leverage_suggestion if not blocked else 0.0),
        "position_fraction": float(position_fraction if not blocked else 0.0),
        "stop_roe": float(cfg.exits.hard_stop_roe),
        "take_profit_roe": float(cfg.exits.take_profit_roe),
        "trailing_activation_roe": float(cfg.exits.trailing_activation_roe),
        "trailing_callback_roe": float(cfg.exits.trailing_callback_roe),
        "votes": default_votes,
        "recent_scores": default_scores,
        "raw": raw_payload,
        "gated": gated_payload,
        "safe_context": safe_context or {
            "regime": "unknown",
            "tail_risk_score": None,
            "safe_action": "unknown",
            "safe_reason": "safe_context_unavailable",
        },
        "risk_guard": risk_guard or {"blocked": False, "reason": None},
    }
    payload["execute"] = False
    payload["live_enabled"] = False
    payload["production_allowed"] = False
    return payload
