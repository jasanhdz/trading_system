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
