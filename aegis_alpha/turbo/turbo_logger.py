from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from aegis_alpha.turbo.config import DEFAULT_TURBO_CONFIG


LOGGER = logging.getLogger(__name__)


def log_turbo_shadow(signal: dict[str, Any], log_dir: Path | None = None) -> Path:
    target_dir = log_dir or DEFAULT_TURBO_CONFIG.log_dir
    target_dir.mkdir(parents=True, exist_ok=True)
    day = datetime.now(timezone.utc).strftime("%Y%m%d")
    path = target_dir / f"turbo_shadow_{day}.jsonl"
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
    }
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, sort_keys=True) + "\n")
    return path


def safe_log_turbo_shadow(signal: dict[str, Any]) -> tuple[Path | None, str | None]:
    try:
        return log_turbo_shadow(signal), None
    except Exception as exc:  # pragma: no cover - logging must not break inference
        LOGGER.exception("turbo shadow logging failed")
        return None, repr(exc)
