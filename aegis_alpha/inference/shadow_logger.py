from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from aegis_alpha.config import REPO_ROOT


LOGGER = logging.getLogger(__name__)
SHADOW_LOG_DIR = REPO_ROOT / "aegis_alpha" / "logs" / "shadow"
MAX_DAILY_LOG_BYTES = int(os.getenv("AEGIS_SHADOW_MAX_DAILY_LOG_BYTES", str(64 * 1024 * 1024)))


def log_shadow_signal(
    shadow: dict[str, Any],
    *,
    price: float | None = None,
    model_paths: dict[str, str] | None = None,
    model_versions: dict[str, str] | None = None,
    request_source: str = "ml-v2-predict",
    log_dir: Path = SHADOW_LOG_DIR,
) -> Path:
    log_dir.mkdir(parents=True, exist_ok=True)
    day = datetime.now(timezone.utc).strftime("%Y%m%d")
    path = log_dir / f"shadow_signals_{day}.jsonl"
    row = {
        "logged_at": datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
        "timestamp": shadow.get("timestamp"),
        "symbol": shadow.get("symbol"),
        "price": price,
        "candidate": shadow.get("candidate"),
        "candidate_status": shadow.get("candidate_status"),
        "mode": shadow.get("mode"),
        "execute": False,
        "action": shadow.get("action"),
        "would_execute": bool(shadow.get("would_execute", False)),
        "reason": shadow.get("reason"),
        "size_mode": shadow.get("size_mode"),
        "position_fraction": float(shadow.get("position_fraction", 0.0) or 0.0),
        "edge_score_h12": float(shadow.get("edge_score_h12", 0.0) or 0.0),
        "tail_risk_score": float(shadow.get("tail_risk_score", 0.0) or 0.0),
        "regime": shadow.get("regime"),
        "risk_tier": shadow.get("risk_tier"),
        "model_paths": model_paths or {},
        "model_versions": model_versions or {},
        "request_source": request_source,
    }
    if MAX_DAILY_LOG_BYTES > 0 and path.exists() and path.stat().st_size >= MAX_DAILY_LOG_BYTES:
        return path
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, sort_keys=True) + "\n")
    return path


def safe_log_shadow_signal(*args: Any, **kwargs: Any) -> tuple[Path | None, str | None]:
    try:
        return log_shadow_signal(*args, **kwargs), None
    except Exception as exc:  # pragma: no cover - logging must not break inference
        LOGGER.exception("shadow signal logging failed")
        return None, repr(exc)
