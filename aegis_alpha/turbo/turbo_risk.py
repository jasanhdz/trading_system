from __future__ import annotations

import glob
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from aegis_alpha.turbo.config import DEFAULT_TURBO_CONFIG


def load_turbo_shadow_history(log_glob: str | None = None) -> list[dict[str, Any]]:
    pattern = log_glob or str(DEFAULT_TURBO_CONFIG.log_dir / "turbo_shadow_*.jsonl")
    rows: list[dict[str, Any]] = []
    for item in sorted(glob.glob(pattern)):
        path = Path(item)
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
    return rows


def _today_prefix() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


def count_today_turbo_signals(rows: list[dict[str, Any]] | None = None) -> int:
    history = rows if rows is not None else load_turbo_shadow_history()
    today = _today_prefix()
    return int(
        sum(
            1
            for row in history
            if str(row.get("timestamp", "")).startswith(today) and bool(row.get("would_execute", False))
        )
    )


def count_recent_losses_if_evaluated(rows: list[dict[str, Any]] | None = None) -> int:
    history = rows if rows is not None else load_turbo_shadow_history()
    losses = 0
    for row in reversed(history):
        result = row.get("paper_result") or row.get("evaluation")
        if not isinstance(result, dict) or "estimated_account_return" not in result:
            continue
        if float(result.get("estimated_account_return") or 0.0) < 0.0:
            losses += 1
        else:
            break
    return int(losses)


def should_block_turbo_today(rows: list[dict[str, Any]] | None = None) -> tuple[bool, str | None]:
    cfg = DEFAULT_TURBO_CONFIG.risk
    history = rows if rows is not None else load_turbo_shadow_history()
    if count_today_turbo_signals(history) >= cfg.max_turbo_trades_per_day:
        return True, "max_turbo_trades_per_day"
    if count_recent_losses_if_evaluated(history) >= cfg.max_consecutive_losses:
        return True, "max_consecutive_losses"
    return False, None


def build_turbo_risk_status(rows: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    history = rows if rows is not None else load_turbo_shadow_history()
    blocked, reason = should_block_turbo_today(history)
    return {
        "blocked": bool(blocked),
        "reason": reason,
        "today_signal_count": count_today_turbo_signals(history),
        "recent_loss_count": count_recent_losses_if_evaluated(history),
        "max_turbo_trades_per_day": DEFAULT_TURBO_CONFIG.risk.max_turbo_trades_per_day,
        "max_consecutive_losses": DEFAULT_TURBO_CONFIG.risk.max_consecutive_losses,
    }
