from __future__ import annotations

import glob
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from aegis_alpha.turbo.config import DEFAULT_TURBO_CONFIG, get_runtime_turbo_config
from aegis_alpha.turbo.jsonl_utils import load_jsonl_safe


LOGGER = logging.getLogger(__name__)


def load_turbo_shadow_history(log_glob: str | None = None) -> list[dict[str, Any]]:
    pattern = log_glob or str(DEFAULT_TURBO_CONFIG.log_dir / "turbo_shadow_*.jsonl")
    rows: list[dict[str, Any]] = []
    for item in sorted(glob.glob(pattern)):
        path = Path(item)
        file_rows, errors = load_jsonl_safe(path)
        if errors:
            first = errors[0]
            skipped_count = sum(1 for error in errors if not bool(error.get("recovered")))
            recovered_count = len(errors) - skipped_count
            LOGGER.warning(
                "turbo_history_corrupt_lines_skipped skipped_count=%s recovered_count=%s file=%s first_line=%s first_error=%s first_error_preview=%r",
                skipped_count,
                recovered_count,
                path,
                first.get("line_no"),
                first.get("error"),
                first.get("preview"),
            )
        rows.extend(file_rows)
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
    cfg = get_runtime_turbo_config().risk
    history = rows if rows is not None else load_turbo_shadow_history()
    if cfg.max_turbo_trades_per_day <= 0:
        return False, None
    if count_today_turbo_signals(history) >= cfg.max_turbo_trades_per_day:
        return True, "max_turbo_trades_per_day"
    if count_recent_losses_if_evaluated(history) >= cfg.max_consecutive_losses:
        return True, "max_consecutive_losses"
    return False, None


def build_turbo_risk_status(rows: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    cfg = get_runtime_turbo_config().risk
    history = rows if rows is not None else load_turbo_shadow_history()
    blocked, reason = should_block_turbo_today(history)
    return {
        "blocked": bool(blocked),
        "reason": reason,
        "today_signal_count": count_today_turbo_signals(history),
        "recent_loss_count": count_recent_losses_if_evaluated(history),
        "max_turbo_trades_per_day": cfg.max_turbo_trades_per_day,
        "max_consecutive_losses": cfg.max_consecutive_losses,
    }
