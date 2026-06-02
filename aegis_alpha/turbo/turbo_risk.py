from __future__ import annotations

import glob
import logging
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from aegis_alpha.turbo.config import DEFAULT_TURBO_CONFIG, get_runtime_turbo_config
from aegis_alpha.turbo.jsonl_utils import load_jsonl_safe


LOGGER = logging.getLogger(__name__)
_RISK_STATUS_CACHE: dict[str, Any] = {"expires_at": 0.0, "status": None}


def _risk_cache_ttl_seconds() -> float:
    try:
        return max(float(os.environ.get("AEGIS_TURBO_RISK_CACHE_SECONDS", "30")), 0.0)
    except ValueError:
        return 30.0


def _risk_history_glob() -> str:
    configured = os.environ.get("AEGIS_TURBO_RISK_HISTORY_GLOB")
    if configured:
        return configured
    today = datetime.now(timezone.utc).strftime("%Y%m%d")
    return str(DEFAULT_TURBO_CONFIG.log_dir / f"turbo_shadow_{today}.jsonl")


def _trade_events_glob() -> str:
    configured = os.environ.get("AEGIS_TURBO_TRADE_EVENTS_GLOB")
    if configured:
        return configured
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    repo_root = Path(__file__).resolve().parents[2]
    return str(repo_root / "binance-futures-bot-ts" / "logs" / "aegis" / f"turbo_trade_events_{today}.jsonl")


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


def load_turbo_trade_events(log_glob: str | None = None) -> list[dict[str, Any]]:
    pattern = log_glob or _trade_events_glob()
    rows: list[dict[str, Any]] = []
    for item in sorted(glob.glob(pattern)):
        file_rows, errors = load_jsonl_safe(Path(item))
        if errors:
            LOGGER.warning("turbo_trade_event_corrupt_lines_skipped file=%s count=%s", item, len(errors))
        rows.extend(file_rows)
    return rows


def _today_prefix() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


def count_today_turbo_signals(rows: list[dict[str, Any]] | None = None) -> int:
    history = rows if rows is not None else load_turbo_shadow_history(_risk_history_glob())
    today = _today_prefix()
    return int(
        sum(
            1
            for row in history
            if str(row.get("timestamp", "")).startswith(today) and bool(row.get("would_execute", False))
        )
    )


def count_today_turbo_opened_trades(rows: list[dict[str, Any]] | None = None) -> int:
    events = rows if rows is not None else load_turbo_trade_events()
    today = _today_prefix()
    trade_ids: set[str] = set()
    fallback_count = 0
    for row in events:
        if not str(row.get("timestamp", "")).startswith(today):
            continue
        if str(row.get("event", "")) != "POSITION_CONFIRMED":
            continue
        trade_id = str(row.get("trade_id") or "").strip()
        if trade_id:
            trade_ids.add(trade_id)
        else:
            fallback_count += 1
    return len(trade_ids) + fallback_count


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


def should_block_turbo_today(
    rows: list[dict[str, Any]] | None = None,
    trade_events: list[dict[str, Any]] | None = None,
) -> tuple[bool, str | None]:
    cfg = get_runtime_turbo_config().risk
    history = rows if rows is not None else load_turbo_shadow_history(_risk_history_glob())
    if cfg.max_turbo_trades_per_day <= 0:
        return False, None
    if count_today_turbo_opened_trades(trade_events) >= cfg.max_turbo_trades_per_day:
        return True, "max_turbo_trades_per_day"
    if count_recent_losses_if_evaluated(history) >= cfg.max_consecutive_losses:
        return True, "max_consecutive_losses"
    return False, None


def build_turbo_risk_status(
    rows: list[dict[str, Any]] | None = None,
    trade_events: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    if rows is None:
        now = time.time()
        cached_status = _RISK_STATUS_CACHE.get("status")
        if cached_status is not None and now < float(_RISK_STATUS_CACHE.get("expires_at") or 0.0):
            return dict(cached_status)

    cfg = get_runtime_turbo_config().risk
    history = rows if rows is not None else load_turbo_shadow_history(_risk_history_glob())
    events = trade_events if trade_events is not None else load_turbo_trade_events()
    opened_trade_count = count_today_turbo_opened_trades(events)
    blocked, reason = should_block_turbo_today(history, events)
    status = {
        "blocked": bool(blocked),
        "reason": reason,
        "count_source": "position_confirmed",
        "today_trade_count": opened_trade_count,
        "current_count": opened_trade_count,
        "today_signal_count": count_today_turbo_signals(history),
        "recent_loss_count": count_recent_losses_if_evaluated(history),
        "max_turbo_trades_per_day": cfg.max_turbo_trades_per_day,
        "max_consecutive_losses": cfg.max_consecutive_losses,
    }
    if rows is None:
        _RISK_STATUS_CACHE.update({
            "expires_at": time.time() + _risk_cache_ttl_seconds(),
            "status": dict(status),
        })
    return status
