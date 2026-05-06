#!/usr/bin/env python3
from __future__ import annotations

import argparse
import glob
import json
import sys
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from aegis_alpha.edge.common import profit_factor, safe_float  # noqa: E402
from aegis_alpha.signals.common import load_signal_market  # noqa: E402
from aegis_alpha.turbo.jsonl_utils import load_jsonl_safe  # noqa: E402


DEFAULT_CONFIG = "aegis_alpha/configs/base.yaml"
DEFAULT_SAFE_GLOB = "aegis_alpha/logs/shadow/shadow_signals_*.jsonl"
DEFAULT_TURBO_GLOB = "aegis_alpha/logs/turbo/turbo_shadow_*.jsonl"
DEFAULT_OUTPUT_DIR = Path("aegis_alpha/logs/shadow")
HORIZONS = (3, 6, 12, 24, 48)


def _utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _parse_bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def _parse_dt(value: Any) -> datetime | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    formats = (
        "%Y%m%dT%H%M%SZ",
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%dT%H:%M:%S",
        "%Y-%m-%dT%H:%M:%SZ",
        "%Y-%m-%dT%H:%M:%S%z",
    )
    for fmt in formats:
        try:
            dt = datetime.strptime(text, fmt)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt.astimezone(timezone.utc)
        except ValueError:
            pass
    try:
        dt = datetime.fromisoformat(text.replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)
    except ValueError:
        return None


def _event_dt(row: dict[str, Any]) -> datetime | None:
    return _parse_dt(row.get("logged_at")) or _parse_dt(row.get("timestamp"))


def _read_jsonl(pattern: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    for item in sorted(glob.glob(pattern)):
        path = Path(item)
        file_rows, file_errors = load_jsonl_safe(path)
        for row in file_rows:
            row["_log_file"] = str(path)
        rows.extend(file_rows)
        errors.extend(file_errors)
    return rows, errors


def _filter_rows(rows: list[dict[str, Any]], symbol: str, since: datetime) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in rows:
        if symbol and str(row.get("symbol", "")).upper() != symbol.upper():
            continue
        dt = _event_dt(row)
        if dt is None or dt < since:
            continue
        row["_event_time"] = dt.isoformat()
        out.append(row)
    return out


def _dist(rows: list[dict[str, Any]], key: str) -> dict[str, int]:
    return dict(Counter(str(row.get(key, "unknown")) for row in rows))


def _num_values(rows: list[dict[str, Any]], key: str) -> np.ndarray:
    values: list[float] = []
    for row in rows:
        try:
            value = float(row.get(key))
        except (TypeError, ValueError):
            continue
        if np.isfinite(value):
            values.append(value)
    return np.asarray(values, dtype=np.float32)


def _stats(values: np.ndarray) -> dict[str, float | None]:
    if len(values) == 0:
        return {"avg": None, "min": None, "max": None, "p50": None, "p90": None, "p95": None}
    return {
        "avg": safe_float(np.mean(values)),
        "min": safe_float(np.min(values)),
        "max": safe_float(np.max(values)),
        "p50": safe_float(np.quantile(values, 0.50)),
        "p90": safe_float(np.quantile(values, 0.90)),
        "p95": safe_float(np.quantile(values, 0.95)),
    }


def _action_counts(rows: list[dict[str, Any]], actions: tuple[str, ...]) -> dict[str, int]:
    counts = Counter(str(row.get("action", "unknown")).upper() for row in rows)
    return {action: int(counts.get(action, 0)) for action in actions}


def _raw_view(row: dict[str, Any]) -> dict[str, Any]:
    raw = row.get("raw") if isinstance(row.get("raw"), dict) else {}
    return {
        "action": str(raw.get("action", row.get("raw_action", row.get("action", "HOLD")))).upper(),
        "would_execute": bool(raw.get("would_execute", row.get("raw_would_execute", row.get("would_execute", False)))),
        "reason": raw.get("reason", row.get("raw_reason", row.get("reason"))),
        "turbo_score": raw.get("turbo_score", row.get("raw_turbo_score", row.get("turbo_score", 0.0))),
        "confidence": raw.get("confidence", row.get("raw_confidence", row.get("confidence", "blocked"))),
        "leverage_suggestion": raw.get("leverage_suggestion", row.get("raw_leverage_suggestion", row.get("leverage_suggestion", 0.0))),
        "position_fraction": raw.get("position_fraction", row.get("raw_position_fraction", row.get("position_fraction", 0.0))),
        "votes": raw.get("votes", row.get("raw_votes", row.get("votes", {}))),
        "recent_scores": raw.get("recent_scores", row.get("raw_recent_scores", row.get("recent_scores", {}))),
    }


def _gated_view(row: dict[str, Any]) -> dict[str, Any]:
    gated = row.get("gated") if isinstance(row.get("gated"), dict) else {}
    return {
        "action": str(gated.get("action", row.get("gated_action", row.get("action", "HOLD")))).upper(),
        "would_execute": bool(gated.get("would_execute", row.get("gated_would_execute", row.get("would_execute", False)))),
        "reason": gated.get("reason", row.get("gated_reason", row.get("reason"))),
        "blocked_by": gated.get("blocked_by", row.get("gated_blocked_by")),
    }


def _safe_summary(rows: list[dict[str, Any]], raw_found: bool) -> dict[str, Any]:
    return {
        "no_safe_logs_found": not raw_found,
        "total_evaluations": int(len(rows)),
        "action_counts": _action_counts(rows, ("LONG", "SHORT", "HOLD", "CLOSE")),
        "would_execute_true": int(sum(1 for row in rows if bool(row.get("would_execute")))),
        "would_execute_false": int(sum(1 for row in rows if not bool(row.get("would_execute")))),
        "reasons_distribution": _dist(rows, "reason"),
        "regimes_distribution": _dist(rows, "regime"),
        "size_mode_distribution": _dist(rows, "size_mode"),
        "risk_tier_distribution": _dist(rows, "risk_tier"),
        "edge_score_h12": _stats(_num_values(rows, "edge_score_h12")),
        "tail_risk_score": _stats(_num_values(rows, "tail_risk_score")),
    }


def _turbo_summary(rows: list[dict[str, Any]], raw_found: bool) -> dict[str, Any]:
    raw_rows = [_raw_view(row) for row in rows]
    gated_rows = [_gated_view(row) for row in rows]
    long_votes = np.asarray([float((row.get("votes") or {}).get("long", 0) or 0) for row in raw_rows], dtype=np.float32)
    short_votes = np.asarray([float((row.get("votes") or {}).get("short", 0) or 0) for row in raw_rows], dtype=np.float32)
    neutral_votes = np.asarray([float((row.get("votes") or {}).get("neutral", 0) or 0) for row in raw_rows], dtype=np.float32)
    raw_long = sum(1 for row in raw_rows if row["action"] == "LONG")
    raw_short = sum(1 for row in raw_rows if row["action"] == "SHORT")
    raw_hold = sum(1 for row in raw_rows if row["action"] == "HOLD")
    raw_would = sum(1 for row in raw_rows if row["would_execute"])
    gated_long = sum(1 for row in gated_rows if row["action"] == "LONG")
    gated_short = sum(1 for row in gated_rows if row["action"] == "SHORT")
    gated_hold = sum(1 for row in gated_rows if row["action"] == "HOLD")
    gated_would = sum(1 for row in gated_rows if row["would_execute"])
    safe_blocked_raw_long = sum(1 for raw, gated in zip(raw_rows, gated_rows) if raw["action"] == "LONG" and raw["would_execute"] and gated.get("blocked_by") in {"safe_regime", "safe_tail_risk"})
    safe_blocked_raw_short = sum(1 for raw, gated in zip(raw_rows, gated_rows) if raw["action"] == "SHORT" and raw["would_execute"] and gated.get("blocked_by") in {"safe_regime", "safe_tail_risk"})
    return {
        "no_turbo_logs_found": not raw_found,
        "total_evaluations": int(len(rows)),
        "action_counts": _action_counts(rows, ("LONG", "SHORT", "HOLD")),
        "would_execute_true": int(sum(1 for row in rows if bool(row.get("would_execute")))),
        "would_execute_false": int(sum(1 for row in rows if not bool(row.get("would_execute")))),
        "reasons_distribution": _dist(rows, "reason"),
        "confidence_distribution": _dist(rows, "confidence"),
        "leverage_suggestion_distribution": _dist(rows, "leverage_suggestion"),
        "position_fraction_distribution": _dist(rows, "position_fraction"),
        "votes": {
            "avg_long_votes": safe_float(np.mean(long_votes)) if len(long_votes) else 0.0,
            "avg_short_votes": safe_float(np.mean(short_votes)) if len(short_votes) else 0.0,
            "avg_neutral_votes": safe_float(np.mean(neutral_votes)) if len(neutral_votes) else 0.0,
            "count_long_votes_gte_2": int(np.sum(long_votes >= 2)) if len(long_votes) else 0,
            "count_short_votes_gte_2": int(np.sum(short_votes >= 2)) if len(short_votes) else 0,
        },
        "turbo_score": _stats(_num_values(rows, "turbo_score")),
        "safe_block_counts": {
            "safe_regime_block": int(sum(1 for row in rows if row.get("reason") == "safe_regime_block")),
            "safe_tail_risk_block": int(sum(1 for row in rows if row.get("reason") == "safe_tail_risk_block")),
            "insufficient_recent_model_agreement": int(sum(1 for row in rows if row.get("reason") == "insufficient_recent_model_agreement")),
            "turbo_score_below_threshold": int(sum(1 for row in rows if row.get("reason") in {"turbo_score_below_threshold", "turbo_score_below_shadow_threshold"})),
            "short_disabled_in_turbo_v010": int(sum(1 for row in rows if row.get("reason") == "short_disabled_in_turbo_v010")),
            "max_trades_per_day_block": int(sum(1 for row in rows if "max_turbo_trades_per_day" in str(row.get("reason")))),
            "other": int(sum(1 for row in rows if row.get("reason") not in {
                "safe_regime_block",
                "safe_tail_risk_block",
                "insufficient_recent_model_agreement",
                "turbo_score_below_threshold",
                "turbo_score_below_shadow_threshold",
                "short_disabled_in_turbo_v010",
            } and "max_turbo_trades_per_day" not in str(row.get("reason")))),
        },
        "raw_vs_gated": {
            "raw_long_count": int(raw_long),
            "raw_short_count": int(raw_short),
            "raw_hold_count": int(raw_hold),
            "raw_would_execute_count": int(raw_would),
            "gated_long_count": int(gated_long),
            "gated_short_count": int(gated_short),
            "gated_hold_count": int(gated_hold),
            "gated_would_execute_count": int(gated_would),
            "safe_blocked_raw_long_count": int(safe_blocked_raw_long),
            "safe_blocked_raw_short_count": int(safe_blocked_raw_short),
            "raw_to_gated_conversion_rate": safe_float(gated_would / raw_would) if raw_would else 0.0,
            "main_gating_reasons": dict(Counter(str(row.get("blocked_by") or "none") for row in gated_rows)),
        },
    }


def _market_context(config_path: str) -> tuple[dict[str, int], np.ndarray] | None:
    try:
        market = load_signal_market(config_path)
        return {str(ts): idx for idx, ts in enumerate(market.timestamps)}, market.close
    except Exception:
        return None


def _future_stats(close: np.ndarray, idx: int, horizon: int, action: str) -> dict[str, Any]:
    if idx < 0 or idx + horizon >= len(close):
        return {"status": "pending", "pending_reason": "insufficient_future_candles"}
    entry = float(close[idx])
    future = close[idx + 1 : idx + horizon + 1]
    if entry <= 0.0 or len(future) < horizon:
        return {"status": "pending", "pending_reason": "insufficient_future_candles"}
    if action == "SHORT":
        path = entry / np.maximum(future, 1e-10) - 1.0
        ret = entry / float(close[idx + horizon]) - 1.0
    else:
        path = future / entry - 1.0
        ret = float(close[idx + horizon]) / entry - 1.0
    return {
        "status": "complete",
        "future_return": safe_float(ret),
        "MFE": safe_float(np.max(path)),
        "MAE": safe_float(max(0.0, -np.min(path))),
    }


def _evaluate_entry(row: dict[str, Any], market_ctx: tuple[dict[str, int], np.ndarray] | None, is_turbo: bool) -> dict[str, Any]:
    entry = dict(row)
    entry.pop("_log_file", None)
    entry.pop("_line_no", None)
    if market_ctx is None:
        entry["future_evaluation"] = {"status": "pending", "pending_reason": "market_data_unavailable"}
        entry["quality_label"] = "PENDING"
        return entry
    timestamp_to_idx, close = market_ctx
    idx = timestamp_to_idx.get(str(row.get("timestamp")))
    if idx is None:
        entry["future_evaluation"] = {"status": "pending", "pending_reason": "timestamp_not_found"}
        entry["quality_label"] = "PENDING"
        return entry

    action = str(row.get("action", "LONG")).upper()
    future: dict[str, Any] = {}
    complete_12 = None
    for horizon in HORIZONS:
        stats = _future_stats(close, idx, horizon, action)
        future[f"h{horizon}"] = stats
        if horizon == 12:
            complete_12 = stats if stats.get("status") == "complete" else None

    if is_turbo:
        leverage = float(row.get("leverage_suggestion", 0.0) or 0.0)
        best_mfe_roe = 0.0
        worst_mae_roe = 0.0
        for horizon, stats in future.items():
            if stats.get("status") != "complete":
                continue
            stats["estimated_roe"] = safe_float(float(stats["future_return"]) * leverage * 100.0)
            stats["mfe_roe"] = safe_float(float(stats["MFE"]) * leverage * 100.0)
            stats["mae_roe"] = safe_float(-float(stats["MAE"]) * leverage * 100.0)
            best_mfe_roe = max(best_mfe_roe, float(stats["mfe_roe"]))
            worst_mae_roe = min(worst_mae_roe, float(stats["mae_roe"]))
        stop_roe = float(row.get("stop_roe", -15.0) or -15.0)
        take_profit_roe = float(row.get("take_profit_roe", 25.0) or 25.0)
        trailing_activation_roe = float(row.get("trailing_activation_roe", 15.0) or 15.0)
        future["turbo_path_flags"] = {
            "hit_stop_possible": bool(worst_mae_roe <= stop_roe),
            "hit_take_profit_possible": bool(best_mfe_roe >= take_profit_roe),
            "hit_trailing_activation_possible": bool(best_mfe_roe >= trailing_activation_roe),
            "best_mfe_roe": safe_float(best_mfe_roe),
            "worst_mae_roe": safe_float(worst_mae_roe),
        }
        if complete_12 is None:
            quality = "PENDING"
        elif worst_mae_roe <= stop_roe or float(complete_12["MAE"]) > float(complete_12["MFE"]):
            quality = "BAD_CANDIDATE"
        elif best_mfe_roe >= trailing_activation_roe or float(future["h12"].get("estimated_roe", 0.0)) >= 10.0 or float(complete_12["MFE"]) > float(complete_12["MAE"]):
            quality = "GOOD_CANDIDATE"
        else:
            quality = "MIXED_CANDIDATE"
    else:
        if complete_12 is None:
            quality = "PENDING"
        elif float(complete_12["future_return"]) > 0.0 and float(complete_12["MFE"]) > abs(float(complete_12["MAE"])) and float(row.get("tail_risk_score", 1.0) or 1.0) <= 0.50:
            quality = "GOOD_CANDIDATE"
        elif float(complete_12["future_return"]) < 0.0 and abs(float(complete_12["MAE"])) > float(complete_12["MFE"]):
            quality = "BAD_CANDIDATE"
        else:
            quality = "MIXED_CANDIDATE"

    entry["future_evaluation"] = future
    entry["quality_label"] = quality
    return entry


def _safe_potential(rows: list[dict[str, Any]], market_ctx: tuple[dict[str, int], np.ndarray] | None, include_pending: bool) -> list[dict[str, Any]]:
    out = []
    for row in rows:
        if row.get("action") == "LONG" and bool(row.get("would_execute")) and not bool(row.get("execute")):
            evaluated = _evaluate_entry(row, market_ctx, is_turbo=False)
            if include_pending or evaluated.get("quality_label") != "PENDING":
                out.append(evaluated)
    return out


def _turbo_potential(rows: list[dict[str, Any]], market_ctx: tuple[dict[str, int], np.ndarray] | None, include_pending: bool) -> list[dict[str, Any]]:
    out = []
    for row in rows:
        if row.get("action") in {"LONG", "SHORT"} and bool(row.get("would_execute")) and not bool(row.get("execute")):
            evaluated = _evaluate_entry(row, market_ctx, is_turbo=True)
            if include_pending or evaluated.get("quality_label") != "PENDING":
                out.append(evaluated)
    return out


def _turbo_raw_potential(rows: list[dict[str, Any]], market_ctx: tuple[dict[str, int], np.ndarray] | None, include_pending: bool) -> list[dict[str, Any]]:
    out = []
    for row in rows:
        raw = _raw_view(row)
        if raw["action"] not in {"LONG", "SHORT"} or not raw["would_execute"]:
            continue
        raw_row = dict(row)
        raw_row["action"] = raw["action"]
        raw_row["would_execute"] = raw["would_execute"]
        raw_row["reason"] = raw["reason"]
        raw_row["turbo_score"] = raw["turbo_score"]
        raw_row["confidence"] = raw["confidence"]
        raw_row["leverage_suggestion"] = raw["leverage_suggestion"]
        raw_row["position_fraction"] = raw["position_fraction"]
        raw_row["votes"] = raw["votes"]
        raw_row["recent_scores"] = raw["recent_scores"]
        evaluated = _evaluate_entry(raw_row, market_ctx, is_turbo=True)
        evaluated["gated"] = _gated_view(row)
        evaluated["raw"] = raw
        if include_pending or evaluated.get("quality_label") != "PENDING":
            out.append(evaluated)
    return out


def _turbo_safe_blocked_entries(rows: list[dict[str, Any]], market_ctx: tuple[dict[str, int], np.ndarray] | None, include_pending: bool) -> list[dict[str, Any]]:
    out = []
    for row in rows:
        raw = _raw_view(row)
        gated = _gated_view(row)
        if raw["action"] not in {"LONG", "SHORT"} or not raw["would_execute"]:
            continue
        if gated.get("blocked_by") not in {"safe_regime", "safe_tail_risk"}:
            continue
        raw_row = dict(row)
        raw_row["action"] = raw["action"]
        raw_row["would_execute"] = raw["would_execute"]
        raw_row["reason"] = raw["reason"]
        raw_row["turbo_score"] = raw["turbo_score"]
        raw_row["confidence"] = raw["confidence"]
        raw_row["leverage_suggestion"] = raw["leverage_suggestion"]
        raw_row["position_fraction"] = raw["position_fraction"]
        raw_row["votes"] = raw["votes"]
        raw_row["recent_scores"] = raw["recent_scores"]
        evaluated = _evaluate_entry(raw_row, market_ctx, is_turbo=True)
        evaluated["gated"] = gated
        evaluated["raw"] = raw
        if include_pending or evaluated.get("quality_label") != "PENDING":
            out.append(evaluated)
    return out


def _safe_near_entries(rows: list[dict[str, Any]], limit: int = 5) -> list[dict[str, Any]]:
    def key(row: dict[str, Any]) -> tuple[float, float]:
        edge = float(row.get("edge_score_h12", -999.0) or -999.0)
        tail = float(row.get("tail_risk_score", 999.0) or 999.0)
        return edge, -tail

    selected = sorted(rows, key=key, reverse=True)[:limit]
    return [
        {
            "timestamp": row.get("timestamp"),
            "logged_at": row.get("logged_at"),
            "symbol": row.get("symbol"),
            "action": row.get("action"),
            "reason": row.get("reason"),
            "edge_score_h12": row.get("edge_score_h12"),
            "tail_risk_score": row.get("tail_risk_score"),
            "regime": row.get("regime"),
            "size_mode": row.get("size_mode"),
            "position_fraction": row.get("position_fraction"),
        }
        for row in selected
    ]


def _turbo_near_entries(rows: list[dict[str, Any]], limit: int = 10) -> list[dict[str, Any]]:
    def key(row: dict[str, Any]) -> tuple[float, int]:
        raw = _raw_view(row)
        votes = raw.get("votes") or {}
        agreement = max(int(votes.get("long", 0) or 0), int(votes.get("short", 0) or 0))
        score = float(raw.get("turbo_score", 0.0) or 0.0)
        return score, agreement

    selected = sorted(rows, key=key, reverse=True)[:limit]
    return [
        {
            "raw_action": _raw_view(row).get("action"),
            "raw_would_execute": _raw_view(row).get("would_execute"),
            "raw_reason": _raw_view(row).get("reason"),
            "timestamp": row.get("timestamp"),
            "logged_at": row.get("logged_at"),
            "symbol": row.get("symbol"),
            "action": row.get("action"),
            "reason": row.get("reason"),
            "turbo_score": _raw_view(row).get("turbo_score"),
            "confidence": _raw_view(row).get("confidence"),
            "leverage_suggestion": _raw_view(row).get("leverage_suggestion"),
            "position_fraction": _raw_view(row).get("position_fraction"),
            "votes": _raw_view(row).get("votes", {}),
            "recent_scores": _raw_view(row).get("recent_scores", {}),
            "gated": _gated_view(row),
            "safe_context": row.get("safe_context", {}),
        }
        for row in selected
    ]


def _quality_counts(entries: list[dict[str, Any]]) -> dict[str, int]:
    counts = Counter(str(entry.get("quality_label", "UNKNOWN")) for entry in entries)
    return {key: int(counts.get(key, 0)) for key in ("GOOD_CANDIDATE", "MIXED_CANDIDATE", "BAD_CANDIDATE", "PENDING")}


def _diagnosis(
    safe_rows: list[dict[str, Any]],
    turbo_rows: list[dict[str, Any]],
    safe_entries: list[dict[str, Any]],
    turbo_entries: list[dict[str, Any]],
    turbo_raw_entries: list[dict[str, Any]] | None = None,
    turbo_safe_blocked_entries: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    turbo_raw_entries = turbo_raw_entries or []
    turbo_safe_blocked_entries = turbo_safe_blocked_entries or []
    safe_reasons = Counter(str(row.get("reason", "unknown")) for row in safe_rows)
    turbo_reasons = Counter(str(row.get("reason", "unknown")) for row in turbo_rows)
    safe_blocked_too_much = bool(safe_rows and not safe_entries and sum(1 for row in safe_rows if row.get("action") == "HOLD") / len(safe_rows) > 0.95)
    turbo_agreements = sum(1 for row in turbo_rows if max(int((_raw_view(row).get("votes") or {}).get("long", 0) or 0), int((_raw_view(row).get("votes") or {}).get("short", 0) or 0)) >= 2)
    turbo_blocked_too_much = bool(turbo_rows and not turbo_entries and turbo_agreements > 0)
    safe_good = _quality_counts(safe_entries)["GOOD_CANDIDATE"]
    turbo_good = _quality_counts(turbo_entries)["GOOD_CANDIDATE"]
    if safe_good > turbo_good:
        best = "safe"
    elif turbo_good > safe_good:
        best = "turbo"
    elif safe_entries or turbo_entries:
        best = "pending"
    else:
        best = "none"
    recommendations: list[str] = []
    if safe_blocked_too_much:
        recommendations.append("Safe appears highly conservative in this window; keep collecting shadow data before changing thresholds.")
    if turbo_blocked_too_much:
        recommendations.append("Turbo has recent-model agreement but is fully gated; review Safe veto interaction before changing risk.")
    raw_quality = _quality_counts(turbo_raw_entries)
    safe_blocked_quality = _quality_counts(turbo_safe_blocked_entries)
    has_raw_logging = any("raw_turbo_score" in row or "raw" in row for row in turbo_rows)
    lacks_raw = (not has_raw_logging) and any(row.get("reason") == "safe_regime_block" and max(int((_raw_view(row).get("votes") or {}).get("long", 0) or 0), int((_raw_view(row).get("votes") or {}).get("short", 0) or 0)) >= 2 for row in turbo_rows)
    if lacks_raw:
        recommendations.append("Implement Turbo v0.1.1 raw_vs_gated logging: store raw_turbo_action, raw_turbo_score, raw_votes before Safe veto.")
    if not safe_rows and not turbo_rows:
        recommendations.append("No recent shadow rows found; verify API traffic and log paths.")
    elif not safe_entries and not turbo_entries and not turbo_raw_entries:
        recommendations.append("No potential entries in the review window; collect more time or inspect raw-vs-gated signal intent.")
    if safe_blocked_quality["GOOD_CANDIDATE"] > 0:
        recommendations.append("Safe gate may be too strict for Turbo research.")
    if safe_blocked_quality["BAD_CANDIDATE"] > 0 and safe_blocked_quality["GOOD_CANDIDATE"] == 0:
        recommendations.append("Safe gate protected correctly in this review window.")
    if raw_quality["PENDING"] > 0 and raw_quality["GOOD_CANDIDATE"] == 0 and raw_quality["BAD_CANDIDATE"] == 0:
        recommendations.append("Need more future candles.")
    return {
        "did_safe_find_entries": bool(safe_entries),
        "did_turbo_find_entries": bool(turbo_entries),
        "turbo_blocked_too_much": turbo_blocked_too_much,
        "safe_blocked_too_much": safe_blocked_too_much,
        "best_signal_source": best,
        "main_block_reasons": [item[0] for item in (safe_reasons + turbo_reasons).most_common(5)],
        "turbo_raw_quality": raw_quality,
        "turbo_safe_blocked_quality": safe_blocked_quality,
        "recommendations": recommendations,
    }


def review_shadow_last_hours(hours: float, symbol: str, include_pending: bool, output_json: str | None = None) -> dict[str, Any]:
    now = datetime.now(timezone.utc)
    since = now - timedelta(hours=float(hours))
    raw_safe, safe_corrupt_errors = _read_jsonl(DEFAULT_SAFE_GLOB)
    raw_turbo, turbo_corrupt_errors = _read_jsonl(DEFAULT_TURBO_GLOB)
    safe_rows = _filter_rows(raw_safe, symbol, since)
    turbo_rows = _filter_rows(raw_turbo, symbol, since)
    market_ctx = _market_context(DEFAULT_CONFIG) if safe_rows or turbo_rows else None
    safe_entries = _safe_potential(safe_rows, market_ctx, include_pending)
    turbo_entries = _turbo_potential(turbo_rows, market_ctx, include_pending)
    turbo_raw_entries = _turbo_raw_potential(turbo_rows, market_ctx, include_pending)
    turbo_safe_blocked_entries = _turbo_safe_blocked_entries(turbo_rows, market_ctx, include_pending)
    report = {
        "schema_version": "aegis_shadow_last_hours_review_v1",
        "hours": float(hours),
        "symbol": symbol,
        "generated_at": now.strftime("%Y%m%dT%H%M%SZ"),
        "window_start": since.strftime("%Y%m%dT%H%M%SZ"),
        "corrupted_lines": {
            "safe_count": int(len(safe_corrupt_errors)),
            "safe_files": sorted({str(error.get("path")) for error in safe_corrupt_errors}),
            "turbo_count": int(len(turbo_corrupt_errors)),
            "turbo_files": sorted({str(error.get("path")) for error in turbo_corrupt_errors}),
            "first_turbo_error": turbo_corrupt_errors[0] if turbo_corrupt_errors else None,
        },
        "safe_summary": _safe_summary(safe_rows, bool(raw_safe)),
        "turbo_summary": _turbo_summary(turbo_rows, bool(raw_turbo)),
        "safe_potential_entries": safe_entries,
        "turbo_potential_entries": turbo_entries,
        "turbo_raw_potential_entries": turbo_raw_entries,
        "turbo_gated_potential_entries": turbo_entries,
        "turbo_safe_blocked_entries": turbo_safe_blocked_entries,
        "safe_near_entries": _safe_near_entries(safe_rows),
        "turbo_near_entries": _turbo_near_entries(turbo_rows),
        "signal_quality": {
            "safe": _quality_counts(safe_entries),
            "turbo": _quality_counts(turbo_entries),
            "turbo_raw": _quality_counts(turbo_raw_entries),
            "turbo_safe_blocked": _quality_counts(turbo_safe_blocked_entries),
            "combined": _quality_counts(safe_entries + turbo_entries + turbo_raw_entries),
        },
    }
    report["combined_diagnosis"] = _diagnosis(safe_rows, turbo_rows, safe_entries, turbo_entries, turbo_raw_entries, turbo_safe_blocked_entries)
    output_path = Path(output_json) if output_json else DEFAULT_OUTPUT_DIR / f"shadow_review_{_utc_stamp()}.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    report["output_path"] = str(output_path)
    _print_human(report)
    return report


def _print_human(report: dict[str, Any]) -> None:
    safe = report["safe_summary"]
    turbo = report["turbo_summary"]
    diag = report["combined_diagnosis"]
    safe_reasons = Counter(safe.get("reasons_distribution", {}))
    turbo_reasons = Counter(turbo.get("reasons_distribution", {}))
    safe_near = report.get("safe_near_entries", [{}])[0] if report.get("safe_near_entries") else {}
    turbo_near = report.get("turbo_near_entries", [{}])[0] if report.get("turbo_near_entries") else {}
    quality = report["signal_quality"]["combined"]
    print(f"Aegis Shadow Review - last {report['hours']:g}h")
    print("Safe:")
    print(f"- evaluations: {safe['total_evaluations']}")
    print(f"- LONG shadow: {safe['action_counts'].get('LONG', 0)}")
    print(f"- HOLD: {safe['action_counts'].get('HOLD', 0)}")
    print(f"- would_execute: {safe['would_execute_true']}")
    print(f"- main reason: {safe_reasons.most_common(1)[0][0] if safe_reasons else 'none'}")
    if safe_near:
        print(
            "- best near-entry: "
            f"edge_score_h12 {safe_near.get('edge_score_h12')}, "
            f"tail {safe_near.get('tail_risk_score')}, "
            f"regime {safe_near.get('regime')}, reason {safe_near.get('reason')}"
        )
    print("Turbo:")
    print(f"- evaluations: {turbo['total_evaluations']}")
    print(f"- raw agreement long>=2: {turbo['votes']['count_long_votes_gte_2']}")
    print(f"- raw agreement short>=2: {turbo['votes']['count_short_votes_gte_2']}")
    print(f"- raw LONG shadow: {turbo['raw_vs_gated']['raw_long_count']}")
    print(f"- raw would_execute: {turbo['raw_vs_gated']['raw_would_execute_count']}")
    print(f"- actual LONG shadow: {turbo['action_counts'].get('LONG', 0)}")
    print(f"- actual SHORT shadow: {turbo['action_counts'].get('SHORT', 0)}")
    print(f"- HOLD: {turbo['action_counts'].get('HOLD', 0)}")
    print(f"- would_execute: {turbo['would_execute_true']}")
    print(f"- raw->gated conversion: {turbo['raw_vs_gated']['raw_to_gated_conversion_rate']:.3f}")
    print(f"- main block: {turbo_reasons.most_common(1)[0][0] if turbo_reasons else 'none'}")
    if turbo_near:
        print(
            "- best near-entry: "
            f"turbo_score {turbo_near.get('turbo_score')}, "
            f"votes {turbo_near.get('votes')}, "
            f"reason {turbo_near.get('reason')}"
        )
    print("Signal quality:")
    print(f"- GOOD: {quality['GOOD_CANDIDATE']}")
    print(f"- MIXED: {quality['MIXED_CANDIDATE']}")
    print(f"- BAD: {quality['BAD_CANDIDATE']}")
    print(f"- PENDING: {quality['PENDING']}")
    print("Diagnosis:")
    for item in diag.get("recommendations", []):
        print(f"- {item}")
    print(f"Report JSON: {report['output_path']}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hours", type=float, default=6.0)
    parser.add_argument("--symbol", default="ETHUSDT")
    parser.add_argument("--include-pending", default="true")
    parser.add_argument("--output-json", default=None)
    args = parser.parse_args()
    review_shadow_last_hours(
        hours=args.hours,
        symbol=args.symbol,
        include_pending=_parse_bool(args.include_pending),
        output_json=args.output_json,
    )


if __name__ == "__main__":
    main()
