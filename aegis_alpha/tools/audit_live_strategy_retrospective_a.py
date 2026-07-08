#!/usr/bin/env python3
"""Read-only retrospective audit for the currently live trading strategy."""
from __future__ import annotations

import argparse
import csv
import json
import math
import random
import subprocess
import sys
from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from statistics import mean, median
from typing import Any

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from aegis_alpha.tools.audit_phase_o_short_live_entries import (  # noqa: E402
    ALL_SYMBOLS,
    REPO,
    events_by_trade,
    json_safe,
    parse_dt,
    read_log_rows,
    to_float,
    utc_stamp,
    write_csv,
)
from aegis_alpha.tools.audit_phase_o_short_live_quality import (  # noqa: E402
    aggregate_bucket,
    aggregate_symbols,
    build_quality_audit,
    candles_between,
    compute_short_mae_mfe,
    hit_before_stop_short,
    load_candles,
    pearson,
    safe_div,
)
from aegis_alpha.tools.forensic_phase_o_short_model_a import (  # noqa: E402
    find_candle_index,
    percentile,
    random_baseline as forensic_random_baseline,
)
from aegis_alpha.turbo.short_quality_v4_labels import (  # noqa: E402
    ShortV4Config,
    build_operable_short_quality_v4_labels,
    compute_short_path_metrics_v4,
)


DEFAULT_FROM = datetime(2026, 6, 1, tzinfo=timezone.utc)
REPORT_PREFIX = "aegis_live_strategy"


def now_utc() -> datetime:
    return datetime.now(timezone.utc)


def now_iso() -> str:
    return now_utc().isoformat()


def assert_read_only_output_path(path: Path) -> None:
    text = str(path.resolve())
    forbidden = ["/active/", "active_manifest.json", "phase_o_short_manifest.json"]
    if any(token in text for token in forbidden):
        raise ValueError(f"refusing live/active output path: {path}")


def safe_mean(values: list[float | None]) -> float | None:
    vals = [v for v in values if v is not None and not math.isnan(v)]
    return mean(vals) if vals else None


def safe_median(values: list[float | None]) -> float | None:
    vals = [v for v in values if v is not None and not math.isnan(v)]
    return median(vals) if vals else None


def profit_factor(pnls: list[float]) -> float | None:
    wins = sum(v for v in pnls if v > 0)
    losses = abs(sum(v for v in pnls if v < 0))
    if losses == 0:
        return math.inf if wins > 0 else None
    return wins / losses


def max_drawdown(pnls: list[float]) -> float:
    equity = 0.0
    peak = 0.0
    dd = 0.0
    for pnl in pnls:
        equity += pnl
        peak = max(peak, equity)
        dd = min(dd, equity - peak)
    return dd


def max_streak(rows: list[dict[str, Any]], want_win: bool) -> int:
    best = cur = 0
    for row in sorted(rows, key=lambda r: r.get("closed_at") or r.get("opened_at") or ""):
        win = row.get("winner") is True
        loss = row.get("winner") is False
        if (want_win and win) or ((not want_win) and loss):
            cur += 1
            best = max(best, cur)
        elif win or loss:
            cur = 0
    return best


def run_git_log() -> list[dict[str, str]]:
    try:
        proc = subprocess.run(
            ["git", "log", "--oneline", "--decorate", "--date=iso", "--pretty=format:%h %ad %s", "-30"],
            cwd=REPO,
            text=True,
            capture_output=True,
            check=False,
            timeout=8,
        )
    except Exception:
        return []
    rows = []
    for line in proc.stdout.splitlines():
        parts = line.split(" ", 3)
        if len(parts) == 4:
            rows.append({"sha": parts[0], "date": f"{parts[1]} {parts[2]}", "message": parts[3]})
    return rows


def detect_strategy_window(
    *,
    from_arg: str,
    to_arg: str,
    symbols: list[str],
    max_log_bytes_per_file: int = 15_000_000,
) -> dict[str, Any]:
    end = parse_dt(to_arg) if to_arg != "now" else now_utc()
    end = end or now_utc()
    if from_arg != "auto":
        start = parse_dt(from_arg) or DEFAULT_FROM
        return {
            "strategy_name": "Phase O SHORT live",
            "strategy_version": "auto_from_user_window",
            "from": start.isoformat(),
            "to": end.isoformat(),
            "classification": "STRATEGY_WINDOW_CONFIRMED",
            "start_reason": "explicit --from",
            "evidence": [],
        }

    broad_start = DEFAULT_FROM
    evidence: list[dict[str, Any]] = []
    for kind in ("signals", "trades", "events"):
        try:
            rows = read_log_rows(kind, broad_start, end, max_log_bytes_per_file)
        except Exception:
            rows = []
        for row in rows:
            if str(row.get("symbol") or "").upper() not in set(symbols):
                continue
            text = json.dumps(row, default=str).lower()
            if "phase_o" in text or "experimental_short_only" in text or "aegis_turbo" in text:
                ts = parse_dt(row.get("timestamp") or row.get("opened_at"))
                if ts:
                    evidence.append({"source": kind, "timestamp": ts.isoformat(), "symbol": row.get("symbol"), "event": row.get("event") or row.get("status")})
    if evidence:
        first = min(parse_dt(e["timestamp"]) for e in evidence if parse_dt(e["timestamp"]))
        return {
            "strategy_name": "Phase O SHORT live",
            "strategy_version": "phase_o_short_current",
            "from": first.isoformat(),
            "to": end.isoformat(),
            "classification": "STRATEGY_WINDOW_INFERRED",
            "start_reason": "first detected Phase O/AEGIS_TURBO signal, trade, or event",
            "evidence": sorted(evidence, key=lambda e: e["timestamp"])[:10],
        }

    commits = run_git_log()
    return {
        "strategy_name": "Phase O SHORT live",
        "strategy_version": "phase_o_short_current",
        "from": broad_start.isoformat(),
        "to": end.isoformat(),
        "classification": "STRATEGY_WINDOW_AMBIGUOUS",
        "start_reason": "fallback default; no Phase O event detected in readable logs",
        "evidence": commits[:5],
    }


def classify_signal(row: dict[str, Any]) -> str:
    meta = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
    text = " ".join(
        str(v)
        for v in (
            row.get("event"),
            row.get("status"),
            row.get("action"),
            row.get("decision"),
            row.get("reason"),
            row.get("finalReason"),
            row.get("guard_reason"),
            row.get("symbol"),
            meta.get("reason"),
            meta.get("finalReason"),
            meta.get("guard_reason"),
            meta.get("phase"),
            meta.get("source"),
        )
        if v is not None
    ).lower()
    event = str(row.get("event") or row.get("status") or row.get("action") or "").upper()
    reason = str(row.get("reason") or row.get("finalReason") or row.get("guard_reason") or "").lower()
    if "link" in str(row.get("symbol") or "").lower() and ("avoid" in text or "no_entry" in text):
        return "SIGNAL_LINK_AVOID_ONLY"
    if "error" in text or "timeout" in text or "unavailable" in text:
        return "SIGNAL_ERROR"
    if "max_consecutive_losses" in text:
        return "SIGNAL_BLOCKED_BY_MAX_LOSSES"
    if "max_turbo_trades_per_day" in text or "max_phase_o_trades_per_day" in text or "daily" in reason:
        return "SIGNAL_BLOCKED_BY_DAILY_LIMIT"
    if "cooldown" in text:
        return "SIGNAL_BLOCKED_BY_COOLDOWN"
    if "position already open" in text or "existing_position" in text:
        return "SIGNAL_BLOCKED_BY_EXISTING_POSITION"
    if "blocked" in text or "denied" in text or "wouldblock" in text:
        return "SIGNAL_BLOCKED_BY_RISK"
    if event in {"ORDER_SUBMITTED", "POSITION_CONFIRMED", "TRADE_OPENED"} or "order_submitted" in text:
        return "SIGNAL_EXECUTED_TRADE"
    if "hold" in text or event == "HOLD":
        return "SIGNAL_HOLD"
    return "SIGNAL_UNKNOWN"


def normalize_signal(row: dict[str, Any]) -> dict[str, Any]:
    meta = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
    score = row.get("turbo_score") or row.get("score") or meta.get("score")
    bucket = row.get("bucket") or meta.get("bucket")
    signal = {
        "signal_id": row.get("signal_id") or row.get("trade_id") or row.get("client_order_id") or "",
        "timestamp": row.get("timestamp") or row.get("opened_at"),
        "symbol": str(row.get("symbol") or "").upper(),
        "side": str(row.get("side") or meta.get("side") or "").upper(),
        "strategy_name": row.get("strategy") or "AEGIS_TURBO",
        "strategy_version": meta.get("strategyVersion") or meta.get("phase") or "",
        "phase": "phase_o" if any("phase_o" in str(v).lower() for v in (row.get("phase"), meta.get("phase"), meta.get("source"), row.get("strategy"))) else "",
        "model_family": meta.get("modelFamily") or meta.get("model_family") or "",
        "model_path": meta.get("modelPath") or meta.get("model_path") or "",
        "manifest_stamp": meta.get("manifestStamp") or meta.get("manifest_stamp") or "",
        "score": to_float(score),
        "bucket": bucket,
        "confidence": to_float(row.get("confidence") or meta.get("confidence")),
        "predicted_hit_probability": to_float(meta.get("predicted_hit_probability")),
        "predicted_quality": to_float(meta.get("predicted_quality")),
        "predicted_mae_danger": to_float(meta.get("predicted_mae_danger")),
        "predicted_danger": to_float(meta.get("predicted_danger")),
        "finalStrategy": row.get("finalStrategy") or meta.get("finalStrategy"),
        "finalReason": row.get("finalReason") or row.get("reason") or meta.get("finalReason"),
        "action_proposed": row.get("action") or row.get("decision") or "",
        "action_final": row.get("event") or row.get("status") or row.get("action") or "",
        "guard_result": row.get("guard_result") or meta.get("guard_result") or "",
        "guard_reason": row.get("guard_reason") or row.get("reason") or "",
        "position_already_open": "existing_position" in str(row.get("reason") or row.get("guard_reason") or meta.get("reason") or "").lower(),
        "link_avoid_only": str(row.get("symbol") or "").upper() == "LINKUSDT" and "avoid" in str(row.get("reason") or row.get("guard_reason") or meta.get("reason") or "").lower(),
        "snapshot_age": to_float(row.get("snapshot_age") or meta.get("snapshot_age")),
        "api_latency_ms": to_float(row.get("latency_ms") or meta.get("latency_ms")),
        "source_log": row.get("_log_path"),
    }
    signal["classification"] = classify_signal(row)
    signal["was_executed"] = signal["classification"] == "SIGNAL_EXECUTED_TRADE"
    signal["was_blocked"] = signal["classification"].startswith("SIGNAL_BLOCKED")
    signal["was_allowed"] = signal["was_executed"] or not signal["was_blocked"]
    return signal


def collect_signals(start: datetime, end: datetime, symbols: list[str], max_log_bytes_per_file: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for kind in ("signals", "events"):
        try:
            rows.extend(read_log_rows(kind, start, end, max_log_bytes_per_file))
        except Exception:
            continue
    symbol_set = set(symbols)
    signals = [normalize_signal(r) for r in rows if str(r.get("symbol") or "").upper() in symbol_set]
    seen: set[tuple[Any, ...]] = set()
    unique = []
    for row in sorted(signals, key=lambda r: r.get("timestamp") or ""):
        key = (row.get("timestamp"), row.get("symbol"), row.get("classification"), row.get("finalReason"), row.get("action_final"))
        if key in seen:
            continue
        seen.add(key)
        unique.append(row)
    return unique


def signal_funnel(signals: list[dict[str, Any]], trades: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    total = len(signals)
    counter = Counter(r.get("classification") for r in signals)
    rows.append({"stage": "total_signals", "count": total, "fraction": 1.0 if total else None})
    rows.append({"stage": "short_signals", "count": sum(1 for r in signals if r.get("side") == "SHORT"), "fraction": safe_div(sum(1 for r in signals if r.get("side") == "SHORT"), total)})
    rows.append({"stage": "long_signals", "count": sum(1 for r in signals if r.get("side") == "LONG"), "fraction": safe_div(sum(1 for r in signals if r.get("side") == "LONG"), total)})
    rows.append({"stage": "hold_signals", "count": counter.get("SIGNAL_HOLD", 0), "fraction": safe_div(counter.get("SIGNAL_HOLD", 0), total)})
    for cls, count in sorted(counter.items()):
        rows.append({"stage": cls, "count": count, "fraction": safe_div(count, total)})
    rows.append({"stage": "executed_trades", "count": len(trades), "fraction": safe_div(len(trades), total)})
    rows.append({"stage": "closed_trades", "count": sum(1 for r in trades if r.get("status") == "CLOSED"), "fraction": safe_div(sum(1 for r in trades if r.get("status") == "CLOSED"), total)})
    rows.append({"stage": "open_trades", "count": sum(1 for r in trades if r.get("status") == "OPEN"), "fraction": safe_div(sum(1 for r in trades if r.get("status") == "OPEN"), total)})
    return rows


def classify_trade_retrospective(row: dict[str, Any]) -> str:
    if row.get("brackets_confirmed") is not True:
        return "TRADE_BRACKET_WARNING"
    if row.get("status") == "OPEN" and (to_float(row.get("mae_roe")) or 0) > 0.25:
        return "TRADE_OPEN_RISK"
    pnl = to_float(row.get("net_pnl_estimated"))
    mfe = to_float(row.get("mfe_roe")) or 0.0
    mae = to_float(row.get("mae_roe")) or 0.0
    ratio = to_float(row.get("mfe_mae_ratio")) or 0.0
    if pnl is not None and pnl < -1.0 and mae > 0.20:
        return "TRADE_BIG_LOSS"
    if pnl is not None and pnl < 0:
        return "TRADE_NORMAL_LOSS"
    if pnl is not None and pnl >= 0 and mae >= mfe:
        return "TRADE_BAD_ENTRY_SAVED"
    if pnl is not None and pnl > 0 and ratio >= 1.5:
        return "TRADE_CLEAN_WIN"
    if pnl is not None and pnl > 0:
        return "TRADE_SMALL_WIN_MANAGED"
    return "TRADE_UNKNOWN"


def enrich_trade_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for row in rows:
        r = dict(row)
        r["trade_classification"] = classify_trade_retrospective(r)
        r["big_loss"] = r["trade_classification"] == "TRADE_BIG_LOSS"
        r["big_win"] = (to_float(r.get("net_pnl_estimated")) or 0) > 1.0 and (to_float(r.get("mfe_roe")) or 0) > 0.20
        out.append(r)
    return out


def profitability_summary(trades: list[dict[str, Any]], signals: list[dict[str, Any]], start: datetime, end: datetime) -> dict[str, Any]:
    closed = [r for r in trades if r.get("status") == "CLOSED"]
    pnls = [to_float(r.get("net_pnl_estimated")) for r in closed if to_float(r.get("net_pnl_estimated")) is not None]
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p < 0]
    days = max((end - start).total_seconds() / 86400.0, 1e-9)
    pf = profit_factor(pnls)
    expectancy = mean(pnls) if pnls else None
    pnl_total = sum(pnls) if pnls else 0.0
    if len(closed) < 5:
        cls = "LIVE_STRATEGY_TOO_EARLY"
    elif pnl_total > 0 and pf and pf > 1.0 and expectancy and expectancy > 0:
        cls = "LIVE_STRATEGY_PROFITABLE"
    elif pnl_total > 0:
        cls = "LIVE_STRATEGY_PROMISING_BUT_NOT_PROFITABLE"
    elif abs(pnl_total) < 1.0:
        cls = "LIVE_STRATEGY_FLAT"
    elif any((to_float(r.get("mae_roe")) or 0) > 0.50 for r in trades):
        cls = "LIVE_STRATEGY_DANGEROUS"
    else:
        cls = "LIVE_STRATEGY_NEGATIVE"
    return {
        "classification": cls,
        "signals_count": len(signals),
        "trades_count": len(trades),
        "closed_count": len(closed),
        "open_count": sum(1 for r in trades if r.get("status") == "OPEN"),
        "total_net_pnl": pnl_total,
        "gross_pnl": sum(to_float(r.get("gross_pnl")) or 0 for r in closed),
        "fees_estimated": sum(to_float(r.get("fee_estimate")) or 0 for r in closed),
        "win_rate": safe_div(len(wins), len(closed)),
        "profit_factor": pf,
        "expectancy_per_trade": expectancy,
        "expectancy_per_signal": safe_div(pnl_total, len(signals)),
        "avg_win": mean(wins) if wins else None,
        "avg_loss": mean(losses) if losses else None,
        "median_win": median(wins) if wins else None,
        "median_loss": median(losses) if losses else None,
        "p90_loss": sorted(losses)[int(0.1 * (len(losses) - 1))] if losses else None,
        "max_drawdown": max_drawdown(pnls),
        "max_consecutive_losses": max_streak(closed, want_win=False),
        "max_consecutive_wins": max_streak(closed, want_win=True),
        "trades_per_day": len(trades) / days,
        "signals_per_day": len(signals) / days,
        "execution_rate": safe_div(len(trades), len(signals)),
    }


def management_attribution(trades: list[dict[str, Any]], candles_by_symbol: dict[str, list[dict[str, Any]]]) -> list[dict[str, Any]]:
    rows = []
    for trade in trades:
        symbol = str(trade.get("symbol"))
        candles = candles_by_symbol.get(symbol, [])
        opened = parse_dt(trade.get("opened_at"))
        idx = find_candle_index(candles, opened) if opened else None
        entry = to_float(trade.get("entry_price"))
        leverage = to_float(trade.get("leverage")) or 20.0
        real_roe = to_float(trade.get("roe"))
        row = {"trade_id": trade.get("trade_id"), "symbol": symbol, "real_roe": real_roe}
        if idx is None or entry is None:
            row["classification"] = "UNKNOWN"
            rows.append(row)
            continue
        for bars in (3, 6, 12, 24):
            if idx + bars < len(candles):
                close = float(candles[idx + bars]["close"])
                row[f"simple_exit_roe_{bars}"] = (entry - close) / entry * leverage
        hit_rows = {
            "hit5_before_minus3": hit_before_stop_short(entry, leverage, candles[idx:idx + 24], 0.05, 0.03),
            "hit8_before_minus5": hit_before_stop_short(entry, leverage, candles[idx:idx + 24], 0.08, 0.05),
            "hit10_before_minus8": hit_before_stop_short(entry, leverage, candles[idx:idx + 24], 0.10, 0.08),
            "hit12_before_minus8": hit_before_stop_short(entry, leverage, candles[idx:idx + 24], 0.12, 0.08),
        }
        for key, value in hit_rows.items():
            row[key] = value.get("hit")
        simple = [v for k, v in row.items() if k.startswith("simple_exit_roe_") and v is not None]
        best_simple = max(simple) if simple else None
        row["best_simple_roe"] = best_simple
        row["management_value_added"] = real_roe - best_simple if real_roe is not None and best_simple is not None else None
        mfe = to_float(trade.get("mfe_roe")) or 0.0
        mae = to_float(trade.get("mae_roe")) or 0.0
        pnl = to_float(trade.get("net_pnl_estimated"))
        if pnl is not None and pnl > 0 and mfe > mae and (best_simple or 0) > 0:
            cls = "ENTRY_EDGE_CONFIRMED"
        elif pnl is not None and pnl >= 0 and mae >= mfe:
            cls = "MANAGEMENT_SAVED"
        elif row.get("management_value_added") is not None and row["management_value_added"] < -0.05:
            cls = "MANAGEMENT_HURT"
        elif pnl is not None and pnl < 0 and mfe <= mae:
            cls = "ENTRY_EDGE_WEAK"
        else:
            cls = "UNKNOWN"
        row["classification"] = cls
        rows.append(row)
    return rows


def score_calibration(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = [r for r in rows if r.get("status") == "CLOSED"]
    metrics = [
        ("score_vs_pnl", "net_pnl_estimated"),
        ("score_vs_mfe", "mfe_roe"),
        ("score_vs_mae", "mae_roe"),
        ("score_vs_mfe_mae", "mfe_mae_ratio"),
        ("score_vs_win", "winner"),
        ("score_vs_big_loss", "big_loss"),
    ]
    out = []
    scores = [to_float(r.get("model_score")) for r in rows]
    valid_scores = [s for s in scores if s is not None]
    for name, field in metrics:
        pairs = []
        for row in rows:
            s = to_float(row.get("model_score"))
            if s is None:
                continue
            raw = row.get(field)
            if isinstance(raw, bool):
                val = 1.0 if raw else 0.0
            else:
                val = to_float(raw)
            if val is not None:
                pairs.append((s, val))
        corr = pearson([p[0] for p in pairs], [p[1] for p in pairs]) if len(pairs) >= 3 else None
        out.append({"metric": name, "sample_count": len(pairs), "correlation": corr, "status": "OK" if corr is not None else "INSUFFICIENT_DATA"})
    pnl_corr = next((r.get("correlation") for r in out if r["metric"] == "score_vs_pnl"), None)
    status = "SCORE_NOT_CALIBRATED"
    if pnl_corr is None or len(valid_scores) < 5:
        status = "INSUFFICIENT_DATA"
    elif pnl_corr > 0.25:
        status = "SCORE_CALIBRATED"
    elif pnl_corr > 0.05:
        status = "SCORE_WEAKLY_CALIBRATED"
    elif pnl_corr < -0.05:
        status = "SCORE_INVERTED"
    out.insert(0, {"metric": "global_score_calibration", "sample_count": len(valid_scores), "status": status, "score_vs_pnl": pnl_corr})
    return out


def bucket_calibration(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = [r for r in rows if r.get("status") == "CLOSED"]
    out = aggregate_bucket(rows)
    for row in out:
        if row.get("bucket") == "premium":
            row["classification"] = "PREMIUM_JUSTIFIED" if (row.get("pnl_total") or 0) > 0 and (row.get("win_rate") or 0) > 0.5 else "PREMIUM_NOT_JUSTIFIED"
        elif row.get("bucket") == "normal":
            row["classification"] = "NORMAL_OK" if (row.get("pnl_total") or 0) >= 0 else "NORMAL_WEAK"
        else:
            row["classification"] = "BUCKETS_INSUFFICIENT_DATA" if row.get("trades_count", 0) < 3 else "CONSERVATIVE_ONLY"
    return out


def build_candles_by_symbol(trades: list[dict[str, Any]], start: datetime, end: datetime) -> dict[str, list[dict[str, Any]]]:
    out = {}
    for symbol in sorted({str(r.get("symbol") or "").upper() for r in trades if r.get("symbol")}):
        out[symbol] = load_candles(symbol, start - timedelta(hours=1), end + timedelta(hours=1))
    return out


def v4_overlap(trades: list[dict[str, Any]], candles_by_symbol: dict[str, list[dict[str, Any]]], horizon: int = 12) -> list[dict[str, Any]]:
    rows = []
    for trade in trades:
        symbol = str(trade.get("symbol") or "").upper()
        candles = candles_by_symbol.get(symbol, [])
        opened = parse_dt(trade.get("opened_at"))
        idx = find_candle_index(candles, opened) if opened else None
        if idx is None or idx >= len(candles) - 2:
            rows.append({"trade_id": trade.get("trade_id"), "symbol": symbol, "status": "INSUFFICIENT_DATA"})
            continue
        high = [float(c["high"]) for c in candles]
        low = [float(c["low"]) for c in candles]
        close = [float(c["close"]) for c in candles]
        cfg = ShortV4Config(horizon=horizon)
        metrics = compute_short_path_metrics_v4(high=high, low=low, close=close, entry_index=idx, horizon=horizon, config=cfg)
        label_rows = build_operable_short_quality_v4_labels(symbol=symbol, high=high, low=low, close=close, steps=[idx], horizon=horizon, config=cfg)
        labels = label_rows[0] if label_rows else {}
        rows.append({
            "trade_id": trade.get("trade_id"),
            "symbol": symbol,
            "horizon": horizon,
            "short_clean_entry_v4": labels["short_clean_entry_v4"],
            "short_bad_entry_v4": labels["short_bad_entry_v4"],
            "short_premium_allowed_v4": labels["short_premium_allowed_v4"],
            "short_management_dependent_v4": labels["short_management_dependent_v4"],
            "short_no_trade_v4": labels["short_no_trade_v4"],
            "v4_net_quality": metrics.get("net_quality_after_costs"),
            "v4_mfe_mae_ratio": metrics.get("mfe_mae_ratio"),
            "v4_mae_roe": metrics.get("mae_roe_proxy"),
            "status": "OK",
        })
    return rows


def classify_random_strategy(rows: list[dict[str, Any]]) -> str:
    ok = [r for r in rows if r.get("status") == "OK"]
    if len(ok) < 5:
        return "INSUFFICIENT_RANDOM_BASELINE"
    avg = mean([to_float(r.get("live_quality_percentile")) or 0 for r in ok])
    above = sum(1 for r in ok if r.get("live_better_than_random_median"))
    if avg >= 0.60 and above / len(ok) >= 0.60:
        return "STRATEGY_BEATS_RANDOM"
    if avg < 0.45 and above / len(ok) < 0.50:
        return "STRATEGY_NOT_BETTER_THAN_RANDOM"
    return "STRATEGY_MIXED_VS_RANDOM"


def symbol_diagnostics(
    trades: list[dict[str, Any]],
    signals: list[dict[str, Any]],
    random_rows: list[dict[str, Any]],
    v4_rows: list[dict[str, Any]],
    management_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    by_random = defaultdict(list)
    for row in random_rows:
        by_random[row.get("symbol")].append(row)
    by_v4 = {row.get("trade_id"): row for row in v4_rows}
    by_mgmt = {row.get("trade_id"): row for row in management_rows}
    out = []
    for symbol in sorted(set([r.get("symbol") for r in trades] + [r.get("symbol") for r in signals if r.get("symbol")])):
        group = [r for r in trades if r.get("symbol") == symbol]
        sigs = [r for r in signals if r.get("symbol") == symbol]
        closed = [r for r in group if r.get("status") == "CLOSED"]
        pnls = [to_float(r.get("net_pnl_estimated")) for r in closed if to_float(r.get("net_pnl_estimated")) is not None]
        ptotal = sum(pnls) if pnls else 0.0
        pf = profit_factor(pnls)
        v4_bad = sum(1 for r in group if (by_v4.get(r.get("trade_id")) or {}).get("short_bad_entry_v4") == 1)
        mgmt_dep = sum(1 for r in group if (by_mgmt.get(r.get("trade_id")) or {}).get("classification") in {"MANAGEMENT_SAVED", "MANAGEMENT_MASKS_BAD_ENTRY"})
        avg_rand = safe_mean([to_float(r.get("live_quality_percentile")) for r in by_random[symbol] if r.get("status") == "OK"])
        if len(group) < 2:
            rec = "INSUFFICIENT_DATA"
        elif ptotal > 0 and (pf or 0) >= 1.0 and v4_bad == 0:
            rec = "KEEP_LIVE_CANDIDATE"
        elif ptotal < 0 and v4_bad >= max(1, len(group) // 2):
            rec = "PAUSE_LIVE"
        elif ptotal < 0:
            rec = "REDUCE_LIVE"
        else:
            rec = "RESEARCH_ONLY"
        out.append({
            "symbol": symbol,
            "signals": len(sigs),
            "trades": len(group),
            "execution_rate": safe_div(len(group), len(sigs)),
            "closed": len(closed),
            "pnl_total": ptotal,
            "win_rate": safe_div(sum(1 for r in closed if r.get("winner") is True), len(closed)),
            "profit_factor": pf,
            "expectancy": mean(pnls) if pnls else None,
            "avg_mfe_roe": safe_mean([to_float(r.get("mfe_roe")) for r in group]),
            "avg_mae_roe": safe_mean([to_float(r.get("mae_roe")) for r in group]),
            "p90_mae_roe": sorted([to_float(r.get("mae_roe")) for r in group if to_float(r.get("mae_roe")) is not None])[int(0.9 * (len(group) - 1))] if group else None,
            "big_losses": sum(1 for r in group if r.get("big_loss")),
            "v4_bad_count": v4_bad,
            "management_dependent_count": mgmt_dep,
            "avg_random_quality_percentile": avg_rand,
            "recommendation": rec,
        })
    return out


def safety_audit(signals: list[dict[str, Any]], trades: list[dict[str, Any]]) -> dict[str, Any]:
    def row_text(row: dict[str, Any]) -> str:
        fields = (
            row.get("classification"),
            row.get("finalReason"),
            row.get("guard_reason"),
            row.get("reason"),
            row.get("action_final"),
            row.get("trade_classification"),
            row.get("source_log"),
        )
        return " ".join(str(v) for v in fields if v is not None).lower()

    texts = [row_text(r) for r in signals] + [row_text(r) for r in trades]
    joined = "\n".join(texts)
    issues = {
        "max_consecutive_losses_hits": joined.count("max_consecutive_losses"),
        "daily_limit_hits": joined.count("max_turbo_trades_per_day") + joined.count("max_phase_o_trades_per_day"),
        "cooldown_hits": joined.count("cooldown"),
        "link_avoid_only_hits": sum(1 for t in texts if "link" in t and "avoid" in t),
        "immediate_trigger_risk_count": joined.count("immediate_trigger_risk"),
        "order_endpoint_fallback_warnings": joined.count("order type not supported"),
        "stale_snapshot_mentions": joined.count("stale") + joined.count("snapshot_stale"),
        "bracket_missing_count": sum(1 for r in trades if r.get("brackets_confirmed") is not True),
    }
    if issues["bracket_missing_count"] or issues["immediate_trigger_risk_count"]:
        status = "SAFETY_RISK"
    elif any(v for k, v in issues.items() if k not in {"link_avoid_only_hits"}):
        status = "SAFETY_WARNINGS"
    else:
        status = "SAFETY_OK"
    return {"classification": status, **issues}


def learning_candidates(
    trades: list[dict[str, Any]],
    random_rows: list[dict[str, Any]],
    v4_rows: list[dict[str, Any]],
    management_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    random_by_id = {r.get("trade_id"): r for r in random_rows}
    v4_by_id = {r.get("trade_id"): r for r in v4_rows}
    mgmt_by_id = {r.get("trade_id"): r for r in management_rows}
    out = []
    for trade in trades:
        tid = trade.get("trade_id")
        v4 = v4_by_id.get(tid, {})
        rnd = random_by_id.get(tid, {})
        mgmt = mgmt_by_id.get(tid, {})
        pnl = to_float(trade.get("net_pnl_estimated"))
        mfe = to_float(trade.get("mfe_roe"))
        mae = to_float(trade.get("mae_roe"))
        random_pct = to_float(rnd.get("live_quality_percentile"))
        if v4.get("short_clean_entry_v4") == 1 and pnl is not None and pnl > 0:
            label, use = "positive_clean_winner", "train_positive"
        elif trade.get("big_loss") or v4.get("short_bad_entry_v4") == 1 or (pnl is not None and pnl < 0):
            label, use = "negative_bad_or_loss", "train_negative"
        elif mgmt.get("classification") == "MANAGEMENT_SAVED":
            label, use = "ambiguous_management_saved", "manual_review"
        elif random_pct is not None and random_pct < 0.45:
            label, use = "negative_random_like", "train_negative"
        else:
            label, use = "ambiguous", "manual_review"
        out.append({
            "symbol": trade.get("symbol"),
            "timestamp": trade.get("opened_at"),
            "side": trade.get("side"),
            "label_candidate": label,
            "reason": trade.get("trade_classification"),
            "pnl": pnl,
            "mfe": mfe,
            "mae": mae,
            "mfe_mae_ratio": trade.get("mfe_mae_ratio"),
            "score": trade.get("model_score"),
            "bucket": trade.get("bucket"),
            "v4_labels": {k: v4.get(k) for k in ("short_clean_entry_v4", "short_bad_entry_v4", "short_premium_allowed_v4", "short_management_dependent_v4", "short_no_trade_v4")},
            "management_dependency": mgmt.get("classification"),
            "random_percentile": random_pct,
            "recommended_use": use,
        })
    return out


def final_decision(
    profitability: dict[str, Any],
    symbols: list[dict[str, Any]],
    score_rows: list[dict[str, Any]],
    random_status: str,
    safety: dict[str, Any],
) -> dict[str, str]:
    pause = [r["symbol"] for r in symbols if r.get("recommendation") == "PAUSE_LIVE"]
    reduce = [r["symbol"] for r in symbols if r.get("recommendation") == "REDUCE_LIVE"]
    score_status = (score_rows[0] or {}).get("status") if score_rows else "INSUFFICIENT_DATA"
    if safety.get("classification") == "SAFETY_RISK":
        live = "LIVE_STRATEGY_PAUSE"
    elif pause:
        live = "LIVE_STRATEGY_DISABLE_SYMBOLS"
    elif profitability.get("classification") in {"LIVE_STRATEGY_NEGATIVE", "LIVE_STRATEGY_DANGEROUS"}:
        live = "LIVE_STRATEGY_KEEP_REDUCED"
    elif profitability.get("classification") == "LIVE_STRATEGY_TOO_EARLY":
        live = "LIVE_STRATEGY_TOO_EARLY"
    else:
        live = "LIVE_STRATEGY_KEEP"
    if score_status in {"SCORE_NOT_CALIBRATED", "SCORE_INVERTED"}:
        ml = "PROCEED_TO_SHORT_V4_B"
    elif random_status == "STRATEGY_NOT_BETTER_THAN_RANDOM":
        ml = "BUILD_RISK_MODEL_FIRST"
    elif profitability.get("closed_count", 0) < 10:
        ml = "COLLECT_MORE_LIVE_DATA"
    else:
        ml = "PROCEED_TO_SHORT_V4_B"
    return {
        "live_strategy": live,
        "ml_next": ml,
        "pause_symbols": ",".join(pause),
        "reduce_symbols": ",".join(reduce),
    }


def build_retrospective(args: argparse.Namespace) -> dict[str, Any]:
    symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()] if args.symbols else ALL_SYMBOLS
    window = detect_strategy_window(from_arg=args.from_ts, to_arg=args.to, symbols=symbols, max_log_bytes_per_file=args.max_log_bytes_per_file)
    start = parse_dt(window["from"]) or DEFAULT_FROM
    end = parse_dt(window["to"]) or now_utc()
    quality_args = argparse.Namespace(
        from_ts=start.isoformat(),
        to=end.isoformat(),
        symbols=",".join(symbols),
        include_open=True,
        include_closed=True,
        machine_gun_window_seconds=args.machine_gun_window_seconds,
        max_log_bytes_per_file=args.max_log_bytes_per_file,
    )
    quality = build_quality_audit(quality_args)
    trades = enrich_trade_rows(quality.get("trades", []))
    signals = collect_signals(start, end, symbols, args.max_log_bytes_per_file) if args.include_signals else []
    funnel = signal_funnel(signals, trades)
    profitability = profitability_summary(trades, signals, start, end)
    candles_by_symbol = build_candles_by_symbol(trades, start, end) if (args.include_mfe_mae or args.include_management_attribution or args.include_random_baseline or args.include_v4_label_overlap) else {}
    management = management_attribution(trades, candles_by_symbol) if args.include_management_attribution else []
    random_rows = forensic_random_baseline(trades, candles_by_symbol, n=30 if args.fast else 100) if args.include_random_baseline else []
    random_status = classify_random_strategy(random_rows) if random_rows else "INSUFFICIENT_RANDOM_BASELINE"
    v4_rows = v4_overlap(trades, candles_by_symbol) if args.include_v4_label_overlap else []
    score_rows = score_calibration(trades) if args.include_model_score_calibration else []
    bucket_rows = bucket_calibration(trades)
    symbol_rows = symbol_diagnostics(trades, signals, random_rows, v4_rows, management) if args.include_symbol_diagnostics else aggregate_symbols(trades)
    safety = safety_audit(signals, trades) if args.include_bracket_safety or args.include_guard_funnel else {"classification": "SAFETY_UNKNOWN"}
    learning = learning_candidates(trades, random_rows, v4_rows, management) if args.include_learning_dataset or args.export_learning_candidates else []
    decision = final_decision(profitability, symbol_rows, score_rows, random_status, safety)
    return {
        "schema_version": "live_strategy_retrospective_a_v1",
        "created_at": now_iso(),
        "mode": "READ_ONLY",
        "strategy_window": window,
        "summary": {
            **profitability,
            "live_decision": decision["live_strategy"],
            "ml_next": decision["ml_next"],
            "random_baseline": random_status,
            "score_calibration": (score_rows[0].get("status") if score_rows else "INSUFFICIENT_DATA"),
            "safety": safety.get("classification"),
        },
        "signals": signals,
        "trades": trades,
        "signal_funnel": funnel,
        "symbols": symbol_rows,
        "buckets": bucket_rows,
        "score_calibration": score_rows,
        "mfe_mae": [{"trade_id": r.get("trade_id"), "symbol": r.get("symbol"), "mfe_roe": r.get("mfe_roe"), "mae_roe": r.get("mae_roe"), "mfe_mae_ratio": r.get("mfe_mae_ratio")} for r in trades],
        "management_attribution": management,
        "random_baseline": random_rows,
        "random_baseline_status": random_status,
        "v4_overlap": v4_rows,
        "bracket_safety": [safety],
        "learning_candidates": learning,
        "recommendations": build_recommendations(decision, profitability, symbol_rows, score_rows, random_status, safety),
        "confirmations": {
            "no_live_changes": True,
            "no_active_manifest": True,
            "no_yaml": True,
            "no_pm2_restart": True,
            "no_orders": True,
            "no_env": True,
            "no_push": True,
            "no_commit": True,
        },
    }


def build_recommendations(
    decision: dict[str, str],
    profitability: dict[str, Any],
    symbols: list[dict[str, Any]],
    score_rows: list[dict[str, Any]],
    random_status: str,
    safety: dict[str, Any],
) -> list[dict[str, Any]]:
    recs = [
        {"area": "live_strategy", "recommendation": decision["live_strategy"], "reason": profitability.get("classification")},
        {"area": "ml", "recommendation": decision["ml_next"], "reason": f"score={score_rows[0].get('status') if score_rows else 'INSUFFICIENT_DATA'} random={random_status}"},
    ]
    for row in symbols:
        recs.append({"area": "symbol", "symbol": row.get("symbol"), "recommendation": row.get("recommendation") or row.get("entry_quality_grade"), "reason": f"pnl={row.get('pnl_total')} trades={row.get('trades') or row.get('trades_count')}"})
    if safety.get("classification") != "SAFETY_OK":
        recs.append({"area": "safety", "recommendation": "REVIEW_WARNINGS_READ_ONLY", "reason": safety.get("classification")})
    return recs


def write_markdown(path: Path, payload: dict[str, Any]) -> None:
    s = payload["summary"]
    w = payload["strategy_window"]
    lines = [
        "# Live Strategy Retrospective A",
        "",
        "## Safety",
        "- read-only",
        "- no live changes",
        "- no active_manifest",
        "- no YAML",
        "- no PM2 restart",
        "- no orders",
        "",
        "## Executive Summary",
        f"- Strategy: `{w.get('strategy_name')}` `{w.get('strategy_version')}`",
        f"- Window: `{w.get('from')}` to `{w.get('to')}`",
        f"- Window classification: `{w.get('classification')}` ({w.get('start_reason')})",
        f"- Live profitability: `{s.get('classification')}`",
        f"- Live decision: `{s.get('live_decision')}`",
        f"- ML next: `{s.get('ml_next')}`",
        f"- Safety: `{s.get('safety')}`",
        f"- Random baseline: `{s.get('random_baseline')}`",
        f"- Score calibration: `{s.get('score_calibration')}`",
        "",
        "## Signal Funnel",
        "| stage | count | fraction |",
        "| --- | ---: | ---: |",
    ]
    for row in payload["signal_funnel"]:
        lines.append(f"| {row.get('stage')} | {row.get('count')} | {row.get('fraction')} |")
    lines += [
        "",
        "## Profitability",
        f"- Trades: `{s.get('trades_count')}` closed=`{s.get('closed_count')}` open=`{s.get('open_count')}`",
        f"- Net PnL: `{s.get('total_net_pnl')}`",
        f"- Profit factor: `{s.get('profit_factor')}`",
        f"- Expectancy/trade: `{s.get('expectancy_per_trade')}`",
        f"- Expectancy/signal: `{s.get('expectancy_per_signal')}`",
        f"- Win rate: `{s.get('win_rate')}` (not sufficient alone)",
        f"- Max drawdown: `{s.get('max_drawdown')}`",
        f"- Max consecutive losses: `{s.get('max_consecutive_losses')}`",
        "",
        "## Symbols",
        "| symbol | recommendation | signals | trades | pnl | win_rate | profit_factor | big_losses | v4_bad |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in payload["symbols"]:
        lines.append(f"| {row.get('symbol')} | {row.get('recommendation') or row.get('entry_quality_grade')} | {row.get('signals')} | {row.get('trades') or row.get('trades_count')} | {row.get('pnl_total')} | {row.get('win_rate')} | {row.get('profit_factor')} | {row.get('big_losses')} | {row.get('v4_bad_count')} |")
    lines += ["", "## Score And Buckets"]
    for row in payload["score_calibration"][:8]:
        lines.append(f"- {row.get('metric')}: `{row.get('status')}` corr={row.get('correlation') or row.get('score_vs_pnl')} sample={row.get('sample_count')}")
    for row in payload["buckets"]:
        lines.append(f"- bucket {row.get('bucket')}: `{row.get('classification')}` trades={row.get('trades_count')} pnl={row.get('pnl_total')} win_rate={row.get('win_rate')}")
    lines += ["", "## Entry Vs Management"]
    for row in Counter(r.get("classification") for r in payload["management_attribution"]).items():
        lines.append(f"- {row[0]}: {row[1]}")
    lines += ["", "## V4 Overlap"]
    ok_v4 = [r for r in payload["v4_overlap"] if r.get("status") == "OK"]
    lines.append(f"- rows: `{len(ok_v4)}`")
    lines.append(f"- clean: `{sum(1 for r in ok_v4 if r.get('short_clean_entry_v4') == 1)}`")
    lines.append(f"- bad: `{sum(1 for r in ok_v4 if r.get('short_bad_entry_v4') == 1)}`")
    lines.append(f"- premium_allowed: `{sum(1 for r in ok_v4 if r.get('short_premium_allowed_v4') == 1)}`")
    lines += ["", "## Random Baseline", f"- Status: `{payload.get('random_baseline_status')}`"]
    if payload["random_baseline"]:
        avg_pct = safe_mean([to_float(r.get("live_quality_percentile")) for r in payload["random_baseline"] if r.get("status") == "OK"])
        lines.append(f"- Average live quality percentile: `{avg_pct}`")
    lines += ["", "## Learning Dataset"]
    for label, count in Counter(r.get("recommended_use") for r in payload["learning_candidates"]).items():
        lines.append(f"- {label}: {count}")
    lines += ["", "## Recommendations"]
    for row in payload["recommendations"]:
        lines.append(f"- {row.get('area')} {row.get('symbol','')}: `{row.get('recommendation')}` - {row.get('reason')}")
    lines += ["", "## Confirmations"]
    for key, value in payload["confirmations"].items():
        lines.append(f"- {key}: `{value}`")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_reports(payload: dict[str, Any], out_dir: Path) -> dict[str, str]:
    assert_read_only_output_path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = utc_stamp()
    base = out_dir / f"aegis_live_strategy_retrospective_a_{ts}"
    paths = {
        "md": str(base.with_suffix(".md")),
        "json": str(base.with_suffix(".json")),
        "trades_csv": str(out_dir / f"aegis_live_strategy_trades_{ts}.csv"),
        "signals_csv": str(out_dir / f"aegis_live_strategy_signals_{ts}.csv"),
        "signal_funnel_csv": str(out_dir / f"aegis_live_strategy_signal_funnel_{ts}.csv"),
        "symbols_csv": str(out_dir / f"aegis_live_strategy_symbols_{ts}.csv"),
        "buckets_csv": str(out_dir / f"aegis_live_strategy_buckets_{ts}.csv"),
        "score_calibration_csv": str(out_dir / f"aegis_live_strategy_score_calibration_{ts}.csv"),
        "mfe_mae_csv": str(out_dir / f"aegis_live_strategy_mfe_mae_{ts}.csv"),
        "management_attribution_csv": str(out_dir / f"aegis_live_strategy_management_attribution_{ts}.csv"),
        "random_baseline_csv": str(out_dir / f"aegis_live_strategy_random_baseline_{ts}.csv"),
        "v4_overlap_csv": str(out_dir / f"aegis_live_strategy_v4_overlap_{ts}.csv"),
        "bracket_safety_csv": str(out_dir / f"aegis_live_strategy_bracket_safety_{ts}.csv"),
        "learning_candidates_csv": str(out_dir / f"aegis_live_strategy_learning_candidates_{ts}.csv"),
        "recommendations_csv": str(out_dir / f"aegis_live_strategy_recommendations_{ts}.csv"),
    }
    payload["reports"] = paths
    Path(paths["json"]).write_text(json.dumps(json_safe(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_markdown(Path(paths["md"]), payload)
    write_csv(Path(paths["trades_csv"]), payload["trades"])
    write_csv(Path(paths["signals_csv"]), payload["signals"])
    write_csv(Path(paths["signal_funnel_csv"]), payload["signal_funnel"])
    write_csv(Path(paths["symbols_csv"]), payload["symbols"])
    write_csv(Path(paths["buckets_csv"]), payload["buckets"])
    write_csv(Path(paths["score_calibration_csv"]), payload["score_calibration"], fields=["metric", "sample_count", "status", "score_vs_pnl", "correlation"])
    write_csv(Path(paths["mfe_mae_csv"]), payload["mfe_mae"])
    write_csv(Path(paths["management_attribution_csv"]), payload["management_attribution"])
    write_csv(Path(paths["random_baseline_csv"]), payload["random_baseline"])
    write_csv(Path(paths["v4_overlap_csv"]), payload["v4_overlap"])
    write_csv(Path(paths["bracket_safety_csv"]), payload["bracket_safety"])
    write_csv(Path(paths["learning_candidates_csv"]), payload["learning_candidates"])
    write_csv(Path(paths["recommendations_csv"]), payload["recommendations"])
    return paths


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--from", dest="from_ts", default="auto")
    p.add_argument("--to", default="now")
    p.add_argument("--strategy", default="auto")
    p.add_argument("--out-dir", default="/home/jasan/Develop")
    p.add_argument("--symbols", default=",".join(ALL_SYMBOLS))
    p.add_argument("--include-signals", action="store_true")
    p.add_argument("--include-trades", action="store_true")
    p.add_argument("--include-guard-funnel", action="store_true")
    p.add_argument("--include-mfe-mae", action="store_true")
    p.add_argument("--include-management-attribution", action="store_true")
    p.add_argument("--include-random-baseline", action="store_true")
    p.add_argument("--include-v4-label-overlap", action="store_true")
    p.add_argument("--include-symbol-diagnostics", action="store_true")
    p.add_argument("--include-model-score-calibration", action="store_true")
    p.add_argument("--include-bracket-safety", action="store_true")
    p.add_argument("--include-learning-dataset", action="store_true")
    p.add_argument("--export-learning-candidates", action="store_true")
    p.add_argument("--machine-gun-window-seconds", type=int, default=300)
    p.add_argument("--max-log-bytes-per-file", type=int, default=20_000_000)
    p.add_argument("--fast", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    payload = build_retrospective(args)
    paths = write_reports(payload, Path(args.out_dir))
    print(json.dumps(paths, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
