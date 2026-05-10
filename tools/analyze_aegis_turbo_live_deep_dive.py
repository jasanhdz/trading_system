#!/usr/bin/env python3
"""Deep-dive analyzer for local Aegis Turbo live logs.

This script is intentionally offline-only. It reads JSONL/log/report files from
the local repositories and writes diagnostic reports under ./reports.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import subprocess
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from statistics import median
from typing import Any, Iterable


ROOT = Path("/home/jasan/Develop/trading_system")
BOT = ROOT / "binance-futures-bot-ts"
SYMBOLS = [
    "ETHUSDT",
    "BTCUSDT",
    "SOLUSDT",
    "BNBUSDT",
    "XRPUSDT",
    "DOGEUSDT",
    "ADAUSDT",
    "AVAXUSDT",
    "LINKUSDT",
    "SUIUSDT",
    "LTCUSDT",
]

MIN_DATES = {"2026-05-08", "2026-05-09", "2026-05-10"}

SCORE_BUCKETS = [
    ("<0.60", float("-inf"), 0.60),
    ("0.60-0.65", 0.60, 0.65),
    ("0.65-0.70", 0.65, 0.70),
    ("0.70-0.80", 0.70, 0.80),
    ("0.80-0.90", 0.80, 0.90),
    (">=0.90", 0.90, float("inf")),
]


@dataclass
class JsonlRead:
    rows: list[dict[str, Any]]
    corrupt_lines: int = 0
    files_read: int = 0
    missing_files: int = 0


def parse_dt(value: Any) -> datetime | None:
    if not value:
        return None
    if isinstance(value, datetime):
        return value
    text = str(value)
    try:
        if text.endswith("Z"):
            text = text[:-1] + "+00:00"
        return datetime.fromisoformat(text)
    except ValueError:
        return None


def utc_now_tag() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def safe_float(value: Any) -> float | None:
    try:
        if value is None or value == "":
            return None
        out = float(value)
        if math.isnan(out) or math.isinf(out):
            return None
        return out
    except (TypeError, ValueError):
        return None


def pct(value: float | None) -> float | None:
    if value is None:
        return None
    return round(value * 100, 2)


def round_or_none(value: Any, digits: int = 6) -> float | None:
    num = safe_float(value)
    if num is None:
        return None
    return round(num, digits)


def mean(values: Iterable[float | None]) -> float | None:
    nums = [v for v in values if v is not None]
    if not nums:
        return None
    return sum(nums) / len(nums)


def med(values: Iterable[float | None]) -> float | None:
    nums = [v for v in values if v is not None]
    if not nums:
        return None
    return float(median(nums))


def profit_factor(gross_profit: float, gross_loss: float) -> float | None:
    loss_abs = abs(gross_loss)
    if loss_abs == 0:
        if gross_profit > 0:
            return float("inf")
        return None
    return gross_profit / loss_abs


def format_num(value: Any, digits: int = 4) -> str:
    num = safe_float(value)
    if num is None:
        return "n/a"
    if math.isinf(num):
        return "inf"
    return f"{num:.{digits}f}"


def json_safe(value: Any) -> Any:
    if isinstance(value, float):
        if math.isnan(value):
            return None
        if math.isinf(value):
            return "Infinity" if value > 0 else "-Infinity"
        return value
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [json_safe(v) for v in value]
    if isinstance(value, tuple):
        return [json_safe(v) for v in value]
    return value


def run_readonly_command(args: list[str], cwd: Path) -> dict[str, Any]:
    try:
        proc = subprocess.run(args, cwd=str(cwd), text=True, capture_output=True, timeout=10, check=False)
        return {
            "cmd": " ".join(args),
            "cwd": str(cwd),
            "returncode": proc.returncode,
            "stdout": proc.stdout.strip(),
            "stderr": proc.stderr.strip(),
        }
    except Exception as exc:
        return {"cmd": " ".join(args), "cwd": str(cwd), "error": str(exc)}


def format_pct(value: Any) -> str:
    num = safe_float(value)
    if num is None:
        return "n/a"
    return f"{num:.2f}%"


def extract_date_from_name(path: Path) -> str | None:
    match = re.search(r"(\d{4}-\d{2}-\d{2})", path.name)
    if match:
        return match.group(1)
    match = re.search(r"(20\d{6})", path.name)
    if match:
        raw = match.group(1)
        return f"{raw[:4]}-{raw[4:6]}-{raw[6:8]}"
    return None


def available_jsonl_dates() -> dict[str, list[str]]:
    roots = {
        "bot_aegis": BOT / "logs" / "aegis",
        "turbo": ROOT / "aegis_alpha" / "logs" / "turbo",
        "shadow": ROOT / "aegis_alpha" / "logs" / "shadow",
        "turbo_retrain": ROOT / "aegis_alpha" / "logs" / "turbo_retrain",
    }
    out: dict[str, list[str]] = {}
    for label, folder in roots.items():
        dates = set()
        if folder.exists():
            for path in folder.rglob("*"):
                if path.is_file():
                    date = extract_date_from_name(path)
                    if date:
                        dates.add(date)
        out[label] = sorted(dates)
    return out


def decide_dates(available: dict[str, list[str]], requested: list[str] | None) -> list[str]:
    if requested:
        return sorted(set(requested))
    all_dates = set()
    for dates in available.values():
        all_dates.update(dates)
    selected = set(MIN_DATES)
    selected.update(date for date in all_dates if date > "2026-05-10")
    return sorted(selected)


def read_jsonl_files(paths: Iterable[Path]) -> JsonlRead:
    result = JsonlRead(rows=[])
    for path in paths:
        if not path.exists():
            result.missing_files += 1
            continue
        result.files_read += 1
        with path.open("r", encoding="utf-8", errors="replace") as handle:
            for raw_line in handle:
                line = raw_line.replace("\x00", "").strip()
                if not line:
                    continue
                try:
                    parsed = json.loads(line)
                    if isinstance(parsed, dict):
                        parsed["_source_file"] = str(path)
                        result.rows.append(parsed)
                    else:
                        result.corrupt_lines += 1
                except json.JSONDecodeError:
                    result.corrupt_lines += 1
    return result


def read_json(path: Path) -> dict[str, Any] | None:
    try:
        return json.loads(path.read_text(encoding="utf-8", errors="replace"))
    except Exception:
        return None


def date_paths(prefix: str, dates: list[str]) -> list[Path]:
    return [BOT / "logs" / "aegis" / f"{prefix}_{date}.jsonl" for date in dates]


def trade_pnl(trade: dict[str, Any]) -> float | None:
    for key in ("net_pnl_usdt", "pnl_usdt", "pnl", "realized_pnl"):
        value = safe_float(trade.get(key))
        if value is not None:
            return value
    metadata = trade.get("metadata") or {}
    if isinstance(metadata, dict):
        return safe_float(metadata.get("pnl"))
    return None


def trade_roe(trade: dict[str, Any]) -> float | None:
    for key in ("roe", "roe_pct"):
        value = safe_float(trade.get(key))
        if value is not None:
            return value
    metadata = trade.get("metadata") or {}
    if isinstance(metadata, dict):
        return safe_float(metadata.get("roe"))
    return None


def enrich_trade(close: dict[str, Any], open_by_id: dict[str, dict[str, Any]]) -> dict[str, Any]:
    out = dict(open_by_id.get(str(close.get("trade_id")), {}))
    out.update(close)
    src = open_by_id.get(str(close.get("trade_id")), {})
    for key in ("turbo_score", "votes", "raw_action", "gated_action", "final_action"):
        if out.get(key) is None and src.get(key) is not None:
            out[key] = src[key]
    return out


def classify_exit_reason(value: Any) -> str:
    text = str(value or "UNKNOWN").upper()
    if "STOP_LOSS" in text or text == "SL" or "STOP LOSS" in text:
        return "STOP_LOSS"
    if "TAKE_PROFIT" in text or "TAKE PROFIT" in text or text == "TP":
        return "TAKE_PROFIT"
    if "TRAILING" in text:
        return "TRAILING_STOP"
    if "BREAK_EVEN" in text or "BREAKEVEN" in text:
        return "BREAK_EVEN"
    if "NEUTRAL_DECAY" in text:
        return "AEGIS_EXIT_EYE_NEUTRAL_DECAY"
    if "OPPOSITE_SIGNAL" in text:
        return "AEGIS_EXIT_EYE_OPPOSITE_SIGNAL"
    if "MAX_HOLD" in text or "TIME_LIMIT" in text or "TIME LIMIT" in text:
        return "MAX_HOLD"
    if "EMERGENCY" in text:
        return "EMERGENCY_CLOSE"
    if "MANUAL" in text:
        return "MANUAL"
    return "UNKNOWN"


def score_bucket(score: Any) -> str:
    num = safe_float(score)
    if num is None:
        return "UNKNOWN"
    for label, low, high in SCORE_BUCKETS:
        if low <= num < high:
            return label
    return "UNKNOWN"


def vote_pattern(votes: Any) -> str:
    if not isinstance(votes, dict):
        return "UNKNOWN"
    l = int(safe_float(votes.get("long")) or 0)
    s = int(safe_float(votes.get("short")) or 0)
    n = int(safe_float(votes.get("neutral")) or 0)
    if l == 2 and s == 0 and n == 1:
        return "LONG_L2_S0_N1"
    if l == 3 and s == 0 and n == 0:
        return "LONG_L3_S0_N0"
    if s == 2 and l == 0 and n == 1:
        return "SHORT_S2_L0_N1"
    if s == 3 and l == 0 and n == 0:
        return "SHORT_S3_L0_N0"
    if l == 1 and s == 1 and n == 1:
        return "MIXED_L1_S1_N1"
    if n == 2:
        return "MIXED_N2"
    if n == 3:
        return "MIXED_N3"
    return f"L{l}_S{s}_N{n}"


def metrics_for_trades(trades: list[dict[str, Any]]) -> dict[str, Any]:
    pnls = [trade_pnl(t) for t in trades]
    roes = [trade_roe(t) for t in trades]
    valid_pnls = [p for p in pnls if p is not None]
    wins = sum(1 for p in valid_pnls if p > 0)
    losses = sum(1 for p in valid_pnls if p < 0)
    gross_profit = sum(p for p in valid_pnls if p > 0)
    gross_loss = sum(p for p in valid_pnls if p < 0)
    mfe = [safe_float(t.get("mfe_roe")) for t in trades]
    mae = [safe_float(t.get("mae_roe")) for t in trades]
    durations = [safe_float(t.get("duration_minutes")) for t in trades]
    best = max(trades, key=lambda t: trade_roe(t) if trade_roe(t) is not None else -10**9, default=None)
    worst = min(trades, key=lambda t: trade_roe(t) if trade_roe(t) is not None else 10**9, default=None)
    fees = [
        safe_float(t.get("fee_usdt"))
        if safe_float(t.get("fee_usdt")) is not None
        else safe_float((t.get("metadata") or {}).get("fee_usdt") if isinstance(t.get("metadata"), dict) else None)
        for t in trades
    ]
    avg_mfe = mean(mfe)
    avg_mae = mean(mae)
    return {
        "trades": len(trades),
        "wins": wins,
        "losses": losses,
        "win_rate": wins / (wins + losses) if wins + losses else None,
        "gross_profit": gross_profit,
        "gross_loss": gross_loss,
        "net_pnl": sum(valid_pnls),
        "profit_factor": profit_factor(gross_profit, gross_loss),
        "avg_roe": mean(roes),
        "median_roe": med(roes),
        "best_roe": trade_roe(best) if best else None,
        "worst_roe": trade_roe(worst) if worst else None,
        "best_trade_id": best.get("trade_id") if best else None,
        "worst_trade_id": worst.get("trade_id") if worst else None,
        "avg_mfe_roe": avg_mfe,
        "avg_mae_roe": avg_mae,
        "mfe_mae_ratio": abs(avg_mfe / avg_mae) if avg_mfe is not None and avg_mae not in (None, 0) else None,
        "avg_duration_minutes": mean(durations),
        "median_duration_minutes": med(durations),
        "total_fees": sum(f for f in fees if f is not None) if any(f is not None for f in fees) else None,
    }


def max_consecutive_losses(trades: list[dict[str, Any]]) -> int:
    ordered = sorted(trades, key=lambda t: parse_dt(t.get("closed_at") or t.get("timestamp")) or datetime.min.replace(tzinfo=timezone.utc))
    streak = 0
    best = 0
    for trade in ordered:
        pnl = trade_pnl(trade)
        if pnl is not None and pnl < 0:
            streak += 1
            best = max(best, streak)
        elif pnl is not None and pnl > 0:
            streak = 0
    return best


def estimated_drawdown_from_snapshots(snapshots: list[dict[str, Any]]) -> dict[str, Any]:
    ordered = sorted(snapshots, key=lambda x: parse_dt(x.get("timestamp")) or datetime.min.replace(tzinfo=timezone.utc))
    balances = []
    for row in ordered:
        wallet = safe_float(row.get("wallet_balance"))
        unrealized = safe_float(row.get("unrealized_pnl")) or 0.0
        if wallet is not None:
            balances.append((row.get("timestamp"), wallet + unrealized))
    peak = None
    max_dd = 0.0
    max_dd_ts = None
    for ts, equity in balances:
        peak = equity if peak is None else max(peak, equity)
        if peak:
            dd = (equity - peak) / peak
            if dd < max_dd:
                max_dd = dd
                max_dd_ts = ts
    return {
        "estimated_drawdown": max_dd,
        "estimated_drawdown_pct": pct(max_dd),
        "estimated_drawdown_at": max_dd_ts,
        "sample_count": len(balances),
        "note": "Estimado desde snapshots de wallet+unrealized; puede duplicar estados intra-loop.",
    }


def latest_account_state(snapshots: list[dict[str, Any]]) -> dict[str, Any]:
    if not snapshots:
        return {}
    ordered = sorted(snapshots, key=lambda x: parse_dt(x.get("timestamp")) or datetime.min.replace(tzinfo=timezone.utc))
    latest = ordered[-1]
    latest_by_symbol: dict[str, dict[str, Any]] = {}
    for snap in ordered:
        symbols = snap.get("symbols")
        if isinstance(symbols, list):
            for pos in symbols:
                if isinstance(pos, dict) and pos.get("symbol"):
                    latest_by_symbol[str(pos["symbol"])] = dict(pos, snapshot_timestamp=snap.get("timestamp"))
        elif snap.get("symbol"):
            latest_by_symbol[str(snap["symbol"])] = dict(snap, snapshot_timestamp=snap.get("timestamp"))

    open_positions = [p for p in latest_by_symbol.values() if p.get("position_open") or safe_float(p.get("notional"))]
    total_unrealized = sum(safe_float(p.get("unrealized_pnl")) or 0.0 for p in open_positions)
    total_margin = sum(safe_float(p.get("margin_used")) or 0.0 for p in open_positions)
    total_notional = sum(safe_float(p.get("notional")) or 0.0 for p in open_positions)
    wallet = safe_float(latest.get("wallet_balance"))
    available = safe_float(latest.get("available_balance"))
    return {
        "latest_timestamp": latest.get("timestamp"),
        "wallet_balance": wallet,
        "available_balance": available,
        "unrealized_pnl_snapshot": safe_float(latest.get("unrealized_pnl")),
        "open_unrealized_pnl_estimated": total_unrealized,
        "equity_estimated": wallet + total_unrealized if wallet is not None else None,
        "open_positions_count": len(open_positions),
        "long_positions_count": sum(1 for p in open_positions if str(p.get("side")).upper() == "LONG"),
        "short_positions_count": sum(1 for p in open_positions if str(p.get("side")).upper() == "SHORT"),
        "total_margin_used_estimated": total_margin,
        "total_notional_estimated": total_notional,
        "margin_used_pct_estimated": total_margin / wallet if wallet else None,
        "notional_to_equity_estimated": total_notional / (wallet + total_unrealized) if wallet is not None and (wallet + total_unrealized) else None,
        "open_positions": sorted(open_positions, key=lambda p: str(p.get("symbol"))),
        "note": "Portfolio agregado desde el ultimo estado por simbolo en account_snapshots.",
    }


def grouped_metrics(trades: list[dict[str, Any]], key_fn) -> dict[str, dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for trade in trades:
        groups[str(key_fn(trade))].append(trade)
    return {key: metrics_for_trades(rows) for key, rows in sorted(groups.items())}


def most_common_exit(trades: list[dict[str, Any]]) -> str | None:
    counts = Counter(classify_exit_reason(t.get("exit_reason") or t.get("reason")) for t in trades)
    if not counts:
        return None
    return counts.most_common(1)[0][0]


def symbol_recommendation(symbol: str, trades: list[dict[str, Any]], side_metrics: dict[str, dict[str, Any]]) -> dict[str, Any]:
    m = metrics_for_trades(trades)
    reasons: list[str] = []
    rec = "NEED_MORE_DATA"
    if m["trades"] < 3:
        reasons.append("Menos de 3 cierres: muestra insuficiente para decision fuerte.")
    else:
        pf = m["profit_factor"]
        win_rate = m["win_rate"] or 0
        net = m["net_pnl"] or 0
        worst = m["worst_roe"] or 0
        short = side_metrics.get("SHORT", {})
        short_trades = short.get("trades", 0)
        short_net = short.get("net_pnl") or 0
        short_pf = short.get("profit_factor")
        if short_trades >= 2 and short_net < 0 and (short_pf is None or short_pf < 0.9):
            rec = "DISABLE_SHORTS"
            reasons.append("SHORT con perdida neta/profit factor debil en la muestra.")
        elif net < 0 and (pf is None or pf < 1.0) and win_rate < 0.45:
            rec = "SHADOW_ONLY"
            reasons.append("PnL neto negativo, win rate bajo y profit factor < 1.")
        elif net < 0 or (pf is not None and pf < 1.15) or worst < -0.25:
            rec = "REDUCE_RISK"
            reasons.append("Edge debil o peor ROE amplio; conviene reducir riesgo antes de escalar.")
        else:
            rec = "KEEP_LIVE"
            reasons.append("PnL/edge positivo dentro de la muestra analizada.")
    return {"recommendation": rec, "why": reasons}


def analyze_events(events: list[dict[str, Any]], closed_trades: list[dict[str, Any]]) -> dict[str, Any]:
    event_names = Counter(str(e.get("event") or e.get("type") or e.get("reason") or "UNKNOWN") for e in events)
    text_counts = Counter()
    symbol_counts: dict[str, Counter] = defaultdict(Counter)
    tracked = [
        "BREAK_EVEN_ARMED",
        "BREAK_EVEN_EXECUTED",
        "SL_MOVED",
        "aegis_break_even_stop_moved",
        "aegis_break_even_stop_move_skipped_immediate_trigger",
        "TRAILING_ACTIVATED",
        "MOVE_SL_TRAILING",
        "TRAILING_STOP",
        "AEGIS_EXIT_EYE_SHADOW_PROTECT",
        "AEGIS_EXIT_EYE_SHADOW_CLOSE",
        "AEGIS_EXIT_EYE_PROTECT_PROFIT",
        "AEGIS_EXIT_EYE_CLOSE_POSITION",
        "AEGIS_EXIT_EYE_NEUTRAL_DECAY",
        "AEGIS_EXIT_EYE_OPPOSITE_SIGNAL",
    ]
    for event in events:
        blob = json.dumps(event, ensure_ascii=False)
        for needle in tracked:
            if needle in blob:
                text_counts[needle] += 1
                symbol = str(event.get("symbol") or "UNKNOWN")
                symbol_counts[needle][symbol] += 1

    closes_by_reason = grouped_metrics(closed_trades, lambda t: classify_exit_reason(t.get("exit_reason") or t.get("reason")))
    break_even_trades = [t for t in closed_trades if classify_exit_reason(t.get("exit_reason") or t.get("reason")) == "BREAK_EVEN"]
    trailing_trades = [t for t in closed_trades if classify_exit_reason(t.get("exit_reason") or t.get("reason")) == "TRAILING_STOP"]
    exit_eye_trades = [
        t
        for t in closed_trades
        if classify_exit_reason(t.get("exit_reason") or t.get("reason")).startswith("AEGIS_EXIT_EYE")
    ]
    return {
        "event_name_counts_top": dict(event_names.most_common(50)),
        "tracked_event_counts": dict(text_counts),
        "tracked_event_symbols": {k: dict(v.most_common()) for k, v in symbol_counts.items()},
        "exit_reasons": closes_by_reason,
        "break_even": {
            "armed_count": text_counts.get("BREAK_EVEN_ARMED", 0),
            "executed_count": text_counts.get("BREAK_EVEN_EXECUTED", 0) + closes_by_reason.get("BREAK_EVEN", {}).get("trades", 0),
            "skipped_immediate_trigger_count": text_counts.get("aegis_break_even_stop_move_skipped_immediate_trigger", 0),
            "failed_count": event_names.get("BREAK_EVEN_FAILED", 0),
            "symbols_with_skips": dict(symbol_counts.get("aegis_break_even_stop_move_skipped_immediate_trigger", Counter()).most_common()),
            "closed_trade_metrics": metrics_for_trades(break_even_trades),
            "avg_peak_roe_when_executed": mean(safe_float(t.get("mfe_roe")) for t in break_even_trades),
            "avg_giveback_before_be": mean(
                (safe_float(t.get("mfe_roe")) or 0) - (trade_roe(t) or 0) for t in break_even_trades if safe_float(t.get("mfe_roe")) is not None and trade_roe(t) is not None
            ),
        },
        "trailing": {
            "activation_count": text_counts.get("TRAILING_ACTIVATED", 0),
            "move_sl_count": text_counts.get("MOVE_SL_TRAILING", 0) + text_counts.get("SL_MOVED", 0),
            "closures": closes_by_reason.get("TRAILING_STOP", {}).get("trades", 0),
            "closed_trade_metrics": metrics_for_trades(trailing_trades),
            "average_profit_captured_roe": mean(trade_roe(t) for t in trailing_trades),
            "average_giveback_from_peak_roe": mean(
                (safe_float(t.get("mfe_roe")) or 0) - (trade_roe(t) or 0) for t in trailing_trades if safe_float(t.get("mfe_roe")) is not None and trade_roe(t) is not None
            ),
            "trades_where_trailing_never_activated_and_ended_loss": sum(
                1 for t in closed_trades if (safe_float(t.get("mfe_roe")) or 0) < 0.15 and (trade_pnl(t) or 0) < 0
            ),
        },
        "exit_eye": {
            "neutral_decay_detected": text_counts.get("AEGIS_EXIT_EYE_NEUTRAL_DECAY", 0),
            "neutral_decay_closures": closes_by_reason.get("AEGIS_EXIT_EYE_NEUTRAL_DECAY", {}).get("trades", 0),
            "opposite_signal_closures": closes_by_reason.get("AEGIS_EXIT_EYE_OPPOSITE_SIGNAL", {}).get("trades", 0),
            "tracked_event_counts": {k: v for k, v in text_counts.items() if k.startswith("AEGIS_EXIT_EYE")},
            "closed_trade_metrics": metrics_for_trades(exit_eye_trades),
            "avg_peak_roe_before_close": mean(safe_float(t.get("mfe_roe")) for t in exit_eye_trades),
            "avg_giveback_from_peak": mean(
                (safe_float(t.get("mfe_roe")) or 0) - (trade_roe(t) or 0) for t in exit_eye_trades if safe_float(t.get("mfe_roe")) is not None and trade_roe(t) is not None
            ),
            "symbols": dict(Counter(str(t.get("symbol") or "UNKNOWN") for t in exit_eye_trades).most_common()),
        },
    }


def parse_pm2_logs() -> dict[str, Any]:
    log_dir = Path.home() / ".pm2" / "logs"
    files = sorted(log_dir.glob("01-Trading-Bot*.log")) + sorted(log_dir.glob("02-Aegis-API*.log"))
    counters = Counter()
    recent_counters = Counter()
    latency_by_symbol: dict[str, list[float]] = defaultdict(list)
    recent_latency_by_symbol: dict[str, list[float]] = defaultdict(list)
    recent_lines: dict[str, list[str]] = {}
    latency_re = re.compile(r"aegis_predict_latency\s+symbol=(\w+).*?total_ms=([0-9.]+).*?fallback_used=(\w+)", re.I)

    def scan_line(line: str, target_counters: Counter, target_latency: dict[str, list[float]]) -> None:
        low = line.lower()
        if "timeout" in low or "axioserror" in low:
            target_counters["timeout_or_axios_mentions"] += 1
        if "telegram_signal_failed" in line:
            target_counters["telegram_signal_failed"] += 1
        if "stale" in low:
            target_counters["stale_mentions"] += 1
        if "missing artifact" in low or "missing_artifact" in low:
            target_counters["missing_artifact_mentions"] += 1
        if "corrupt" in low:
            target_counters["corrupt_mentions"] += 1
        if "aegis_break_even_stop_move_skipped_immediate_trigger" in line:
            target_counters["be_skipped_immediate_trigger_pm2"] += 1
        match = latency_re.search(line)
        if match:
            target_latency[match.group(1)].append(float(match.group(2)))

    for path in files:
        try:
            lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        except Exception:
            continue
        recent_lines[str(path)] = lines[-300:]
        for line in lines:
            scan_line(line, counters, latency_by_symbol)
        for line in lines[-300:]:
            scan_line(line, recent_counters, recent_latency_by_symbol)

    def summarize_latency(source: dict[str, list[float]]) -> dict[str, dict[str, Any]]:
        summary = {}
        for symbol, values in sorted(source.items()):
            summary[symbol] = {
                "count": len(values),
                "avg_ms": mean(values),
                "max_ms": max(values) if values else None,
                "median_ms": med(values),
            }
        return summary

    latency_summary = summarize_latency(latency_by_symbol)
    recent_latency_summary = summarize_latency(recent_latency_by_symbol)
    return {
        "files_scanned": [str(p) for p in files],
        "counters": dict(counters),
        "recent_tail_counters": dict(recent_counters),
        "predict_latency_by_symbol": latency_summary,
        "recent_predict_latency_by_symbol": recent_latency_summary,
        "recent_tail_files": {path: len(lines) for path, lines in recent_lines.items()},
        "note": "PM2 logs se leyeron como archivos locales; recent_* usa las ultimas 300 lineas por archivo.",
    }


def analyze_retraining_and_freshness(signals: list[dict[str, Any]]) -> dict[str, Any]:
    turbo_dir = ROOT / "aegis_alpha" / "logs" / "turbo"
    retrain_dir = ROOT / "aegis_alpha" / "logs" / "turbo_retrain"
    model_dir = ROOT / "aegis_alpha" / "models" / "turbo"
    train_reports = sorted(turbo_dir.glob("turbo_train_report_*.json")) if turbo_dir.exists() else []
    snapshot_reports = sorted(turbo_dir.glob("turbo_snapshot_refresh_*.json")) if turbo_dir.exists() else []
    retrain_reports = sorted(retrain_dir.glob("turbo_retrain_*.json")) if retrain_dir.exists() else []

    latest_signal_by_symbol: dict[str, dict[str, Any]] = {}
    for signal in sorted(signals, key=lambda s: parse_dt(s.get("timestamp")) or datetime.min.replace(tzinfo=timezone.utc)):
        symbol = signal.get("symbol")
        if symbol:
            latest_signal_by_symbol[str(symbol)] = signal

    freshness = {}
    for symbol in SYMBOLS:
        row = latest_signal_by_symbol.get(symbol)
        fr = row.get("freshness") if isinstance(row, dict) else None
        freshness[symbol] = {
            "latest_signal_timestamp": row.get("timestamp") if row else None,
            "is_fresh": fr.get("is_fresh") if isinstance(fr, dict) else None,
            "stale": fr.get("stale") if isinstance(fr, dict) else None,
            "snapshot_age_seconds": fr.get("snapshot_age_seconds") if isinstance(fr, dict) else None,
            "turbo_score": row.get("turbo_score") if row else None,
            "fallback_used": row.get("fallback_used") or (row.get("metadata") or {}).get("fallback_used") if row else None,
        }

    manifests = {}
    for symbol in SYMBOLS:
        path = model_dir / symbol / "active_manifest.json"
        parsed = read_json(path)
        manifests[symbol] = {
            "exists": path.exists(),
            "path": str(path),
            "updated_at": parsed.get("updated_at") if isinstance(parsed, dict) else None,
            "created_at": parsed.get("created_at") if isinstance(parsed, dict) else None,
            "model_version": parsed.get("model_version") if isinstance(parsed, dict) else None,
            "keys": sorted(parsed.keys()) if isinstance(parsed, dict) else [],
        }

    def newest(paths: list[Path], limit: int = 10) -> list[str]:
        return [str(p) for p in sorted(paths, key=lambda p: p.stat().st_mtime, reverse=True)[:limit]]

    return {
        "latest_train_reports": newest(train_reports),
        "latest_snapshot_refresh_reports": newest(snapshot_reports),
        "latest_retrain_reports": newest(retrain_reports),
        "freshness_by_symbol_from_signals": freshness,
        "active_manifests": manifests,
        "models_updated": all(v["exists"] for v in manifests.values()),
        "fresh_symbols": [s for s, v in freshness.items() if v.get("is_fresh") is True],
        "stale_symbols": [s for s, v in freshness.items() if v.get("is_fresh") is False or v.get("stale") is True],
    }


def recommendation_items(report: dict[str, Any]) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    by_symbol = report["by_symbol"]
    long_short = report["long_vs_short"]["global"]
    open_state = report["portfolio_exposure"]
    score = report["score_buckets"]["global"]
    votes = report["vote_patterns"]["global"]
    events = report["event_analysis"]

    short = long_short.get("SHORT", {})
    if short.get("trades", 0) >= 2 and (short.get("net_pnl") or 0) < 0:
        items.append({
            "area": "A. Cambios de configuracion inmediatos",
            "priority": "HIGH",
            "recommendation": "Endurecer entradas SHORT o exigir 3/3 donde la muestra SHORT sea negativa.",
            "reason": "SHORT global tiene PnL neto negativo en la muestra.",
            "evidence": {"short_metrics": short},
            "risk": "Puede perder oportunidades en reversals, pero reduce cola izquierda en regimen alcista/lateral.",
        })

    for symbol, payload in by_symbol.items():
        rec = payload["recommendation"]["recommendation"]
        if rec in {"REDUCE_RISK", "SHADOW_ONLY", "DISABLE_SHORTS"}:
            items.append({
                "area": "A. Cambios de configuracion inmediatos",
                "priority": "HIGH" if rec != "REDUCE_RISK" else "MEDIUM",
                "recommendation": f"{symbol}: {rec}.",
                "reason": "; ".join(payload["recommendation"]["why"]),
                "evidence": payload["metrics"],
                "risk": "Cambiar riesgo/estado puede reducir drawdown, pero tambien baja frecuencia y aprendizaje live.",
            })

    weak_buckets = [
        (label, m)
        for label, m in score.items()
        if label != "UNKNOWN" and m.get("trades", 0) >= 2 and (m.get("profit_factor") is None or m.get("profit_factor") < 1)
    ]
    if weak_buckets:
        items.append({
            "area": "A. Cambios de configuracion inmediatos",
            "priority": "MEDIUM",
            "recommendation": "Revisar threshold minimo por simbolo antes de subirlo globalmente.",
            "reason": "Hay buckets de score con profit factor debil; el score no debe asumirse calibrado globalmente.",
            "evidence": {label: m for label, m in weak_buckets},
            "risk": "Un threshold global puede eliminar trades buenos de simbolos donde el score bajo si funciona.",
        })

    if open_state.get("open_positions_count", 0) >= 6 or (open_state.get("margin_used_pct_estimated") or 0) > 0.5:
        items.append({
            "area": "B. Cambios de riesgo/portfolio",
            "priority": "HIGH",
            "recommendation": "Agregar caps de portfolio: max_open_positions, max_same_direction_positions y max_margin_used_pct.",
            "reason": "La exposicion agregada puede amplificar un movimiento correlacionado cripto.",
            "evidence": open_state,
            "risk": "Reduce simultaneidad y puede dejar senales sin ejecutar durante ventanas fuertes.",
        })

    be = events["break_even"]
    if be.get("skipped_immediate_trigger_count", 0) or report["pm2_log_analysis"]["counters"].get("be_skipped_immediate_trigger_pm2", 0):
        items.append({
            "area": "A. Cambios de configuracion inmediatos",
            "priority": "MEDIUM",
            "recommendation": "Evaluar be_roe/offset por simbolo; no cambiar global sin backtest de eventos BE.",
            "reason": "Se detectaron skips de movimiento BE por riesgo de trigger inmediato.",
            "evidence": {"event_be": be, "pm2": report["pm2_log_analysis"]["counters"]},
            "risk": "BE demasiado temprano puede sacar trades buenos por ruido; demasiado tarde devuelve ganancias.",
        })

    if events["trailing"].get("trades_where_trailing_never_activated_and_ended_loss", 0) > 0:
        items.append({
            "area": "A. Cambios de configuracion inmediatos",
            "priority": "MEDIUM",
            "recommendation": "Analizar trailing_activation_roe por simbolo, especialmente altcoins volatiles.",
            "reason": "Hay trades perdedores que nunca llegaron a activar trailing.",
            "evidence": events["trailing"],
            "risk": "Activar trailing antes puede capturar menos upside en tendencias limpias.",
        })

    items.extend([
        {
            "area": "C. Cambios de modelo/features",
            "priority": "HIGH",
            "recommendation": "Calibrar score por simbolo y lado, no solo global.",
            "reason": "Los buckets y LONG/SHORT muestran comportamiento heterogeneo por simbolo.",
            "evidence": {"score_buckets": score, "long_short": long_short},
            "risk": "Requiere mas datos y validacion walk-forward para no sobreajustar.",
        },
        {
            "area": "C. Cambios de modelo/features",
            "priority": "MEDIUM",
            "recommendation": "Agregar features 15m/1h, regime classifier y tail-risk model.",
            "reason": "El sistema multi-symbol necesita distinguir tendencia, chop y shocks correlacionados.",
            "evidence": {"vote_patterns": votes},
            "risk": "Mas features aumentan complejidad y pueden degradar si no se validan por simbolo.",
        },
        {
            "area": "D. Cosas que NO conviene tocar todavia",
            "priority": "HIGH",
            "recommendation": "No subir leverage ni agregar mas simbolos hasta cerrar caps de portfolio y calibracion SHORT.",
            "reason": "La muestra live aun es pequena y la exposicion correlacionada importa mas que la latencia ya corregida.",
            "evidence": {"portfolio": open_state, "trades": report["global_metrics"]["closed_trades"]},
            "risk": "Esperar limita upside, pero evita escalar un edge todavia no estable.",
        },
    ])
    return items[:18]


def render_table(rows: list[dict[str, Any]], columns: list[tuple[str, str]]) -> str:
    if not rows:
        return "_Sin datos._\n"
    header = "| " + " | ".join(label for label, _ in columns) + " |"
    sep = "| " + " | ".join("---" for _ in columns) + " |"
    body = []
    for row in rows:
        vals = []
        for _, key in columns:
            val = row.get(key)
            if isinstance(val, float):
                vals.append(format_num(val))
            elif val is None:
                vals.append("n/a")
            else:
                vals.append(str(val))
        body.append("| " + " | ".join(vals) + " |")
    return "\n".join([header, sep, *body]) + "\n"


def compact_metrics_row(name: str, metrics: dict[str, Any]) -> dict[str, Any]:
    return {
        "name": name,
        "trades": metrics.get("trades"),
        "wins": metrics.get("wins"),
        "losses": metrics.get("losses"),
        "win_rate": format_pct(pct(metrics.get("win_rate"))),
        "net_pnl": format_num(metrics.get("net_pnl")),
        "profit_factor": format_num(metrics.get("profit_factor")),
        "avg_roe": format_pct(pct(metrics.get("avg_roe"))),
        "worst_roe": format_pct(pct(metrics.get("worst_roe"))),
    }


def render_markdown(report: dict[str, Any], summary_only: bool = False) -> str:
    gm = report["global_metrics"]
    lines: list[str] = []
    title = "Aegis Turbo Live - Resumen para ChatGPT" if summary_only else "Aegis Turbo Live - Diagnostico Deep Dive"
    lines.append(f"# {title}")
    lines.append("")
    lines.append(f"- Generado: {report['generated_at_utc']}")
    lines.append(f"- Periodo analizado: {report['period']['start']} a {report['period']['end']}")
    lines.append(f"- Simbolos activos: {', '.join(SYMBOLS)}")
    lines.append(f"- Nota de precision: {report['precision_note']}")
    lines.append("")
    lines.append("## 0. Estado general")
    lines.append("")
    gs = report.get("general_state", {})
    root_status = (gs.get("root_git_status_short") or {}).get("stdout") or "clean"
    bot_status = (gs.get("bot_git_status_short") or {}).get("stdout") or "clean"
    lines.append(f"- Repo raiz `git status --short`: `{root_status}`")
    lines.append(f"- Repo bot `git status --short`: `{bot_status}`")
    pm2_rc = (gs.get("pm2_list") or {}).get("returncode")
    lines.append(f"- `pm2 list` returncode: {pm2_rc}")
    lines.append("")
    lines.append("## 1. PnL global")
    lines.append("")
    lines.append(f"- Starting balance aproximado: {format_num(gm.get('starting_balance_approx'))} USDT")
    lines.append(f"- Ending wallet aproximado: {format_num(gm.get('ending_balance_approx'))} USDT")
    lines.append(f"- Equity estimado actual: {format_num(gm.get('current_equity_estimated'))} USDT")
    lines.append(f"- PnL cerrado total: {format_num(gm.get('closed_pnl_total'))} USDT")
    lines.append(f"- PnL abierto/no realizado estimado: {format_num(gm.get('open_unrealized_pnl'))} USDT")
    lines.append(f"- Trades cerrados: {gm.get('closed_trades')} / abiertos actuales: {gm.get('open_trades')}")
    lines.append(f"- Win rate: {format_pct(pct(gm.get('win_rate')))}")
    lines.append(f"- Profit factor: {format_num(gm.get('profit_factor'))}")
    lines.append(f"- Max consecutive losses: {gm.get('max_consecutive_losses')}")
    lines.append(f"- Drawdown estimado: {format_pct(gm.get('estimated_drawdown_pct'))}")
    lines.append("")

    sym_rows = []
    for symbol, payload in report["by_symbol"].items():
        row = compact_metrics_row(symbol, payload["metrics"])
        row["recommendation"] = payload["recommendation"]["recommendation"]
        row["exit"] = payload.get("most_common_exit_reason") or "n/a"
        sym_rows.append(row)
    lines.append("## 2. Rendimiento por simbolo")
    lines.append("")
    lines.append(render_table(sym_rows, [
        ("Simbolo", "name"),
        ("Trades", "trades"),
        ("W", "wins"),
        ("L", "losses"),
        ("Win", "win_rate"),
        ("Net", "net_pnl"),
        ("PF", "profit_factor"),
        ("Avg ROE", "avg_roe"),
        ("Worst ROE", "worst_roe"),
        ("Exit comun", "exit"),
        ("Recomendacion", "recommendation"),
    ]))

    lines.append("## 3. LONG vs SHORT")
    lines.append("")
    side_rows = [compact_metrics_row(side, m) for side, m in report["long_vs_short"]["global"].items()]
    lines.append(render_table(side_rows, [
        ("Lado", "name"),
        ("Trades", "trades"),
        ("W", "wins"),
        ("L", "losses"),
        ("Win", "win_rate"),
        ("Net", "net_pnl"),
        ("PF", "profit_factor"),
        ("Worst ROE", "worst_roe"),
    ]))
    lines.append(report["answers"]["long_vs_short"])
    lines.append("")

    lines.append("## 4. Score buckets")
    lines.append("")
    bucket_rows = [compact_metrics_row(label, m) for label, m in report["score_buckets"]["global"].items()]
    lines.append(render_table(bucket_rows, [
        ("Bucket", "name"),
        ("Trades", "trades"),
        ("W", "wins"),
        ("L", "losses"),
        ("Win", "win_rate"),
        ("Net", "net_pnl"),
        ("PF", "profit_factor"),
        ("Avg ROE", "avg_roe"),
        ("Worst", "worst_roe"),
    ]))
    lines.append(report["answers"]["score_buckets"])
    lines.append("")

    lines.append("## 5. Votos 2/3 vs 3/3")
    lines.append("")
    vote_rows = [compact_metrics_row(label, m) for label, m in report["vote_patterns"]["global"].items()]
    lines.append(render_table(vote_rows, [
        ("Patron", "name"),
        ("Trades", "trades"),
        ("W", "wins"),
        ("L", "losses"),
        ("Win", "win_rate"),
        ("Net", "net_pnl"),
        ("PF", "profit_factor"),
        ("Worst", "worst_roe"),
    ]))
    lines.append(report["answers"]["vote_patterns"])
    lines.append("")

    ev = report["event_analysis"]
    lines.append("## 6. Exit reasons")
    lines.append("")
    exit_rows = [compact_metrics_row(label, m) for label, m in ev["exit_reasons"].items()]
    lines.append(render_table(exit_rows, [
        ("Reason", "name"),
        ("Trades", "trades"),
        ("W", "wins"),
        ("L", "losses"),
        ("Win", "win_rate"),
        ("Net", "net_pnl"),
        ("PF", "profit_factor"),
        ("Avg ROE", "avg_roe"),
        ("Worst", "worst_roe"),
    ]))
    lines.append(report["answers"]["exit_reasons"])
    lines.append("")

    lines.append("## 7. Break-even")
    lines.append("")
    be = ev["break_even"]
    lines.append(f"- Armed: {be.get('armed_count')}")
    lines.append(f"- Executed/cierres BE: {be.get('executed_count')}")
    lines.append(f"- Skipped immediate trigger: {be.get('skipped_immediate_trigger_count')} (PM2: {report['pm2_log_analysis']['counters'].get('be_skipped_immediate_trigger_pm2', 0)})")
    lines.append(f"- Avg ROE BE: {format_pct(pct(be['closed_trade_metrics'].get('avg_roe')))}")
    lines.append(f"- Avg peak ROE BE: {format_pct(pct(be.get('avg_peak_roe_when_executed')))}")
    lines.append(report["answers"]["break_even"])
    lines.append("")

    lines.append("## 8. Trailing")
    lines.append("")
    tr = ev["trailing"]
    lines.append(f"- Activations: {tr.get('activation_count')}")
    lines.append(f"- Closures: {tr.get('closures')}")
    lines.append(f"- Avg captured ROE: {format_pct(pct(tr.get('average_profit_captured_roe')))}")
    lines.append(f"- Avg giveback from peak: {format_pct(pct(tr.get('average_giveback_from_peak_roe')))}")
    lines.append(f"- Losses sin activar trailing: {tr.get('trades_where_trailing_never_activated_and_ended_loss')}")
    lines.append(report["answers"]["trailing"])
    lines.append("")

    lines.append("## 9. Exit Eye")
    lines.append("")
    ee = ev["exit_eye"]
    lines.append(f"- Neutral decay detectado: {ee.get('neutral_decay_detected')}")
    lines.append(f"- Neutral decay cierres: {ee.get('neutral_decay_closures')}")
    lines.append(f"- Opposite signal cierres: {ee.get('opposite_signal_closures')}")
    lines.append(f"- Avg ROE cierres Exit Eye: {format_pct(pct(ee['closed_trade_metrics'].get('avg_roe')))}")
    lines.append(f"- Avg giveback: {format_pct(pct(ee.get('avg_giveback_from_peak')))}")
    lines.append(report["answers"]["exit_eye"])
    lines.append("")

    lines.append("## 10. Portfolio exposure y posiciones abiertas")
    lines.append("")
    pe = report["portfolio_exposure"]
    lines.append(f"- Open positions: {pe.get('open_positions_count')} ({pe.get('long_positions_count')} LONG / {pe.get('short_positions_count')} SHORT)")
    lines.append(f"- Margin used estimado: {format_num(pe.get('total_margin_used_estimated'))} USDT ({format_pct(pct(pe.get('margin_used_pct_estimated')))})")
    lines.append(f"- Notional estimado: {format_num(pe.get('total_notional_estimated'))} USDT")
    lines.append(f"- Notional/equity estimado: {format_num(pe.get('notional_to_equity_estimated'))}x")
    pos_rows = []
    for p in pe.get("open_positions", []):
        pos_rows.append({
            "symbol": p.get("symbol"),
            "side": p.get("side"),
            "entry": format_num(p.get("entry_price")),
            "mark": format_num(p.get("mark_price")),
            "roe": format_pct(pct(safe_float(p.get("roe")))),
            "upnl": format_num(p.get("unrealized_pnl")),
            "margin": format_num(p.get("margin_used")),
            "notional": format_num(p.get("notional")),
        })
    lines.append(render_table(pos_rows, [
        ("Symbol", "symbol"),
        ("Side", "side"),
        ("Entry", "entry"),
        ("Mark", "mark"),
        ("ROE", "roe"),
        ("uPnL", "upnl"),
        ("Margin", "margin"),
        ("Notional", "notional"),
    ]))
    lines.append(report["answers"]["portfolio_exposure"])
    lines.append("")

    lines.append("## 11. Retraining / freshness / API")
    lines.append("")
    rf = report["retraining_freshness_api"]
    lines.append(f"- Fresh symbols por signals: {', '.join(rf['fresh_symbols']) or 'ninguno'}")
    lines.append(f"- Stale symbols por signals: {', '.join(rf['stale_symbols']) or 'ninguno'}")
    lines.append(f"- Manifests presentes: {sum(1 for v in rf['active_manifests'].values() if v.get('exists'))}/{len(SYMBOLS)}")
    lines.append(f"- PM2 timeout/Axios mentions historicas: {report['pm2_log_analysis']['counters'].get('timeout_or_axios_mentions', 0)}")
    lines.append(f"- PM2 timeout/Axios mentions recientes: {report['pm2_log_analysis']['recent_tail_counters'].get('timeout_or_axios_mentions', 0)}")
    lines.append(f"- PM2 telegram_signal_failed reciente: {report['pm2_log_analysis']['recent_tail_counters'].get('telegram_signal_failed', 0)}")
    latency_rows = []
    for symbol, m in report["pm2_log_analysis"]["recent_predict_latency_by_symbol"].items():
        latency_rows.append({
            "symbol": symbol,
            "count": m.get("count"),
            "avg": format_num(m.get("avg_ms"), 2),
            "median": format_num(m.get("median_ms"), 2),
            "max": format_num(m.get("max_ms"), 2),
        })
    lines.append(render_table(latency_rows, [
        ("Symbol", "symbol"),
        ("N", "count"),
        ("Avg ms", "avg"),
        ("Median ms", "median"),
        ("Max ms", "max"),
    ]))
    lines.append(report["answers"]["retraining_freshness_api"])
    lines.append("")

    lines.append("## 12. Recomendaciones")
    lines.append("")
    for idx, item in enumerate(report["recommendations"], 1):
        lines.append(f"{idx}. [{item['priority']}] {item['area']}: {item['recommendation']}")
        lines.append(f"   - Motivo: {item['reason']}")
        lines.append(f"   - Riesgo: {item['risk']}")
    lines.append("")

    if not summary_only:
        lines.append("## 13. Warnings y calidad de datos")
        lines.append("")
        for warning in report["warnings"]:
            lines.append(f"- {warning}")
        lines.append("")
        lines.append("## 14. Preguntas abiertas para ChatGPT")
    else:
        lines.append("## 13. Preguntas abiertas para ChatGPT")
    lines.append("")
    for question in report["open_questions"]:
        lines.append(f"- {question}")
    lines.append("")

    if summary_only:
        return "\n".join(lines)

    lines.append("## 15. Archivos usados")
    lines.append("")
    for key, value in report["inputs"].items():
        if isinstance(value, list):
            lines.append(f"- {key}: {len(value)} archivos")
        else:
            lines.append(f"- {key}: {value}")
    lines.append("")
    return "\n".join(lines)


def build_answers(report: dict[str, Any]) -> dict[str, str]:
    side = report["long_vs_short"]["global"]
    long_net = side.get("LONG", {}).get("net_pnl")
    short_net = side.get("SHORT", {}).get("net_pnl")
    short_pf = side.get("SHORT", {}).get("profit_factor")
    short_symbols_ok = []
    short_symbols_bad = []
    for symbol, sm in report["long_vs_short"]["by_symbol"].items():
        sh = sm.get("SHORT", {})
        if sh.get("trades", 0) >= 1:
            if (sh.get("net_pnl") or 0) > 0 and (sh.get("profit_factor") is None or sh.get("profit_factor") >= 1):
                short_symbols_ok.append(symbol)
            elif (sh.get("net_pnl") or 0) < 0:
                short_symbols_bad.append(symbol)

    score_metrics = report["score_buckets"]["global"]
    best_bucket = max(
        ((k, v) for k, v in score_metrics.items() if k != "UNKNOWN" and v.get("trades", 0) > 0),
        key=lambda kv: kv[1].get("profit_factor") if kv[1].get("profit_factor") is not None else -1,
        default=(None, {}),
    )

    votes = report["vote_patterns"]["global"]
    long_2 = votes.get("LONG_L2_S0_N1", {})
    long_3 = votes.get("LONG_L3_S0_N0", {})
    short_2 = votes.get("SHORT_S2_L0_N1", {})
    short_3 = votes.get("SHORT_S3_L0_N0", {})

    ev = report["event_analysis"]
    strongest_exit = max(
        ev["exit_reasons"].items(),
        key=lambda kv: kv[1].get("net_pnl") if kv[1].get("net_pnl") is not None else -10**9,
        default=("UNKNOWN", {}),
    )
    weakest_exit = min(
        ev["exit_reasons"].items(),
        key=lambda kv: kv[1].get("net_pnl") if kv[1].get("net_pnl") is not None else 10**9,
        default=("UNKNOWN", {}),
    )

    return {
        "long_vs_short": (
            f"Respuesta: LONG net={format_num(long_net)} USDT y SHORT net={format_num(short_net)} USDT. "
            f"SHORT {'esta danando' if short_net is not None and short_net < 0 else 'no aparece danando globalmente'} en esta muestra "
            f"(PF SHORT={format_num(short_pf)}). SHORT funciona mejor en: {', '.join(short_symbols_ok) or 'sin evidencia suficiente'}; "
            f"conviene endurecer/desactivar en: {', '.join(short_symbols_bad) or 'sin evidencia suficiente'}. "
            "Exigir 3/3 para SHORT tiene sentido si S=2 muestra peor PF que S=3; revisar tabla de votos antes de hacerlo global."
        ),
        "score_buckets": (
            f"Respuesta: el mejor bucket por PF observado es {best_bucket[0] or 'n/a'} con PF={format_num(best_bucket[1].get('profit_factor'))}. "
            "No asumir que score predice profit de forma monotona si buckets altos no dominan net/PF. "
            "El score minimo debe calibrarse por simbolo cuando haya al menos 5-10 cierres por simbolo."
        ),
        "vote_patterns": (
            f"Respuesta: LONG 2/3 net={format_num(long_2.get('net_pnl'))}, LONG 3/3 net={format_num(long_3.get('net_pnl'))}; "
            f"SHORT 2/3 net={format_num(short_2.get('net_pnl'))}, SHORT 3/3 net={format_num(short_3.get('net_pnl'))}. "
            "Si 3/3 tiene menor frecuencia pero mejor PF/worst loss, usarlo para SHORT y altcoins volatiles es razonable; si no hay muestra, marcar NEED_MORE_DATA."
        ),
        "exit_reasons": (
            f"Respuesta: la salida que mas aporta por PnL es {strongest_exit[0]} ({format_num(strongest_exit[1].get('net_pnl'))} USDT). "
            f"La salida con mas perdida neta es {weakest_exit[0]} ({format_num(weakest_exit[1].get('net_pnl'))} USDT). "
            "Stop/trailing/BE deben evaluarse por giveback y MAE, no solo por conteo."
        ),
        "break_even": (
            "Respuesta: be_roe=0.08 no se debe cambiar globalmente con muestra pequena. "
            "Si hay skips immediate trigger o giveback alto antes de BE, conviene probar BE por simbolo; 0.05 en altcoins puede ayudar, pero aumenta salidas por ruido. "
            "El offset actual parece agresivo solo si los skips se concentran en simbolos especificos."
        ),
        "trailing": (
            "Respuesta: trailing_activation_roe=0.15 puede estar alto si varias operaciones alcanzan MFE positivo y terminan negativas sin activarlo. "
            "Callback 0.08 debe validarse por simbolo; altcoins pueden necesitar activacion antes y callback mas amplio."
        ),
        "exit_eye": (
            "Respuesta: close_on_neutral_decay ayuda si sus cierres tienen ROE mejor que el drawdown posterior esperado. "
            "Con poca muestra, require_consecutive_neutral_close=2 es un punto medio sano; 1 puede cerrar demasiado pronto y 3 puede llegar tarde."
        ),
        "portfolio_exposure": (
            "Respuesta: el riesgo principal es correlacion cripto multi-symbol. "
            "Si hay muchas posiciones en la misma direccion o margin_used_pct alto, se necesitan max_open_positions, max_same_direction_positions y max_margin_used_pct antes de subir riesgo."
        ),
        "retraining_freshness_api": (
            "Respuesta: modelos/snapshots se consideran sanos si manifests existen, los ultimos signals por simbolo son fresh y PM2 no muestra latencias/timeout recientes. "
            "Cualquier stale symbol o missing artifact debe pasar a prioridad HIGH antes de optimizar thresholds."
        ),
    }


def build_report(dates: list[str], timestamp: str) -> dict[str, Any]:
    available = available_jsonl_dates()
    trade_read = read_jsonl_files(date_paths("turbo_trades", dates))
    signal_read = read_jsonl_files(date_paths("turbo_signals", dates))
    event_read = read_jsonl_files(date_paths("turbo_trade_events", dates))
    account_read = read_jsonl_files(date_paths("account_snapshots", dates))

    open_by_id: dict[str, dict[str, Any]] = {}
    raw_closed: list[dict[str, Any]] = []
    for row in trade_read.rows:
        tid = str(row.get("trade_id") or "")
        status = str(row.get("status") or "").upper()
        if status == "OPEN" and tid:
            open_by_id[tid] = row
        elif status == "CLOSED":
            raw_closed.append(row)

    closed_trades = [enrich_trade(row, open_by_id) for row in raw_closed]
    closed_ids = {str(t.get("trade_id")) for t in closed_trades if t.get("trade_id")}
    open_records = [row for tid, row in open_by_id.items() if tid not in closed_ids]

    # Enrich closed trades from signal rows if a direct signal id is not present.
    # Trade logs usually carry score/votes on OPEN rows; this is a fallback for partial logs.
    latest_signal_by_symbol_action: dict[tuple[str, str], dict[str, Any]] = {}
    for sig in sorted(signal_read.rows, key=lambda s: parse_dt(s.get("timestamp")) or datetime.min.replace(tzinfo=timezone.utc)):
        action = str(sig.get("final_action") or sig.get("gated_action") or sig.get("raw_action") or "").upper()
        symbol = str(sig.get("symbol") or "")
        if symbol and action:
            latest_signal_by_symbol_action[(symbol, action)] = sig
    for trade in closed_trades:
        if trade.get("turbo_score") is None:
            sig = latest_signal_by_symbol_action.get((str(trade.get("symbol")), str(trade.get("side")).upper()))
            if sig:
                trade["turbo_score"] = sig.get("turbo_score")
                trade["votes"] = sig.get("votes")

    account_state = latest_account_state(account_read.rows)
    drawdown = estimated_drawdown_from_snapshots(account_read.rows)
    trade_metrics = metrics_for_trades(closed_trades)
    first_snap = min(account_read.rows, key=lambda x: parse_dt(x.get("timestamp")) or datetime.max.replace(tzinfo=timezone.utc), default={})
    last_snap = max(account_read.rows, key=lambda x: parse_dt(x.get("timestamp")) or datetime.min.replace(tzinfo=timezone.utc), default={})

    by_symbol = {}
    by_symbol_side = {}
    for symbol in SYMBOLS:
        rows = [t for t in closed_trades if t.get("symbol") == symbol]
        side_m = grouped_metrics(rows, lambda t: str(t.get("side") or "UNKNOWN").upper())
        by_symbol_side[symbol] = side_m
        by_symbol[symbol] = {
            "metrics": metrics_for_trades(rows),
            "most_common_exit_reason": most_common_exit(rows),
            "current_open_position": next((p for p in account_state.get("open_positions", []) if p.get("symbol") == symbol), None),
            "recommendation": symbol_recommendation(symbol, rows, side_m),
        }

    long_vs_short = {
        "global": grouped_metrics(closed_trades, lambda t: str(t.get("side") or "UNKNOWN").upper()),
        "by_symbol": by_symbol_side,
    }
    score_buckets = {
        "global": grouped_metrics(closed_trades, lambda t: score_bucket(t.get("turbo_score"))),
        "by_symbol": {symbol: grouped_metrics([t for t in closed_trades if t.get("symbol") == symbol], lambda t: score_bucket(t.get("turbo_score"))) for symbol in SYMBOLS},
    }
    vote_patterns = {
        "global": grouped_metrics(closed_trades, lambda t: vote_pattern(t.get("votes"))),
        "by_symbol": {symbol: grouped_metrics([t for t in closed_trades if t.get("symbol") == symbol], lambda t: vote_pattern(t.get("votes"))) for symbol in SYMBOLS},
    }
    event_analysis = analyze_events(event_read.rows, closed_trades)
    pm2 = parse_pm2_logs()
    freshness = analyze_retraining_and_freshness(signal_read.rows)

    warnings = []
    for name, read in [
        ("turbo_trades", trade_read),
        ("turbo_signals", signal_read),
        ("turbo_trade_events", event_read),
        ("account_snapshots", account_read),
    ]:
        if read.corrupt_lines:
            warnings.append(f"{name}: {read.corrupt_lines} lineas JSON corruptas saltadas.")
        if read.missing_files:
            warnings.append(f"{name}: {read.missing_files} archivos esperados no existen para el periodo.")
    if len(closed_trades) < 20:
        warnings.append("Muestra de trades cerrados pequena; recomendaciones de thresholds/modelo son provisionales.")
    older = sorted(set(available.get("bot_aegis", [])) - set(dates))
    if older:
        warnings.append(f"Hay logs bot_aegis fuera del periodo no incluidos: {', '.join(older)}.")
    if freshness["stale_symbols"]:
        warnings.append(f"Symbols con freshness stale en signals: {', '.join(freshness['stale_symbols'])}.")
    if pm2["counters"].get("timeout_or_axios_mentions", 0):
        warnings.append("PM2 contiene menciones historicas de timeout/Axios; revisar timestamps antes de asumir que son actuales.")

    report: dict[str, Any] = {
        "generated_at_utc": timestamp,
        "general_state": {
            "root_git_status_short": run_readonly_command(["git", "status", "--short"], ROOT),
            "bot_git_status_short": run_readonly_command(["git", "status", "--short"], BOT),
            "pm2_list": run_readonly_command(["pm2", "list", "--no-color"], ROOT),
        },
        "period": {"start": min(dates), "end": max(dates), "dates": dates, "available_dates": available},
        "symbols": SYMBOLS,
        "precision_note": "PnL y exposicion se calculan desde logs locales. Campos con metadata.estimated o sin fills exactos se tratan como estimados.",
        "global_metrics": {
            **trade_metrics,
            "total_trades": len(closed_trades) + len(open_records),
            "closed_trades": len(closed_trades),
            "open_trades": len(open_records),
            "closed_pnl_total": trade_metrics.get("net_pnl"),
            "open_unrealized_pnl": account_state.get("open_unrealized_pnl_estimated"),
            "starting_balance_approx": safe_float(first_snap.get("wallet_balance")),
            "ending_balance_approx": safe_float(last_snap.get("wallet_balance")),
            "current_wallet_balance": account_state.get("wallet_balance"),
            "current_equity_estimated": account_state.get("equity_estimated"),
            "max_consecutive_losses": max_consecutive_losses(closed_trades),
            **drawdown,
        },
        "by_symbol": by_symbol,
        "long_vs_short": long_vs_short,
        "score_buckets": score_buckets,
        "vote_patterns": vote_patterns,
        "event_analysis": event_analysis,
        "portfolio_exposure": account_state,
        "retraining_freshness_api": freshness,
        "pm2_log_analysis": pm2,
        "inputs": {
            "trade_files": [str(p) for p in date_paths("turbo_trades", dates)],
            "signal_files": [str(p) for p in date_paths("turbo_signals", dates)],
            "event_files": [str(p) for p in date_paths("turbo_trade_events", dates)],
            "account_files": [str(p) for p in date_paths("account_snapshots", dates)],
            "pm2_files_scanned": pm2["files_scanned"],
        },
        "warnings": warnings,
        "open_questions": [
            "¿Conviene separar thresholds por simbolo y lado ya, o esperar mas cierres por simbolo?",
            "¿Se debe exigir 3/3 para SHORT globalmente o solo en simbolos con SHORT negativo?",
            "¿El objetivo principal es maximizar PnL o reducir drawdown/correlacion en esta fase live?",
            "¿Cual es el limite aceptable de margin_used_pct para operar 11 simbolos correlacionados?",
            "¿Se desea calibrar BE/trailing por volatilidad del simbolo o mantener config global por simplicidad?",
        ],
    }
    report["answers"] = build_answers(report)
    report["recommendations"] = recommendation_items(report)
    return report


def write_reports(report: dict[str, Any], timestamp: str) -> dict[str, str]:
    reports_dir = ROOT / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    json_path = reports_dir / f"aegis_turbo_live_deep_dive_{timestamp}.json"
    md_path = reports_dir / f"aegis_turbo_live_deep_dive_{timestamp}.md"
    summary_path = reports_dir / f"aegis_turbo_live_summary_for_chat_{timestamp}.md"
    safe_report = json_safe(report)
    json_path.write_text(json.dumps(safe_report, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    md_path.write_text(render_markdown(report, summary_only=False), encoding="utf-8")
    summary_path.write_text(render_markdown(report, summary_only=True), encoding="utf-8")
    return {"json": str(json_path), "markdown": str(md_path), "summary": str(summary_path)}


def main() -> int:
    parser = argparse.ArgumentParser(description="Analyze local Aegis Turbo live logs and generate reports.")
    parser.add_argument("--dates", nargs="*", help="Dates to analyze, YYYY-MM-DD. Defaults to 2026-05-08..2026-05-10 plus newer logs.")
    parser.add_argument("--timestamp", default=utc_now_tag(), help="Timestamp tag for output filenames.")
    args = parser.parse_args()
    available = available_jsonl_dates()
    dates = decide_dates(available, args.dates)
    report = build_report(dates, args.timestamp)
    paths = write_reports(report, args.timestamp)
    report["output_files"] = paths
    Path(paths["json"]).write_text(json.dumps(json_safe(report), ensure_ascii=False, indent=2, default=str), encoding="utf-8")

    symbols_with_data = sorted({t.get("symbol") for t in read_jsonl_files(date_paths("turbo_trades", dates)).rows if t.get("symbol")})
    print("Aegis Turbo live deep dive generado")
    print(f"Periodo: {min(dates)} -> {max(dates)}")
    print(f"Trades cerrados analizados: {report['global_metrics']['closed_trades']}")
    print(f"Trades abiertos en logs: {report['global_metrics']['open_trades']}")
    print(f"Simbolos con datos: {', '.join(symbols_with_data) if symbols_with_data else 'ninguno'}")
    print(f"Warnings: {len(report['warnings'])}")
    for warning in report["warnings"][:8]:
        print(f"- {warning}")
    if len(report["warnings"]) > 8:
        print(f"- ... {len(report['warnings']) - 8} warnings mas en el reporte")
    print(f"Resumen ChatGPT: {paths['summary']}")
    print(f"JSON completo: {paths['json']}")
    print(f"Markdown completo: {paths['markdown']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
