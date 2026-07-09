#!/usr/bin/env python3
"""Forward-compatible research audit for new live events.

This intentionally does not reconstruct deleted historical logs. It only scans
currently available text/jsonl logs and emits empty-but-valid outputs when live
logs are unavailable or insufficient.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable


REPO = Path(__file__).resolve().parents[2]
DEFAULT_OUT_DIR = Path("/home/jasan/Develop")
LOG_EXTENSIONS = {".jsonl", ".log", ".txt"}
SKIP_DIRS = {".git", "__pycache__", "node_modules", ".venv", "venv", "binance-futures-bot-ts"}
EVENT_TYPES = (
    "SIGNAL",
    "ORDER_SUBMITTED",
    "POSITION_CONFIRMED",
    "TRADE_OPEN",
    "TRADE_CLOSED",
    "BRACKET_CONFIRMED",
    "SL_CONFIRMED",
    "TP_CONFIRMED",
    "TRAILING_EVENT",
    "BREAKEVEN_EVENT",
    "ORDER_ERROR",
    "IMMEDIATE_TRIGGER_RISK",
    "UNKNOWN",
)
TRADE_FIELDS = [
    "trade_id",
    "symbol",
    "side",
    "opened_at",
    "closed_at",
    "is_closed",
    "entry_price",
    "exit_price",
    "qty",
    "leverage",
    "bucket",
    "score",
    "realized_pnl",
    "estimated_net_pnl",
    "close_reason",
    "bracket_status",
    "source_files",
    "match_confidence",
]
EVENT_FIELDS = [
    "timestamp",
    "event_type",
    "symbol",
    "side",
    "trade_id",
    "price",
    "qty",
    "score",
    "bucket",
    "realized_pnl",
    "close_reason",
    "source_file",
    "line_no",
    "raw",
]


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def parse_time_expr(value: str | None, *, now: datetime | None = None) -> datetime | None:
    if not value:
        return None
    base = now or utc_now()
    value = value.strip()
    if value == "now":
        return base
    m = re.fullmatch(r"now-(\d+)([dhm])", value)
    if m:
        amount = int(m.group(1))
        unit = m.group(2)
        delta = {"d": timedelta(days=amount), "h": timedelta(hours=amount), "m": timedelta(minutes=amount)}[unit]
        return base - delta
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)
    except ValueError:
        return None


def parse_event_timestamp(payload: dict[str, Any], raw: str) -> str | None:
    for key in ("timestamp", "time", "ts", "created_at", "opened_at", "closed_at"):
        val = payload.get(key)
        if val:
            return str(val)
    m = re.search(r"20\d{2}-\d{2}-\d{2}[T ][0-9:.+-]+Z?", raw)
    return m.group(0) if m else None


def to_float(value: Any) -> float | None:
    try:
        if value in (None, ""):
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def normalize_symbol(value: Any) -> str:
    if not value:
        return ""
    return str(value).upper().replace("/", "").replace("-", "")


def discover_log_files(paths: Iterable[Path] | None = None) -> list[Path]:
    candidates = list(paths or [
        Path("/home/jasan/Develop"),
        REPO,
        REPO / "logs",
        REPO / "log",
        REPO / "aegis_alpha" / "logs",
        REPO / "data" / "logs",
    ])
    files: list[Path] = []
    for root in candidates:
        if not root.exists():
            continue
        if root.is_file() and root.suffix.lower() in LOG_EXTENSIONS:
            files.append(root)
            continue
        if not root.is_dir():
            continue
        for current, dirs, names in os.walk(root):
            dirs[:] = [d for d in dirs if d not in SKIP_DIRS]
            for name in names:
                f = Path(current) / name
                if f.suffix.lower() in LOG_EXTENSIONS:
                    files.append(f)
    return sorted(set(files))


def classify_event(payload: dict[str, Any], raw: str) -> str:
    text = " ".join(str(payload.get(k, "")) for k in ("event", "type", "message", "action", "status", "reason"))
    text = f"{text} {raw}".lower()
    if "immediate" in text and "trigger" in text:
        return "IMMEDIATE_TRIGGER_RISK"
    if "order" in text and ("error" in text or "reject" in text or "failed" in text):
        return "ORDER_ERROR"
    if "trailing" in text:
        return "TRAILING_EVENT"
    if "breakeven" in text or "break even" in text:
        return "BREAKEVEN_EVENT"
    if "bracket" in text and ("confirm" in text or "placed" in text):
        return "BRACKET_CONFIRMED"
    if re.search(r"\bsl\b|stop.loss|stop_loss", text) and ("confirm" in text or "placed" in text):
        return "SL_CONFIRMED"
    if re.search(r"\btp\b|take.profit|take_profit", text) and ("confirm" in text or "placed" in text):
        return "TP_CONFIRMED"
    if "position" in text and ("confirm" in text or "open" in text):
        return "POSITION_CONFIRMED"
    if "trade" in text and ("closed" in text or "close" in text or "exit" in text):
        return "TRADE_CLOSED"
    if "trade" in text and ("open" in text or "entry" in text):
        return "TRADE_OPEN"
    if "order" in text and ("submit" in text or "submitted" in text or "created" in text):
        return "ORDER_SUBMITTED"
    if "signal" in text:
        return "SIGNAL"
    return "UNKNOWN"


def parse_log_line(raw: str, source: Path, line_no: int) -> dict[str, Any] | None:
    line = raw.strip()
    if not line:
        return None
    payload: dict[str, Any]
    try:
        decoded = json.loads(line)
        payload = decoded if isinstance(decoded, dict) else {"message": decoded}
    except json.JSONDecodeError:
        payload = {"message": line}
    event_type = classify_event(payload, line)
    symbol = normalize_symbol(payload.get("symbol") or payload.get("pair"))
    if not symbol:
        m = re.search(r"\b([A-Z]{2,12}USDT)\b", line.upper())
        symbol = m.group(1) if m else ""
    side = str(payload.get("side") or payload.get("direction") or "").upper()
    trade_id = payload.get("trade_id") or payload.get("id") or payload.get("order_id") or ""
    return {
        "timestamp": parse_event_timestamp(payload, line) or "",
        "event_type": event_type,
        "symbol": symbol,
        "side": side,
        "trade_id": str(trade_id),
        "price": payload.get("price") or payload.get("entry_price") or payload.get("exit_price") or "",
        "qty": payload.get("qty") or payload.get("quantity") or "",
        "score": payload.get("score") or payload.get("confidence") or "",
        "bucket": payload.get("bucket") or "",
        "realized_pnl": payload.get("realized_pnl") or payload.get("pnl") or "",
        "close_reason": payload.get("close_reason") or payload.get("reason") or "",
        "source_file": str(source),
        "line_no": line_no,
        "raw": line[:500],
    }


def iter_log_events(files: Iterable[Path]) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    for f in files:
        try:
            with f.open("r", encoding="utf-8", errors="ignore") as handle:
                for line_no, line in enumerate(handle, start=1):
                    parsed = parse_log_line(line, f, line_no)
                    if parsed and parsed["event_type"] != "UNKNOWN":
                        events.append(parsed)
        except OSError:
            continue
    return events


def build_trade_rows(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    trades: dict[str, dict[str, Any]] = {}
    fallback = 0
    for ev in events:
        if ev["event_type"] not in {"TRADE_OPEN", "TRADE_CLOSED", "POSITION_CONFIRMED", "BRACKET_CONFIRMED"}:
            continue
        key = ev.get("trade_id") or f"{ev.get('symbol')}:{fallback}"
        if not ev.get("trade_id"):
            fallback += 1
        row = trades.setdefault(key, {k: "" for k in TRADE_FIELDS})
        row["trade_id"] = key
        row["symbol"] = row["symbol"] or ev.get("symbol", "")
        row["side"] = row["side"] or ev.get("side", "")
        row["source_files"] = ",".join(sorted(set(filter(None, (row.get("source_files", ""), ev.get("source_file", ""))))))
        if ev["event_type"] in {"TRADE_OPEN", "POSITION_CONFIRMED"}:
            row["opened_at"] = row["opened_at"] or ev.get("timestamp", "")
            row["entry_price"] = row["entry_price"] or ev.get("price", "")
            row["qty"] = row["qty"] or ev.get("qty", "")
            row["bucket"] = row["bucket"] or ev.get("bucket", "")
            row["score"] = row["score"] or ev.get("score", "")
        if ev["event_type"] == "TRADE_CLOSED":
            row["closed_at"] = ev.get("timestamp", "")
            row["exit_price"] = ev.get("price", "")
            row["realized_pnl"] = ev.get("realized_pnl", "")
            row["estimated_net_pnl"] = ev.get("realized_pnl", "")
            row["close_reason"] = ev.get("close_reason", "")
        if ev["event_type"] == "BRACKET_CONFIRMED":
            row["bracket_status"] = "CONFIRMED"
    for row in trades.values():
        row["is_closed"] = bool(row.get("closed_at"))
        row["match_confidence"] = "0.80" if row["is_closed"] and row.get("opened_at") else "0.45"
    return list(trades.values())


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fields})


def run_audit(args: argparse.Namespace) -> dict[str, Any]:
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    timestamp = utc_now().strftime("%Y%m%dT%H%M%SZ")
    log_roots = getattr(args, "log_roots", None)
    files = discover_log_files([Path(p) for p in log_roots]) if log_roots else discover_log_files()
    events = iter_log_events(files) if files else []
    trades = build_trade_rows(events)
    tail_losses = [r for r in trades if (to_float(r.get("realized_pnl")) or 0.0) < 0]
    status = "OK" if trades else ("INSUFFICIENT_LIVE_LOGS" if events else "NO_LIVE_LOGS_AVAILABLE")
    result = {
        "schema_version": "forward_live_events_a_v1",
        "status": status,
        "generated_at": timestamp,
        "log_files_scanned": [str(f) for f in files],
        "event_count": len(events),
        "trade_count": len(trades),
        "tail_loss_count": len(tail_losses),
        "note": "Forward audit is prepared for new logs; deleted historical logs are not reconstructed.",
    }
    json_path = out_dir / f"aegis_forward_live_events_a_{timestamp}.json"
    md_path = out_dir / f"aegis_forward_live_events_a_{timestamp}.md"
    trades_path = out_dir / f"aegis_forward_live_events_a_trades_{timestamp}.csv"
    events_path = out_dir / f"aegis_forward_live_events_a_events_{timestamp}.csv"
    tail_path = out_dir / f"aegis_forward_live_events_a_tail_losses_{timestamp}.csv"
    json_path.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    md_path.write_text(
        "\n".join([
            "# Aegis Forward Live Events Audit A",
            "",
            f"- status: {status}",
            f"- generated_at: {timestamp}",
            f"- log_files_scanned: {len(files)}",
            f"- events: {len(events)}",
            f"- trades: {len(trades)}",
            "",
            "This audit is forward-compatible and does not reconstruct deleted historical logs.",
        ]) + "\n",
        encoding="utf-8",
    )
    write_csv(trades_path, trades, TRADE_FIELDS)
    write_csv(events_path, events, EVENT_FIELDS)
    write_csv(tail_path, tail_losses, TRADE_FIELDS)
    result["outputs"] = {k: str(v) for k, v in {
        "json": json_path,
        "md": md_path,
        "trades_csv": trades_path,
        "events_csv": events_path,
        "tail_losses_csv": tail_path,
    }.items()}
    return result


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Forward-compatible live event audit, research-only.")
    p.add_argument("--from", dest="from_time", default="now-7d")
    p.add_argument("--to", dest="to_time", default="now")
    p.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    p.add_argument("--stream-logs", action="store_true")
    p.add_argument("--include-signals", action="store_true")
    p.add_argument("--include-trades", action="store_true")
    p.add_argument("--include-brackets", action="store_true")
    p.add_argument("--include-order-errors", action="store_true")
    p.add_argument("--include-tail-losses", action="store_true")
    return p


def main() -> int:
    args = build_parser().parse_args()
    result = run_audit(args)
    print(json.dumps({k: result[k] for k in ("status", "event_count", "trade_count", "outputs")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
