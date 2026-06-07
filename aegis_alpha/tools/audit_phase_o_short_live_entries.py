#!/usr/bin/env python3
"""Read-only audit for real Phase O SHORT live entries."""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import shutil
import subprocess
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from urllib.request import Request, urlopen

import numpy as np

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[2]))

try:
    from aegis_alpha.tools.audit_phase_o_short_no_trades import (
        ALL_SYMBOLS,
        ENTRY_SYMBOLS,
        MODEL_ROOT,
        REPO,
        TS_LOG_DIR,
        TS_REPO,
        audit_config,
        audit_manifests,
        audit_snapshots,
        json_safe,
        nested,
        parse_dt,
        post_predict,
        run_cmd,
        utc_stamp,
        write_csv,
    )
except Exception:  # pragma: no cover - fallback for standalone help/import failures
    REPO = Path(__file__).resolve().parents[2]
    TS_REPO = REPO / "binance-futures-bot-ts"
    TS_LOG_DIR = TS_REPO / "logs" / "aegis"
    MODEL_ROOT = REPO / "aegis_alpha" / "models" / "turbo"
    ENTRY_SYMBOLS = ["LTCUSDT", "AVAXUSDT", "ETHUSDT", "SUIUSDT", "ADAUSDT", "DOGEUSDT", "BTCUSDT", "BNBUSDT", "XRPUSDT", "SOLUSDT"]
    ALL_SYMBOLS = ENTRY_SYMBOLS + ["LINKUSDT"]

    def utc_stamp() -> str:
        return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    def parse_dt(value: str | None) -> datetime | None:
        if not value:
            return None
        try:
            dt = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
            return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
        except Exception:
            return None

    def nested(obj: dict[str, Any], *keys: str, default: Any = None) -> Any:
        cur: Any = obj
        for key in keys:
            if not isinstance(cur, dict) or key not in cur:
                return default
            cur = cur[key]
        return cur

    def json_safe(value: Any) -> Any:
        if isinstance(value, Path):
            return str(value)
        if isinstance(value, (np.integer, np.floating)):
            return value.item()
        if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
            return None
        if isinstance(value, dict):
            return {str(k): json_safe(v) for k, v in value.items()}
        if isinstance(value, (list, tuple, set)):
            return [json_safe(v) for v in value]
        return value

    def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str] | None = None) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        names = fields or (list(rows[0].keys()) if rows else ["empty"])
        with path.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=names, extrasaction="ignore")
            writer.writeheader()
            for row in rows:
                writer.writerow({k: json.dumps(v, sort_keys=True) if isinstance(v, (dict, list)) else v for k, v in row.items()})

    def run_cmd(cmd: list[str], timeout: int = 8) -> dict[str, Any]:
        if not cmd or shutil.which(cmd[0]) is None:
            return {"stdout": "", "stderr": "command_not_found", "returncode": None}
        p = subprocess.run(cmd, cwd=REPO, text=True, capture_output=True, timeout=timeout, check=False)
        return {"stdout": p.stdout, "stderr": p.stderr, "returncode": p.returncode}

    def audit_manifests(symbols: list[str]) -> list[dict[str, Any]]:
        return []

    def audit_snapshots(symbols: list[str]) -> list[dict[str, Any]]:
        return []

    def audit_config() -> dict[str, Any]:
        return {}

    def post_predict(symbol: str, timeout: float = 20.0) -> dict[str, Any]:
        started = time.perf_counter()
        req = Request("http://127.0.0.1:8001/ml-v2/predict", data=json.dumps({"symbol": symbol}).encode(), headers={"Content-Type": "application/json"}, method="POST")
        try:
            with urlopen(req, timeout=timeout) as resp:
                payload = json.loads(resp.read().decode("utf-8"))
            turbo = nested(payload, "aegis", "turbo", default={}) or {}
            return {"symbol": symbol, "http_status": resp.status, "latency_ms": round((time.perf_counter() - started) * 1000, 2), "action": turbo.get("action"), "reason": turbo.get("reason")}
        except Exception as exc:
            return {"symbol": symbol, "http_status": None, "latency_ms": round((time.perf_counter() - started) * 1000, 2), "error": repr(exc)}


PHASE_O_ENTRY_CLASSES = {
    "VALID_PHASE_O_SHORT_ENTRY",
    "QUESTIONABLE_PHASE_O_SHORT_ENTRY",
    "INVALID_PHASE_O_SHORT_ENTRY",
    "DUPLICATE_OR_MACHINE_GUN_ENTRY",
    "STALE_DATA_ENTRY",
    "MANIFEST_OR_MODEL_MISMATCH",
    "HARD_SAFETY_ISSUE",
    "UNKNOWN_NEEDS_MANUAL_REVIEW",
}

SYSTEM_CLASSES = {
    "PHASE_O_SHORT_LIVE_HEALTHY",
    "PHASE_O_SHORT_LIVE_WATCH_CLOSELY",
    "PHASE_O_SHORT_LIVE_PAUSE_RECOMMENDED",
    "PHASE_O_SHORT_LIVE_BROKEN",
}


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def date_strings(start: datetime, end: datetime) -> list[str]:
    out: list[str] = []
    cur = start.date()
    while cur <= end.date():
        out.append(cur.isoformat())
        cur += timedelta(days=1)
    return out


def iter_jsonl(path: Path, max_bytes: int | None = None):
    if not path.exists():
        return
    size = path.stat().st_size
    with path.open("rb") as fh:
        if max_bytes and size > max_bytes:
            fh.seek(max(0, size - max_bytes))
            fh.readline()
        for raw in fh:
            try:
                row = json.loads(raw)
            except Exception:
                continue
            yield row


def blob_text(value: Any) -> str:
    try:
        return json.dumps(value, sort_keys=True, default=str).lower()
    except Exception:
        return str(value).lower()


def to_float(value: Any) -> float | None:
    try:
        out = float(value)
    except Exception:
        return None
    if math.isnan(out) or math.isinf(out):
        return None
    return out


def extract_guards(row: dict[str, Any]) -> dict[str, Any]:
    meta = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
    return nested(meta, "entryPolicy", "guards", default={}) or nested(meta, "entry_policy", "guards", default={}) or {}


def guard_mode(row: dict[str, Any], name: str) -> str | None:
    guard = extract_guards(row).get(name)
    if isinstance(guard, dict):
        return str(guard.get("mode") or "").upper() or None
    return None


def guard_enforced(row: dict[str, Any], name: str) -> bool | None:
    guard = extract_guards(row).get(name)
    if isinstance(guard, dict):
        value = guard.get("enforced")
        return bool(value) if value is not None else None
    return None


def is_phase_o_short_trade(row: dict[str, Any]) -> bool:
    if str(row.get("side") or "").upper() != "SHORT":
        return False
    if str(row.get("strategy") or "").upper() != "AEGIS_TURBO":
        return False
    guards = extract_guards(row)
    if guards and guard_mode(row, "short_gate") == "SHADOW":
        return True
    text = blob_text(row)
    return "phase_o" in text or "experimental_short_only" in text


def signal_is_short(signal: dict[str, Any] | None) -> bool:
    if not signal:
        return False
    return str(signal.get("raw_action") or signal.get("gated_action") or signal.get("final_action") or "").upper() == "SHORT"


def signal_model_consistent(signal: dict[str, Any] | None) -> str:
    if not signal:
        return "MODEL_UNKNOWN"
    raw = str(signal.get("raw_action") or "").upper()
    final = str(signal.get("final_action") or signal.get("gated_action") or "").upper()
    if raw in {"HOLD", "WAIT"} and final in {"SHORT", "SELL"}:
        return "MODEL_HOLD_BUT_ORDERED"
    if raw in {"LONG", "BUY"} or final in {"LONG", "BUY"}:
        return "MODEL_SIDE_CONFLICT"
    score = to_float(signal.get("turbo_score"))
    if score is None:
        return "MODEL_NAN_OUTPUT"
    if not 0 <= score <= 1.5:
        return "MODEL_NAN_OUTPUT"
    if raw == "SHORT" or final == "SHORT":
        return "MODEL_DECISION_CONSISTENT"
    return "MODEL_OUTPUT_CONFLICT"


def infer_bucket(position_fraction: float | None) -> str | None:
    if position_fraction is None:
        return None
    if position_fraction <= 0.21:
        return "conservative"
    if position_fraction <= 0.36:
        return "normal"
    if position_fraction <= 0.51:
        return "premium"
    return "above_cap"


def validate_sizing(row: dict[str, Any]) -> str:
    fraction = to_float(row.get("position_fraction"))
    leverage = to_float(row.get("leverage"))
    if fraction is None or leverage is None:
        return "MODEL_UNKNOWN"
    if fraction > 0.50 or leverage > 30:
        return "MODEL_BUCKET_SIZING_MISMATCH"
    return "MODEL_DECISION_CONSISTENT"


def validate_guards(row: dict[str, Any]) -> str:
    if not is_phase_o_short_trade(row):
        return "GUARD_UNKNOWN"
    expected_shadow = ["short_gate", "clean_entry", "event_risk", "entry_quality", "decision_brain", "regime"]
    missing_or_enforced = []
    for name in expected_shadow:
        mode = guard_mode(row, name)
        enforced = guard_enforced(row, name)
        if mode and mode != "SHADOW":
            missing_or_enforced.append(name)
        if enforced is True and name != "momentum_ride":
            missing_or_enforced.append(name)
    if missing_or_enforced:
        return "GUARD_STILL_BLOCKING_BUG"
    return "GUARDS_EXPECTED_PHASE_O_SHADOW"


def validate_hard_safety(row: dict[str, Any], events: list[dict[str, Any]]) -> str:
    if str(row.get("symbol") or "").upper() == "LINKUSDT":
        return "LINK_ENTRY_BUG"
    if row.get("brackets_confirmed") is not True:
        return "BRACKETS_MISSING"
    text = blob_text(events)
    if "max_open_phase_o_positions" in text or "max_phase_o_trades_per_day" in text:
        return "HARD_SAFETY_ISSUE"
    return "HARD_SAFETY_OK"


def classify_signal_match(trade: dict[str, Any], signal: dict[str, Any] | None, order_time: datetime | None) -> str:
    if not signal:
        return "SIGNAL_MISSING"
    trade_symbol = str(trade.get("symbol") or "").upper()
    if str(signal.get("symbol") or "").upper() != trade_symbol:
        return "SIGNAL_SYMBOL_MISMATCH"
    if not signal_is_short(signal):
        return "SIGNAL_SIDE_MISMATCH"
    sig_ts = parse_dt(signal.get("timestamp"))
    if order_time and sig_ts:
        delta = (order_time - sig_ts).total_seconds()
        if delta < -2:
            return "SIGNAL_TOO_OLD"
        if delta > 300:
            return "SIGNAL_TOO_OLD"
    return "SIGNAL_MATCH_OK"


def classify_snapshot(signal: dict[str, Any] | None) -> str:
    fresh = signal.get("freshness") if isinstance(signal, dict) else None
    if not isinstance(fresh, dict):
        return "SNAPSHOT_UNKNOWN"
    if not fresh.get("exists"):
        return "SNAPSHOT_MISSING"
    if fresh.get("stale") is True or fresh.get("is_fresh") is False:
        return "SNAPSHOT_STALE"
    age = to_float(fresh.get("feature_age_seconds") or fresh.get("snapshot_age_seconds"))
    max_age = to_float(fresh.get("max_feature_age_seconds"))
    if age is not None and max_age is not None and age > max_age:
        return "SNAPSHOT_STALE"
    return "SNAPSHOT_OK_FRESH"


def classify_manifest_row(symbol: str, manifest_rows: dict[str, dict[str, Any]], signal: dict[str, Any] | None) -> str:
    row = manifest_rows.get(symbol) or {}
    status = row.get("status")
    if symbol == "LINKUSDT" and status == "LINK_AVOID_ONLY_OK":
        return "LINK_MODEL_OK_AVOID_ONLY"
    if status != "PHASE_O_MANIFEST_OK":
        return "MANIFEST_OR_MODEL_MISMATCH"
    signal_text = blob_text(signal or {})
    phase_paths = row.get("short_paths") or {}
    if isinstance(phase_paths, dict):
        manifest_paths = [str(v) for v in phase_paths.values() if "phase_o" in str(v)]
        if manifest_paths and signal_text and "phase_o" in signal_text:
            return "MODEL_MANIFEST_OK"
    return "MODEL_MANIFEST_OK"


def detect_machine_gun(trades: list[dict[str, Any]], window_seconds: int) -> dict[str, Any]:
    sorted_trades = sorted(
        [t for t in trades if parse_dt(t.get("opened_at") or t.get("timestamp"))],
        key=lambda t: parse_dt(t.get("opened_at") or t.get("timestamp")) or datetime.min.replace(tzinfo=timezone.utc),
    )
    duplicate_symbols: list[dict[str, Any]] = []
    multisymbol_windows: list[dict[str, Any]] = []
    repeated_score_windows: list[dict[str, Any]] = []
    for i, first in enumerate(sorted_trades):
        first_ts = parse_dt(first.get("opened_at") or first.get("timestamp"))
        if first_ts is None:
            continue
        window = []
        for row in sorted_trades[i:]:
            ts = parse_dt(row.get("opened_at") or row.get("timestamp"))
            if ts and 0 <= (ts - first_ts).total_seconds() <= window_seconds:
                window.append(row)
        symbols = [str(r.get("symbol")) for r in window]
        unique_symbols = sorted(set(symbols))
        if len(window) >= 4 and len(unique_symbols) >= 4:
            multisymbol_windows.append({
                "start": first_ts.isoformat(),
                "end": (parse_dt(window[-1].get("opened_at") or window[-1].get("timestamp")) or first_ts).isoformat(),
                "entry_count": len(window),
                "symbols": unique_symbols,
            })
        score_counts = Counter(round(float(r.get("turbo_score") or 0), 6) for r in window if to_float(r.get("turbo_score")) is not None)
        if score_counts and score_counts.most_common(1)[0][1] >= 4:
            repeated_score_windows.append({
                "start": first_ts.isoformat(),
                "score": score_counts.most_common(1)[0][0],
                "count": score_counts.most_common(1)[0][1],
            })
    by_symbol: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in sorted_trades:
        by_symbol[str(row.get("symbol"))].append(row)
    for symbol, rows in by_symbol.items():
        for a, b in zip(rows, rows[1:]):
            at = parse_dt(a.get("opened_at") or a.get("timestamp"))
            bt = parse_dt(b.get("opened_at") or b.get("timestamp"))
            if at and bt and (bt - at).total_seconds() <= window_seconds:
                duplicate_symbols.append({"symbol": symbol, "first": at.isoformat(), "second": bt.isoformat(), "seconds": (bt - at).total_seconds()})
    if duplicate_symbols:
        classification = "MACHINE_GUN_DUPLICATE_SYMBOL"
    elif multisymbol_windows:
        classification = "MACHINE_GUN_MULTISYMBOL_BURST"
    elif repeated_score_windows:
        classification = "MACHINE_GUN_REPEATED_SAME_SNAPSHOT"
    else:
        classification = "MACHINE_GUN_NONE"
    return {
        "classification": classification,
        "duplicate_symbols": duplicate_symbols,
        "multisymbol_windows": multisymbol_windows,
        "repeated_score_windows": repeated_score_windows,
    }


def entry_audit_score(checks: dict[str, Any]) -> int:
    score = 0
    if checks.get("manifest_status") in {"MODEL_MANIFEST_OK", "LINK_MODEL_OK_AVOID_ONLY"}:
        score += 20
    if checks.get("snapshot_status") == "SNAPSHOT_OK_FRESH":
        score += 15
    if checks.get("signal_status") == "SIGNAL_MATCH_OK":
        score += 15
    if checks.get("model_status") == "MODEL_DECISION_CONSISTENT":
        score += 15
    if checks.get("sizing_status") == "MODEL_DECISION_CONSISTENT":
        score += 10
    if checks.get("guard_status") == "GUARDS_EXPECTED_PHASE_O_SHADOW":
        score += 10
    if checks.get("hard_safety_status") == "HARD_SAFETY_OK":
        score += 10
    if checks.get("machine_gun_status") == "MACHINE_GUN_NONE":
        score += 5

    penalties = {
        "SNAPSHOT_STALE": 30,
        "SNAPSHOT_MISSING": 30,
        "MODEL_OUTPUT_CONFLICT": 30,
        "MODEL_HOLD_BUT_ORDERED": 30,
        "SIGNAL_MISSING": 30,
        "SIGNAL_TOO_OLD": 20,
        "MACHINE_GUN_DUPLICATE_SYMBOL": 25,
        "MACHINE_GUN_MULTISYMBOL_BURST": 15,
        "MACHINE_GUN_REPEATED_SAME_SNAPSHOT": 15,
        "HARD_SAFETY_ISSUE": 40,
        "LINK_ENTRY_BUG": 50,
        "MODEL_NAN_OUTPUT": 20,
        "MODEL_BUCKET_SIZING_MISMATCH": 20,
        "BRACKETS_MISSING": 15,
        "MANIFEST_OR_MODEL_MISMATCH": 30,
    }
    for value in checks.values():
        if isinstance(value, str):
            score -= penalties.get(value, 0)
    return max(0, min(100, int(score)))


def quality_label(score: int) -> str:
    if score >= 85:
        return "HIGH_QUALITY_ENTRY"
    if score >= 70:
        return "ACCEPTABLE_ENTRY"
    if score >= 50:
        return "QUESTIONABLE_ENTRY"
    return "BAD_ENTRY"


def classify_entry(checks: dict[str, Any], score: int) -> str:
    if checks.get("hard_safety_status") in {"LINK_ENTRY_BUG", "HARD_SAFETY_ISSUE", "BRACKETS_MISSING"}:
        return "HARD_SAFETY_ISSUE"
    if checks.get("manifest_status") == "MANIFEST_OR_MODEL_MISMATCH":
        return "MANIFEST_OR_MODEL_MISMATCH"
    if checks.get("snapshot_status") in {"SNAPSHOT_STALE", "SNAPSHOT_MISSING", "SNAPSHOT_LOAD_ERROR"}:
        return "STALE_DATA_ENTRY"
    if checks.get("model_status") in {"MODEL_HOLD_BUT_ORDERED", "MODEL_SIDE_CONFLICT", "MODEL_OUTPUT_CONFLICT", "MODEL_NAN_OUTPUT"}:
        return "INVALID_PHASE_O_SHORT_ENTRY"
    if checks.get("machine_gun_status") in {"MACHINE_GUN_DUPLICATE_SYMBOL", "MACHINE_GUN_MULTISYMBOL_BURST", "MACHINE_GUN_REPEATED_SAME_SNAPSHOT"}:
        return "DUPLICATE_OR_MACHINE_GUN_ENTRY"
    if checks.get("signal_status") != "SIGNAL_MATCH_OK" or score < 70:
        return "QUESTIONABLE_PHASE_O_SHORT_ENTRY"
    return "VALID_PHASE_O_SHORT_ENTRY"


def classify_system(entry_rows: list[dict[str, Any]], machine_gun: dict[str, Any]) -> str:
    if not entry_rows:
        return "PHASE_O_SHORT_LIVE_WATCH_CLOSELY"
    classes = Counter(str(r.get("classification")) for r in entry_rows)
    if classes.get("HARD_SAFETY_ISSUE") or classes.get("INVALID_PHASE_O_SHORT_ENTRY") or classes.get("MANIFEST_OR_MODEL_MISMATCH") or classes.get("STALE_DATA_ENTRY"):
        if classes.get("HARD_SAFETY_ISSUE") or classes.get("INVALID_PHASE_O_SHORT_ENTRY"):
            return "PHASE_O_SHORT_LIVE_BROKEN"
        return "PHASE_O_SHORT_LIVE_PAUSE_RECOMMENDED"
    if machine_gun.get("classification") != "MACHINE_GUN_NONE" or classes.get("DUPLICATE_OR_MACHINE_GUN_ENTRY") or classes.get("QUESTIONABLE_PHASE_O_SHORT_ENTRY"):
        return "PHASE_O_SHORT_LIVE_WATCH_CLOSELY"
    if all(int(r.get("entry_audit_score") or 0) >= 70 for r in entry_rows):
        return "PHASE_O_SHORT_LIVE_HEALTHY"
    return "PHASE_O_SHORT_LIVE_WATCH_CLOSELY"


def read_log_rows(kind: str, start: datetime, end: datetime, max_bytes_per_file: int) -> list[dict[str, Any]]:
    template = {
        "trades": "turbo_trades_{}.jsonl",
        "events": "turbo_trade_events_{}.jsonl",
        "signals": "turbo_signals_{}.jsonl",
    }[kind]
    rows: list[dict[str, Any]] = []
    for ds in date_strings(start, end):
        path = TS_LOG_DIR / template.format(ds)
        for row in iter_jsonl(path, max_bytes_per_file):
            ts = parse_dt(row.get("timestamp") or row.get("opened_at"))
            if ts and start <= ts <= end:
                row["_log_kind"] = kind
                row["_log_path"] = str(path)
                rows.append(row)
    return rows


def collect_phase_o_trades(trade_rows: list[dict[str, Any]], include_open: bool, include_closed: bool) -> list[dict[str, Any]]:
    latest_by_trade: dict[str, dict[str, Any]] = {}
    for row in trade_rows:
        trade_id = str(row.get("trade_id") or "")
        if not trade_id:
            continue
        if not is_phase_o_short_trade(row):
            continue
        status = str(row.get("status") or "").upper()
        if status == "OPEN" and not include_open:
            continue
        if status == "CLOSED" and not include_closed:
            continue
        prev = latest_by_trade.get(trade_id)
        if prev is None:
            latest_by_trade[trade_id] = row
            continue
        # Prefer rows with brackets/score for entry context, but keep close metrics if present.
        merged = {**prev, **{k: v for k, v in row.items() if v is not None}}
        if prev.get("brackets_confirmed") is True and row.get("brackets_confirmed") is not True:
            merged["brackets_confirmed"] = True
            merged["sl_price"] = prev.get("sl_price")
            merged["tp_price"] = prev.get("tp_price")
            merged["turbo_score"] = prev.get("turbo_score")
            merged["votes"] = prev.get("votes")
            merged["metadata"] = prev.get("metadata")
        latest_by_trade[trade_id] = merged
    return sorted(latest_by_trade.values(), key=lambda r: parse_dt(r.get("opened_at") or r.get("timestamp")) or datetime.min.replace(tzinfo=timezone.utc))


def events_by_trade(event_rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    out: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in event_rows:
        trade_id = str(row.get("trade_id") or "")
        if trade_id:
            out[trade_id].append(row)
    for rows in out.values():
        rows.sort(key=lambda r: parse_dt(r.get("timestamp")) or datetime.min.replace(tzinfo=timezone.utc))
    return out


def find_order_time(trade: dict[str, Any], trade_events: list[dict[str, Any]]) -> datetime | None:
    for row in trade_events:
        if str(row.get("event") or "").upper() == "ORDER_SUBMITTED":
            ts = parse_dt(row.get("timestamp"))
            if ts:
                return ts
    return parse_dt(trade.get("opened_at") or trade.get("timestamp"))


def find_matching_signal(trade: dict[str, Any], signal_rows: list[dict[str, Any]], order_time: datetime | None, fallback_seconds: int = 300) -> dict[str, Any] | None:
    trade_id = str(trade.get("trade_id") or "")
    symbol = str(trade.get("symbol") or "").upper()
    candidates = []
    for row in signal_rows:
        if str(row.get("symbol") or "").upper() != symbol:
            continue
        ts = parse_dt(row.get("timestamp"))
        if order_time and ts:
            delta = (order_time - ts).total_seconds()
            if -2 <= delta <= fallback_seconds:
                candidates.append((0 if row.get("trade_id") == trade_id else 1, abs(delta), row))
        elif row.get("trade_id") == trade_id:
            candidates.append((0, 0, row))
    if not candidates:
        return None
    candidates.sort(key=lambda x: (x[0], x[1]))
    return candidates[0][2]


def snapshot_check_row(trade: dict[str, Any], signal: dict[str, Any] | None, snapshot_status: str) -> dict[str, Any]:
    fresh = signal.get("freshness") if isinstance(signal, dict) else None
    fresh = fresh if isinstance(fresh, dict) else {}
    signal_ts = parse_dt(signal.get("timestamp")) if isinstance(signal, dict) else None
    order_ts = parse_dt(trade.get("opened_at") or trade.get("timestamp"))
    feature_ts = parse_dt(str(fresh.get("feature_timestamp"))) if fresh.get("feature_timestamp") else None
    return {
        "trade_id": trade.get("trade_id"),
        "symbol": trade.get("symbol"),
        "snapshot_status": snapshot_status,
        "snapshot_path": fresh.get("path"),
        "snapshot_mtime": fresh.get("snapshot_mtime"),
        "feature_timestamp": fresh.get("feature_timestamp"),
        "feature_age_seconds": fresh.get("feature_age_seconds"),
        "snapshot_age_seconds": fresh.get("snapshot_age_seconds"),
        "max_feature_age_seconds": fresh.get("max_feature_age_seconds"),
        "is_fresh": fresh.get("is_fresh"),
        "signal_after_feature_seconds": (signal_ts - feature_ts).total_seconds() if signal_ts and feature_ts else None,
        "order_after_feature_seconds": (order_ts - feature_ts).total_seconds() if order_ts and feature_ts else None,
        "lookback_days": fresh.get("lookback_days"),
        "last_ts": fresh.get("last_ts"),
    }


def model_check_row(trade: dict[str, Any], signal: dict[str, Any] | None, manifest_status: str, model_status: str, sizing_status: str) -> dict[str, Any]:
    return {
        "trade_id": trade.get("trade_id"),
        "symbol": trade.get("symbol"),
        "manifest_status": manifest_status,
        "model_status": model_status,
        "sizing_status": sizing_status,
        "raw_action": signal.get("raw_action") if signal else None,
        "gated_action": signal.get("gated_action") if signal else None,
        "final_action": signal.get("final_action") if signal else None,
        "reason": signal.get("reason") if signal else None,
        "turbo_score": signal.get("turbo_score") if signal else trade.get("turbo_score"),
        "votes": signal.get("votes") if signal else trade.get("votes"),
        "position_fraction": trade.get("position_fraction"),
        "leverage": trade.get("leverage"),
        "bucket": infer_bucket(to_float(trade.get("position_fraction"))),
    }


def guard_trace_row(trade: dict[str, Any], events: list[dict[str, Any]], guard_status: str, hard_safety_status: str) -> dict[str, Any]:
    guards = extract_guards(trade)
    return {
        "trade_id": trade.get("trade_id"),
        "symbol": trade.get("symbol"),
        "guard_status": guard_status,
        "hard_safety_status": hard_safety_status,
        "short_gate_legacy": nested(guards, "short_gate", "mode"),
        "clean_entry": nested(guards, "clean_entry", "mode"),
        "event_risk": nested(guards, "event_risk", "mode"),
        "entry_quality": nested(guards, "entry_quality", "mode"),
        "decision_brain": nested(guards, "decision_brain", "mode"),
        "regime_engine": nested(guards, "regime", "mode"),
        "brackets_confirmed": trade.get("brackets_confirmed"),
        "event_types": [e.get("event") for e in events],
    }


def trade_output_row(trade: dict[str, Any], checks: dict[str, Any], score: int, classification: str, quality: str) -> dict[str, Any]:
    return {
        "trade_id": trade.get("trade_id"),
        "symbol": trade.get("symbol"),
        "side": trade.get("side"),
        "opened_at": trade.get("opened_at") or trade.get("timestamp"),
        "closed_at": trade.get("closed_at"),
        "status": trade.get("status"),
        "source": "phase_o_short",
        "entry_price": trade.get("entry_price"),
        "current_price": trade.get("current_price"),
        "exit_price": trade.get("exit_price"),
        "qty": trade.get("quantity"),
        "leverage": trade.get("leverage"),
        "position_fraction": trade.get("position_fraction"),
        "bucket": infer_bucket(to_float(trade.get("position_fraction"))),
        "margin": trade.get("margin_estimated"),
        "notional": trade.get("notional_estimated"),
        "raw_score": trade.get("turbo_score"),
        "confidence": trade.get("confidence"),
        "votes": trade.get("votes"),
        "tp": trade.get("tp_price"),
        "sl": trade.get("sl_price"),
        "brackets_confirmed": trade.get("brackets_confirmed"),
        "pnl": trade.get("pnl") or trade.get("realized_pnl") or trade.get("unrealized_pnl"),
        "roe": trade.get("roe"),
        "entry_audit_score": score,
        "quality_label": quality,
        "classification": classification,
        **checks,
    }


def build_audit(
    symbols: list[str],
    start: datetime,
    end: datetime,
    include_open: bool,
    include_closed: bool,
    predict_smoke: bool,
    include_pm2: bool,
    include_manifests: bool,
    include_snapshots: bool,
    machine_gun_window_seconds: int,
    max_log_bytes_per_file: int,
) -> dict[str, Any]:
    trade_rows_all = read_log_rows("trades", start, end, max_log_bytes_per_file)
    event_rows_all = read_log_rows("events", start, end, max_log_bytes_per_file)
    signal_rows_all = read_log_rows("signals", start, end, max_log_bytes_per_file)
    symbol_set = set(symbols)
    trade_rows_all = [r for r in trade_rows_all if str(r.get("symbol") or "").upper() in symbol_set]
    event_rows_all = [r for r in event_rows_all if str(r.get("symbol") or "").upper() in symbol_set]
    signal_rows_all = [r for r in signal_rows_all if str(r.get("symbol") or "").upper() in symbol_set]

    trades = collect_phase_o_trades(trade_rows_all, include_open, include_closed)
    manifests = audit_manifests(symbols) if include_manifests else []
    config = audit_config()
    manifest_by_symbol = {r.get("symbol"): r for r in manifests}
    snapshots = audit_snapshots(symbols) if include_snapshots else []
    predicts = [post_predict(s) for s in symbols] if predict_smoke else []
    ev_by_trade = events_by_trade(event_rows_all)
    machine_gun = detect_machine_gun(trades, machine_gun_window_seconds)

    trade_out: list[dict[str, Any]] = []
    signal_out: list[dict[str, Any]] = []
    validation_out: list[dict[str, Any]] = []
    model_out: list[dict[str, Any]] = []
    snapshot_out: list[dict[str, Any]] = []
    guard_out: list[dict[str, Any]] = []

    for trade in trades:
        trade_id = str(trade.get("trade_id") or "")
        events = ev_by_trade.get(trade_id, [])
        close_events = [e for e in events if str(e.get("event") or "").upper() == "TRADE_CLOSED"]
        if close_events:
            trade = dict(trade)
            trade["status"] = "CLOSED"
            trade["closed_at"] = close_events[-1].get("timestamp")
        order_time = find_order_time(trade, events)
        signal = find_matching_signal(trade, signal_rows_all, order_time)
        signal_status = classify_signal_match(trade, signal, order_time)
        snapshot_status = classify_snapshot(signal)
        manifest_status = classify_manifest_row(str(trade.get("symbol") or ""), manifest_by_symbol, signal)
        model_status = signal_model_consistent(signal)
        sizing_status = validate_sizing(trade)
        guard_status = validate_guards(trade)
        hard_safety_status = validate_hard_safety(trade, events)
        per_entry_machine_gun = "MACHINE_GUN_NONE"
        if machine_gun.get("classification") in {"MACHINE_GUN_MULTISYMBOL_BURST", "MACHINE_GUN_REPEATED_SAME_SNAPSHOT"}:
            per_entry_machine_gun = str(machine_gun["classification"])
        for dup in machine_gun.get("duplicate_symbols", []):
            if dup.get("symbol") == trade.get("symbol"):
                per_entry_machine_gun = "MACHINE_GUN_DUPLICATE_SYMBOL"

        checks = {
            "signal_status": signal_status,
            "snapshot_status": snapshot_status,
            "manifest_status": manifest_status,
            "model_status": model_status,
            "sizing_status": sizing_status,
            "guard_status": guard_status,
            "hard_safety_status": hard_safety_status,
            "machine_gun_status": per_entry_machine_gun,
        }
        score = entry_audit_score(checks)
        classification = classify_entry(checks, score)
        quality = quality_label(score)

        trade_out.append(trade_output_row(trade, checks, score, classification, quality))
        signal_out.append({
            "trade_id": trade_id,
            "symbol": trade.get("symbol"),
            "signal_status": signal_status,
            "signal_timestamp": signal.get("timestamp") if signal else None,
            "order_timestamp": order_time.isoformat() if order_time else None,
            "seconds_before_order": (order_time - parse_dt(signal.get("timestamp"))).total_seconds() if signal and order_time and parse_dt(signal.get("timestamp")) else None,
            "signal_id": signal.get("signal_id") if signal else None,
            "raw_action": signal.get("raw_action") if signal else None,
            "gated_action": signal.get("gated_action") if signal else None,
            "final_action": signal.get("final_action") if signal else None,
            "reason": signal.get("reason") if signal else None,
            "turbo_score": signal.get("turbo_score") if signal else None,
            "votes": signal.get("votes") if signal else None,
            "gate_allowed": signal.get("gate_allowed") if signal else None,
            "gate_reason": signal.get("gate_reason") if signal else None,
            "metadata_path": "turbo_signals.jsonl",
            "freshness": signal.get("freshness") if signal else None,
        })
        model_out.append(model_check_row(trade, signal, manifest_status, model_status, sizing_status))
        snapshot_out.append(snapshot_check_row(trade, signal, snapshot_status))
        guard_out.append(guard_trace_row(trade, events, guard_status, hard_safety_status))
        validation_out.append({
            "trade_id": trade_id,
            "symbol": trade.get("symbol"),
            **checks,
            "entry_audit_score": score,
            "quality_label": quality,
            "classification": classification,
        })

    system_status = classify_system(trade_out, machine_gun)
    event_counts = Counter(str(r.get("event") or "TRADE_ROW") for r in event_rows_all + trade_rows_all)
    signal_counts = Counter(str(r.get("final_action") or r.get("raw_action") or "UNKNOWN").upper() for r in signal_rows_all)
    return {
        "schema_version": "phase_o_short_live_entries_audit_v1",
        "created_at": now_iso(),
        "mode": "READ_ONLY",
        "from": start.isoformat(),
        "to": end.isoformat(),
        "symbols": symbols,
        "system_status": system_status,
        "config": config,
        "summary": {
            "phase_o_short_trades_audited": len(trade_out),
            "classifications": dict(Counter(r["classification"] for r in trade_out)),
            "quality_labels": dict(Counter(r["quality_label"] for r in trade_out)),
            "machine_gun_classification": machine_gun.get("classification"),
            "stale_snapshot_entries": sum(1 for r in trade_out if r.get("snapshot_status") != "SNAPSHOT_OK_FRESH"),
            "model_conflict_entries": sum(1 for r in trade_out if r.get("model_status") != "MODEL_DECISION_CONSISTENT"),
            "hard_safety_issues": sum(1 for r in trade_out if r.get("hard_safety_status") != "HARD_SAFETY_OK"),
            "link_entry_attempts": sum(1 for r in trade_out if r.get("symbol") == "LINKUSDT"),
            "signals_by_action": dict(signal_counts),
            "events": dict(event_counts),
        },
        "trades": trade_out,
        "signals": signal_out,
        "validation": validation_out,
        "machine_gun": machine_gun,
        "model_checks": model_out,
        "snapshot_checks": snapshot_out,
        "guard_trace": guard_out,
        "manifests": manifests,
        "snapshot_inventory": snapshots,
        "predicts": predicts,
        "pm2_status": run_cmd(["pm2", "status"]).get("stdout") if include_pm2 else None,
        "confirmations": {
            "no_live_changes": True,
            "no_active_manifest": True,
            "no_yaml": True,
            "no_pm2_restart": True,
            "no_manual_orders": True,
            "no_env": True,
            "no_push": True,
            "no_commit": True,
        },
    }


def write_markdown(path: Path, payload: dict[str, Any]) -> None:
    summary = payload["summary"]
    lines = [
        "# Phase O SHORT Live Entries Audit",
        "",
        "## Safety",
        "- read-only",
        "- no live changes",
        "- no active_manifest",
        "- no YAML",
        "- no PM2 restart",
        "- no orders/manual position actions",
        "",
        "## Executive Summary",
        f"- Global status: `{payload['system_status']}`",
        f"- Phase O SHORT trades audited: `{summary['phase_o_short_trades_audited']}`",
        f"- Classifications: `{summary['classifications']}`",
        f"- Machine gun: `{summary['machine_gun_classification']}`",
        f"- Stale snapshot entries: `{summary['stale_snapshot_entries']}`",
        f"- Model conflict entries: `{summary['model_conflict_entries']}`",
        f"- Hard safety issues: `{summary['hard_safety_issues']}`",
        f"- LINK entry attempts: `{summary['link_entry_attempts']}`",
        f"- max_open_phase_o_positions: `{payload.get('config', {}).get('max_open_phase_o_positions')}`",
        f"- max_phase_o_trades_per_day: `{payload.get('config', {}).get('max_phase_o_trades_per_day')}`",
        "",
        "## Operations",
        "| symbol | opened_at | side | entry | qty | lev | fraction | bucket | TP | SL | brackets | score | audit_score | classification |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | --- | ---: | ---: | --- | ---: | ---: | --- |",
    ]
    for row in payload["trades"]:
        lines.append(
            f"| {row.get('symbol')} | {row.get('opened_at')} | {row.get('side')} | {row.get('entry_price')} | {row.get('qty')} | {row.get('leverage')} | {row.get('position_fraction')} | {row.get('bucket')} | {row.get('tp')} | {row.get('sl')} | {row.get('brackets_confirmed')} | {row.get('raw_score')} | {row.get('entry_audit_score')} | {row.get('classification')} |"
        )
    lines += [
        "",
        "## Signals",
        "| symbol | signal_time | action | score | reason | snapshot | signal_match | model |",
        "| --- | --- | --- | ---: | --- | --- | --- | --- |",
    ]
    signals_by_trade = {r["trade_id"]: r for r in payload["signals"]}
    for row in payload["trades"]:
        sig = signals_by_trade.get(row["trade_id"], {})
        lines.append(
            f"| {row.get('symbol')} | {sig.get('signal_timestamp')} | {sig.get('raw_action')}/{sig.get('final_action')} | {sig.get('turbo_score')} | {sig.get('reason')} | {row.get('snapshot_status')} | {row.get('signal_status')} | {row.get('model_status')} |"
        )
    lines += ["", "## Machine Gun Analysis"]
    mg = payload["machine_gun"]
    lines.append(f"- Classification: `{mg.get('classification')}`")
    lines.append(f"- Multi-symbol windows: `{mg.get('multisymbol_windows')}`")
    lines.append(f"- Duplicate symbols: `{mg.get('duplicate_symbols')}`")
    lines.append(f"- Repeated score windows: `{mg.get('repeated_score_windows')}`")
    lines += ["", "## Model / Snapshot / Guard Validation"]
    for row in payload["validation"]:
        lines.append(
            f"- {row.get('symbol')} {row.get('trade_id')}: signal={row.get('signal_status')} snapshot={row.get('snapshot_status')} manifest={row.get('manifest_status')} model={row.get('model_status')} sizing={row.get('sizing_status')} guards={row.get('guard_status')} hard={row.get('hard_safety_status')} score={row.get('entry_audit_score')}"
        )
    lines += ["", "## Recommendation"]
    if payload["system_status"] == "PHASE_O_SHORT_LIVE_HEALTHY":
        lines.append("- Keep running, continue monitoring first-cycle burst and realized trade behavior.")
    elif payload["system_status"] == "PHASE_O_SHORT_LIVE_WATCH_CLOSELY":
        lines.append("- Watch closely. Do not increase capital. The model/signals may be valid, but the burst or questionable entries need monitoring before scaling.")
    elif payload["system_status"] == "PHASE_O_SHORT_LIVE_PAUSE_RECOMMENDED":
        lines.append("- Consider pausing new entries after a controlled review; do not manually close existing positions from this audit.")
    else:
        lines.append("- Treat live entry path as broken until the listed critical issues are fixed.")
    lines += ["", "## Confirmations"]
    for key, value in payload["confirmations"].items():
        lines.append(f"- {key}: `{value}`")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_reports(payload: dict[str, Any], out_dir: Path) -> dict[str, str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = utc_stamp()
    base = out_dir / f"aegis_phase_o_short_live_entries_{ts}"
    paths = {
        "md": str(base.with_suffix(".md")),
        "json": str(base.with_suffix(".json")),
        "trades_csv": str(out_dir / f"aegis_phase_o_short_live_entries_trades_{ts}.csv"),
        "signals_csv": str(out_dir / f"aegis_phase_o_short_live_entries_signals_{ts}.csv"),
        "validation_csv": str(out_dir / f"aegis_phase_o_short_live_entries_validation_{ts}.csv"),
        "machine_gun_csv": str(out_dir / f"aegis_phase_o_short_live_entries_machine_gun_{ts}.csv"),
        "model_checks_csv": str(out_dir / f"aegis_phase_o_short_live_entries_model_checks_{ts}.csv"),
        "snapshot_checks_csv": str(out_dir / f"aegis_phase_o_short_live_entries_snapshot_checks_{ts}.csv"),
        "guard_trace_csv": str(out_dir / f"aegis_phase_o_short_live_entries_guard_trace_{ts}.csv"),
    }
    payload["reports"] = paths
    Path(paths["json"]).write_text(json.dumps(json_safe(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_markdown(Path(paths["md"]), payload)
    write_csv(Path(paths["trades_csv"]), payload["trades"])
    write_csv(Path(paths["signals_csv"]), payload["signals"])
    write_csv(Path(paths["validation_csv"]), payload["validation"])
    mg_rows = []
    for key in ["duplicate_symbols", "multisymbol_windows", "repeated_score_windows"]:
        for row in payload["machine_gun"].get(key, []):
            mg_rows.append({"type": key, **row})
    if not mg_rows:
        mg_rows = [{"type": payload["machine_gun"].get("classification")}]
    write_csv(Path(paths["machine_gun_csv"]), mg_rows)
    write_csv(Path(paths["model_checks_csv"]), payload["model_checks"])
    write_csv(Path(paths["snapshot_checks_csv"]), payload["snapshot_checks"])
    write_csv(Path(paths["guard_trace_csv"]), payload["guard_trace"])
    return paths


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--from", dest="from_ts", default="2026-06-07T00:00:00Z")
    p.add_argument("--to", default="now")
    p.add_argument("--out-dir", default="/home/jasan/Develop")
    p.add_argument("--symbols", default=",".join(ALL_SYMBOLS))
    p.add_argument("--include-open", action="store_true")
    p.add_argument("--include-closed", action="store_true")
    p.add_argument("--include-pm2", action="store_true")
    p.add_argument("--include-ts-events", action="store_true")
    p.add_argument("--include-api-logs", action="store_true")
    p.add_argument("--include-manifests", action="store_true")
    p.add_argument("--include-snapshots", action="store_true")
    p.add_argument("--predict-smoke", action="store_true")
    p.add_argument("--reconstruct-signals", action="store_true")
    p.add_argument("--validate-model-paths", action="store_true")
    p.add_argument("--machine-gun-window-seconds", type=int, default=300)
    p.add_argument("--max-log-bytes-per-file", type=int, default=250_000_000)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    start = parse_dt(args.from_ts) or datetime(2026, 6, 7, tzinfo=timezone.utc)
    end = parse_dt(args.to) if args.to != "now" else datetime.now(timezone.utc)
    if end is None:
        end = datetime.now(timezone.utc)
    symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    payload = build_audit(
        symbols=symbols,
        start=start,
        end=end,
        include_open=args.include_open,
        include_closed=args.include_closed,
        predict_smoke=args.predict_smoke,
        include_pm2=args.include_pm2,
        include_manifests=args.include_manifests or args.validate_model_paths,
        include_snapshots=args.include_snapshots,
        machine_gun_window_seconds=args.machine_gun_window_seconds,
        max_log_bytes_per_file=args.max_log_bytes_per_file,
    )
    paths = write_reports(payload, Path(args.out_dir))
    print(json.dumps({
        "system_status": payload["system_status"],
        "summary": payload["summary"],
        "reports": paths,
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
