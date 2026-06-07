#!/usr/bin/env python3
"""Read-only Phase O SHORT no-trade audit."""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
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

try:
    import yaml  # type: ignore
except Exception:  # pragma: no cover
    yaml = None

REPO = Path(__file__).resolve().parents[2]
TS_REPO = REPO / "binance-futures-bot-ts"
MODEL_ROOT = REPO / "aegis_alpha" / "models" / "turbo"
TS_LOG_DIR = TS_REPO / "logs" / "aegis"
ENTRY_SYMBOLS = ["LTCUSDT", "AVAXUSDT", "ETHUSDT", "SUIUSDT", "ADAUSDT", "DOGEUSDT", "BTCUSDT", "BNBUSDT", "XRPUSDT", "SOLUSDT"]
AVOID_SYMBOLS = ["LINKUSDT"]
ALL_SYMBOLS = ENTRY_SYMBOLS + AVOID_SYMBOLS
REFRESHER_BY_SYMBOL = {
    "ETHUSDT": "04-Aegis-Turbo-Refresh-A",
    "BTCUSDT": "04-Aegis-Turbo-Refresh-A",
    "SOLUSDT": "04-Aegis-Turbo-Refresh-A",
    "BNBUSDT": "05-Aegis-Turbo-Refresh-B",
    "XRPUSDT": "05-Aegis-Turbo-Refresh-B",
    "DOGEUSDT": "05-Aegis-Turbo-Refresh-B",
    "ADAUSDT": "05-Aegis-Turbo-Refresh-B",
    "AVAXUSDT": "06-Aegis-Turbo-Refresh-C",
    "LINKUSDT": "06-Aegis-Turbo-Refresh-C",
    "SUIUSDT": "06-Aegis-Turbo-Refresh-C",
    "LTCUSDT": "06-Aegis-Turbo-Refresh-C",
}
ROOT_CAUSES = [
    "ROOT_CAUSE_NO_SHORT_EDGE",
    "ROOT_CAUSE_PHASE_O_MANIFEST_DRIFT",
    "ROOT_CAUSE_SNAPSHOT_STALE_OR_MISSING",
    "ROOT_CAUSE_PREDICT_TIMEOUT_OR_ERROR",
    "ROOT_CAUSE_TS_METADATA_NOT_RECOGNIZED",
    "ROOT_CAUSE_GUARD_STILL_BLOCKING",
    "ROOT_CAUSE_HARD_SAFETY_BLOCKING",
    "ROOT_CAUSE_EXCHANGE_REJECTED",
    "ROOT_CAUSE_NO_SCAN_COVERAGE",
    "ROOT_CAUSE_UNKNOWN",
]


def utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def parse_dt(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        dt = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
    except Exception:
        return None


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
    fieldnames = fields or (list(rows[0].keys()) if rows else ["empty"])
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: json.dumps(v, sort_keys=True) if isinstance(v, (dict, list)) else v for k, v in row.items()})


def run_cmd(cmd: list[str], timeout: int = 8) -> dict[str, Any]:
    if not cmd or shutil.which(cmd[0]) is None:
        return {"available": False, "cmd": cmd, "stdout": "", "stderr": "command_not_found", "returncode": None}
    try:
        p = subprocess.run(cmd, cwd=REPO, text=True, capture_output=True, timeout=timeout, check=False)
        return {"available": True, "cmd": cmd, "stdout": p.stdout, "stderr": p.stderr, "returncode": p.returncode}
    except Exception as exc:
        return {"available": True, "cmd": cmd, "stdout": "", "stderr": repr(exc), "returncode": None}


def load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    text = path.read_text(encoding="utf-8", errors="ignore")
    if yaml is not None:
        try:
            return yaml.safe_load(text) or {}
        except Exception:
            return {"_parse_error": True, "_raw": text[:1000]}
    return {"_yaml_unavailable": True, "_raw": text[:1000]}


def nested(obj: dict[str, Any], *keys: str, default: Any = None) -> Any:
    cur: Any = obj
    for key in keys:
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def classify_manifest(symbol: str, manifest: dict[str, Any], exists: bool, short_paths: dict[str, str]) -> str:
    if symbol == "LINKUSDT":
        if not exists:
            return "PHASE_O_MANIFEST_DRIFTED"
        if manifest.get("phase_o_avoid_only") is True and manifest.get("phase_o_live_enabled") is False and manifest.get("phase_o_link_entry_enabled") is False:
            if any("/phase_o_" in str(v) for v in short_paths.values()):
                return "LINK_BAD_ENTRY_ENABLED"
            return "LINK_AVOID_ONLY_OK"
        return "LINK_BAD_ENTRY_ENABLED"
    if not exists:
        return "PHASE_O_MANIFEST_DRIFTED"
    if not manifest.get("phase_o_live_enabled"):
        return "PHASE_O_METADATA_MISSING"
    if not manifest.get("phase_o_overlay_persistence_enabled"):
        return "PHASE_O_METADATA_MISSING"
    phase_paths = [v for v in short_paths.values() if "/phase_o_" in str(v)]
    if not phase_paths:
        return "PHASE_O_MANIFEST_DRIFTED"
    missing = [v for v in phase_paths if not Path(v).exists()]
    if missing:
        return "PHASE_O_MODEL_FILE_MISSING"
    return "PHASE_O_MANIFEST_OK"


def audit_manifests(symbols: list[str]) -> list[dict[str, Any]]:
    rows = []
    for symbol in symbols:
        path = MODEL_ROOT / symbol / "active_manifest.json"
        exists = path.exists()
        manifest: dict[str, Any] = {}
        if exists:
            try:
                manifest = json.loads(path.read_text(encoding="utf-8"))
            except Exception as exc:
                manifest = {"_parse_error": repr(exc)}
        model_paths = manifest.get("model_paths") or {}
        short_paths = {k: str(v) for k, v in model_paths.items() if str(k).startswith("short_")}
        long_paths = {k: str(v) for k, v in model_paths.items() if str(k).startswith("long_")}
        phase_short = {k: v for k, v in short_paths.items() if "/phase_o_" in v or "phase_o_" in Path(v).name}
        short_missing = [v for v in short_paths.values() if v and not Path(v).exists()]
        phase_missing = [v for v in phase_short.values() if v and not Path(v).exists()]
        stat = path.stat() if exists else None
        rows.append({
            "symbol": symbol,
            "manifest_path": str(path),
            "active_manifest_exists": exists,
            "manifest_modified_at": datetime.fromtimestamp(stat.st_mtime, timezone.utc).isoformat() if stat else None,
            "schema_version": manifest.get("schema_version"),
            "created_at": manifest.get("created_at"),
            "phase_o_live_enabled": manifest.get("phase_o_live_enabled"),
            "phase_o_live_mode": manifest.get("phase_o_live_mode"),
            "phase_o_overlay_persistence_enabled": manifest.get("phase_o_overlay_persistence_enabled"),
            "phase_o_live_artifact_stamp": manifest.get("phase_o_live_artifact_stamp"),
            "phase_o_avoid_only": manifest.get("phase_o_avoid_only"),
            "phase_o_link_entry_enabled": manifest.get("phase_o_link_entry_enabled"),
            "research_only": manifest.get("research_only"),
            "not_live_promoted": manifest.get("not_live_promoted"),
            "short_path_count": len(short_paths),
            "phase_o_short_path_count": len(phase_short),
            "phase_o_short_keys": sorted(phase_short),
            "short_paths": short_paths,
            "long_path_count": len(long_paths),
            "short_missing_count": len(short_missing),
            "phase_o_short_missing_count": len(phase_missing),
            "joblib_files_exist": not short_missing,
            "feature_names_present": bool(manifest.get("feature_names")),
            "status": classify_manifest(symbol, manifest, exists, short_paths),
        })
    return rows


def audit_config() -> dict[str, Any]:
    py_cfg = load_yaml(REPO / "aegis_alpha" / "configs" / "turbo.yaml")
    ts_cfg = load_yaml(TS_REPO / "regime_config.live.yaml")
    phase = nested(ts_cfg, "aegis", "phase_o_short_live", default={}) or {}
    turbo = nested(ts_cfg, "aegis", "turbo", default={}) or {}
    guard_modes = phase.get("guard_modes") or {}
    hard = phase.get("hard_safety") or {}
    errors = []
    if phase.get("enabled") is not True:
        errors.append("CONFIG_PHASE_O_DISABLED")
    if phase.get("allow_orders") is not True:
        errors.append("CONFIG_ORDERS_DISABLED")
    if phase.get("require_brackets") is not True or hard.get("brackets") != "ENFORCE":
        errors.append("CONFIG_BRACKETS_MISSING")
    for guard in ["clean_entry", "event_risk", "entry_quality", "decision_brain", "regime_engine", "short_gate_legacy", "risk_shadow_guards"]:
        if guard_modes.get(guard) not in ("SHADOW", None):
            errors.append(f"CONFIG_GUARD_STILL_ENFORCING:{guard}")
    if phase.get("allow_link_entry") is not False or phase.get("link_avoid_only") is not True:
        errors.append("CONFIG_LINK_RISK")
    return {
        "python_turbo": {
            "enabled": py_cfg.get("enabled"),
            "sizing": py_cfg.get("sizing"),
            "thresholds": py_cfg.get("thresholds"),
            "risk": py_cfg.get("risk"),
        },
        "ts_phase_o_short_live": phase,
        "ts_turbo": turbo,
        "classification": "CONFIG_PHASE_O_OK" if not errors else errors,
        "phase_o_enabled": phase.get("enabled"),
        "allow_orders": phase.get("allow_orders"),
        "require_brackets": phase.get("require_brackets"),
        "max_open_phase_o_positions": phase.get("max_open_phase_o_positions"),
        "max_phase_o_trades_per_day": phase.get("max_phase_o_trades_per_day"),
        "position_fraction_cap": turbo.get("position_fraction_cap"),
        "allow_short": turbo.get("allow_short"),
        "live_enabled": turbo.get("live_enabled"),
        "guard_modes": guard_modes,
        "hard_safety": hard,
    }


def snapshot_path(symbol: str, lookback: int) -> Path:
    legacy = REPO / "aegis_alpha" / "data" / "processed" / f"turbo_recent_{lookback}d.npz"
    symbol_path = REPO / "aegis_alpha" / "data" / "processed" / "turbo" / symbol / f"turbo_recent_{lookback}d.npz"
    if symbol == "ETHUSDT" and legacy.exists():
        return legacy
    return symbol_path


def audit_snapshots(symbols: list[str]) -> list[dict[str, Any]]:
    rows = []
    max_age = int(os.getenv("TURBO_MAX_FEATURE_AGE_SECONDS", "900"))
    now = datetime.now(timezone.utc)
    for symbol in symbols:
        for lookback in (7, 14, 30):
            path = snapshot_path(symbol, lookback)
            exists = path.exists()
            row = {
                "symbol": symbol,
                "lookback_days": lookback,
                "snapshot_path": str(path),
                "snapshot_exists": exists,
                "refresher": REFRESHER_BY_SYMBOL.get(symbol),
                "max_feature_age_seconds": max_age,
            }
            if not exists:
                row.update({"status": "SNAPSHOT_MISSING"})
                rows.append(row)
                continue
            stat = path.stat()
            row["modified_at"] = datetime.fromtimestamp(stat.st_mtime, timezone.utc).isoformat()
            row["age_seconds"] = round((now - datetime.fromtimestamp(stat.st_mtime, timezone.utc)).total_seconds(), 3)
            try:
                with np.load(path, allow_pickle=True) as data:
                    keys = set(data.files)
                    row["loadable"] = True
                    row["contains_required_fields"] = all(k in keys for k in ["X", "live_X", "feature_names", "feature_timestamp"])
                    x = data.get("X")
                    live_x = data.get("live_X")
                    names = data.get("feature_names")
                    row["sample_count"] = int(len(x)) if x is not None else 0
                    row["feature_count"] = int(x.shape[1]) if x is not None and getattr(x, "ndim", 0) == 2 else 0
                    row["live_feature_count"] = int(live_x.shape[1]) if live_x is not None and getattr(live_x, "ndim", 0) == 2 else 0
                    row["feature_names_count"] = int(len(names)) if names is not None else 0
                    ft = data.get("feature_timestamp")
                    if ft is not None:
                        ftv = ft.item() if hasattr(ft, "item") else str(ft)
                        row["feature_timestamp"] = str(ftv)
                        ftdt = parse_dt(str(ftv))
                        if ftdt:
                            row["feature_age_seconds"] = round((now - ftdt).total_seconds(), 3)
                    age = row.get("feature_age_seconds", row["age_seconds"])
                    if not row["contains_required_fields"]:
                        row["status"] = "SNAPSHOT_FEATURE_MISMATCH"
                    elif float(age) > max_age:
                        row["status"] = "SNAPSHOT_STALE"
                    else:
                        row["status"] = "SNAPSHOT_OK"
            except Exception as exc:
                row.update({"loadable": False, "status": "SNAPSHOT_LOAD_ERROR", "error": repr(exc)})
            rows.append(row)
    return rows


def post_predict(symbol: str, timeout: float = 20.0) -> dict[str, Any]:
    started = time.perf_counter()
    req = Request("http://127.0.0.1:8001/ml-v2/predict", data=json.dumps({"symbol": symbol}).encode(), headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urlopen(req, timeout=timeout) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
        turbo = nested(payload, "aegis", "turbo", default={}) or {}
        raw = turbo.get("raw") or {}
        gated = turbo.get("gated") or {}
        phase = turbo.get("phase_o") or raw.get("phase_o") or {}
        scores = raw.get("recent_scores") or turbo.get("recent_scores") or {}
        short_scores = {k: v for k, v in scores.items() if str(k).startswith("short_")}
        long_scores = {k: v for k, v in scores.items() if str(k).startswith("long_")}
        action = turbo.get("action") or gated.get("action") or raw.get("action")
        reason = turbo.get("reason") or gated.get("reason") or raw.get("reason")
        status = classify_predict(symbol, resp.status, action, reason, phase, None)
        return {
            "symbol": symbol,
            "http_status": resp.status,
            "latency_ms": round((time.perf_counter() - started) * 1000, 2),
            "decision": action,
            "side": action,
            "action": action,
            "reason": reason,
            "finalReason": reason,
            "finalStrategy": "AEGIS_TURBO",
            "phase_o_metadata_present": bool(phase),
            "phase_o_live_enabled": phase.get("phase_o_live_enabled"),
            "phase_o_short_entry_enabled": phase.get("phase_o_live_enabled") is True and phase.get("phase_o_link_avoid_only") is not True,
            "phase_o_link_avoid_only": phase.get("phase_o_link_avoid_only"),
            "phase_o_link_entry_enabled": phase.get("phase_o_link_entry_enabled"),
            "model_path_used": phase.get("phase_o_source_model_paths"),
            "bucket": turbo.get("confidence") or raw.get("confidence"),
            "position_fraction": turbo.get("position_fraction") or raw.get("position_fraction"),
            "leverage": turbo.get("leverage_suggestion") or raw.get("leverage_suggestion"),
            "score": turbo.get("turbo_score") or raw.get("turbo_score"),
            "short_scores": short_scores,
            "long_scores": long_scores,
            "freshness": turbo.get("freshness") or raw.get("freshness"),
            "status": status,
            "error": None,
        }
    except Exception as exc:
        return {
            "symbol": symbol,
            "http_status": None,
            "latency_ms": round((time.perf_counter() - started) * 1000, 2),
            "decision": None,
            "reason": None,
            "phase_o_metadata_present": False,
            "status": classify_predict(symbol, None, None, None, {}, repr(exc)),
            "error": repr(exc),
        }


def classify_predict(symbol: str, http_status: int | None, action: Any, reason: Any, phase: dict[str, Any], error: str | None) -> str:
    if error:
        if "timed out" in error.lower() or "timeout" in error.lower():
            return "PREDICT_TIMEOUT"
        return "PREDICT_ERROR"
    if http_status != 200:
        return "PREDICT_ERROR"
    if symbol == "LINKUSDT" and phase.get("phase_o_link_avoid_only") is True and phase.get("phase_o_link_entry_enabled") is False:
        return "PREDICT_LINK_AVOID_ONLY_OK"
    if not phase or phase.get("phase_o_live_enabled") is not True:
        return "PREDICT_PHASE_O_METADATA_MISSING"
    if str(action).upper() == "SHORT":
        return "PREDICT_PHASE_O_OK"
    return "PREDICT_HOLD_NO_EDGE"


def date_strings(start: datetime, end: datetime) -> list[str]:
    out = []
    cur = start.date()
    while cur <= end.date():
        out.append(cur.isoformat())
        cur = cur + timedelta(days=1)
    return out


def iter_jsonl_tail(path: Path, max_bytes: int):
    size = path.stat().st_size
    truncated = size > max_bytes
    with path.open("rb") as fh:
        if truncated:
            fh.seek(max(0, size - max_bytes))
            fh.readline()
        for raw in fh:
            try:
                yield json.loads(raw)
            except Exception:
                continue




def blob_text(value: Any) -> str:
    try:
        return json.dumps(value, sort_keys=True, default=str).lower()
    except Exception:
        return str(value).lower()


def extract_side(row: dict[str, Any]) -> str | None:
    meta = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
    candidates = [
        row.get("side"),
        row.get("action"),
        nested(row, "signal", "side"),
        nested(row, "decision", "side"),
        meta.get("side"),
        meta.get("action"),
        meta.get("finalAction"),
        meta.get("signalSide"),
        meta.get("intendedSide"),
        nested(meta, "entryPolicy", "side"),
        nested(meta, "trace", "side"),
        nested(meta, "raw", "action"),
        nested(meta, "turbo", "action"),
        nested(meta, "aegis", "turbo", "action"),
    ]
    for value in candidates:
        side = str(value or "").upper()
        if side in {"SHORT", "SELL"}:
            return "SHORT"
        if side in {"LONG", "BUY"}:
            return "LONG"
        if side in {"HOLD", "WAIT"}:
            return "HOLD"
    reason = str(row.get("reason") or meta.get("gatedReason") or "").lower()
    if "short" in reason:
        return "SHORT"
    if "long" in reason:
        return "LONG"
    return None


def has_phase_o_marker(row: dict[str, Any]) -> bool:
    text = blob_text(row)
    return any(marker in text for marker in [
        "phase_o",
        "phase-o",
        "isphaseoshortlivesignal",
        "phase_o_short_guard_modes_applied",
        "experimental_short_only",
    ])


def extract_block_reason(row: dict[str, Any]) -> str:
    meta = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
    parts = [
        row.get("reason"),
        meta.get("reason"),
        meta.get("gatedReason"),
        meta.get("gatedBlockedBy"),
        meta.get("finalReason"),
        nested(meta, "entryPolicy", "reason"),
    ]
    return " | ".join(str(p) for p in parts if p)


def classify_guard_family(reason: str, meta: dict[str, Any]) -> str:
    text = (reason + " " + blob_text(meta)).lower()
    if any(x in text for x in ["max_phase_o", "max_open", "daily_loss", "max_consecutive", "bracket", "min_notional", "insufficient_balance", "exchange_order", "quantity_below_min"]):
        return "hard_safety"
    if "short_gate" in text or "short_score_below" in text:
        return "short_gate"
    if any(x in text for x in ["clean_entry", "event_risk", "entry_quality", "decision_brain", "regime_engine", "regime"]):
        return "secondary_guard"
    if "link" in text and "avoid" in text:
        return "link_avoid_only"
    if "ml_predict" in text or "timeout" in text or "unavailable" in text:
        return "predict_error"
    return "unknown"


def audit_logs(symbols: list[str], start: datetime, end: datetime, max_bytes_per_file: int) -> dict[str, Any]:
    symbol_set = set(symbols)
    counts: dict[str, Counter] = {sym: Counter() for sym in symbols}
    global_counts = Counter()
    reasons = Counter()
    guard_counts = Counter()
    block_rows: list[dict[str, Any]] = []
    event_rows: list[dict[str, Any]] = []
    top_signals: list[dict[str, Any]] = []
    files_scanned = []
    truncated_files = []
    dates = date_strings(start, end)
    patterns = [
        ("events", "turbo_trade_events_{}.jsonl"),
        ("signals", "turbo_signals_{}.jsonl"),
        ("trades", "turbo_trades_{}.jsonl"),
    ]
    for kind, template in patterns:
        for ds in dates:
            path = TS_LOG_DIR / template.format(ds)
            if not path.exists():
                continue
            stat = path.stat()
            files_scanned.append({"kind": kind, "path": str(path), "size_bytes": stat.st_size, "truncated": stat.st_size > max_bytes_per_file})
            if stat.st_size > max_bytes_per_file:
                truncated_files.append(str(path))
            for row in iter_jsonl_tail(path, max_bytes_per_file):
                ts = parse_dt(row.get("timestamp"))
                if ts and (ts < start or ts > end):
                    continue
                sym = str(row.get("symbol") or "").replace("/", "").upper()
                if sym not in symbol_set:
                    continue
                event = str(row.get("event") or row.get("type") or kind).upper()
                meta = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
                reason = extract_block_reason(row)
                side = extract_side(row)
                is_short = side == "SHORT"
                strict_phase_o = has_phase_o_marker(row)
                guard_family = classify_guard_family(reason, meta)

                counts[sym][event] += 1
                global_counts[event] += 1
                if side:
                    counts[sym][f"{side}_{event}"] += 1
                    global_counts[f"{side}_{event}"] += 1
                if is_short:
                    counts[sym]["SHORT_EVENTS"] += 1
                    global_counts["SHORT_EVENTS"] += 1
                if is_short and strict_phase_o:
                    counts[sym][f"STRICT_PHASE_O_SHORT_{event}"] += 1
                    global_counts[f"STRICT_PHASE_O_SHORT_{event}"] += 1
                if strict_phase_o:
                    counts[sym]["PHASE_O_MARKER_EVENTS"] += 1
                    global_counts["PHASE_O_MARKER_EVENTS"] += 1
                if "phase_o_short_guard_modes_applied" in blob_text(row):
                    counts[sym]["PHASE_O_GUARD_MODES_APPLIED"] += 1
                    global_counts["PHASE_O_GUARD_MODES_APPLIED"] += 1
                if is_short and event in {"GATE_DENIED", "DECISION_ENFORCEMENT_DENIED", "ENTRY_POLICY_DECISION"}:
                    counts[sym][f"SHORT_{guard_family.upper()}"] += 1
                    global_counts[f"SHORT_{guard_family.upper()}"] += 1
                    guard_counts[guard_family] += 1
                if reason:
                    reasons[reason] += 1

                is_block_event = event in {"GATE_DENIED", "DECISION_ENFORCEMENT_DENIED", "ORDER_FAILED", "ORDER_REJECTED"} or "blocked" in reason.lower() or "denied" in reason.lower()
                if is_block_event:
                    block_rows.append({
                        "timestamp": row.get("timestamp"),
                        "symbol": sym,
                        "event": event,
                        "side": side,
                        "strict_phase_o": strict_phase_o,
                        "guard_family": guard_family,
                        "reason": reason,
                        "gatedReason": meta.get("gatedReason"),
                        "gatedBlockedBy": meta.get("gatedBlockedBy"),
                        "turboScore": meta.get("turboScore"),
                        "votes": meta.get("votes"),
                    })
                if event in {"ORDER_SUBMITTED", "BRACKETS_CONFIRMED", "POSITION_CONFIRMED", "TRADE_OPENED", "TRADE_CLOSED", "GATE_DENIED", "ENTRY_POLICY_DECISION", "SIGNAL_RECEIVED"}:
                    if len(event_rows) < 5000:
                        event_rows.append({
                            "timestamp": row.get("timestamp"),
                            "symbol": sym,
                            "event": event,
                            "side": side,
                            "strict_phase_o": strict_phase_o,
                            "guard_family": guard_family,
                            "reason": reason,
                            "metadata": meta,
                        })
                score = meta.get("turboScore") or row.get("turbo_score") or nested(row, "raw", "turbo_score")
                try:
                    score_f = float(score)
                except Exception:
                    score_f = None
                if score_f is not None:
                    top_signals.append({"timestamp": row.get("timestamp"), "symbol": sym, "event": event, "side": side, "strict_phase_o": strict_phase_o, "score": score_f, "reason": reason, "metadata": meta})

    symbol_rows = []
    for sym in symbols:
        c = counts[sym]
        order_sub = c.get("SHORT_ORDER_SUBMITTED", 0)
        pos = c.get("SHORT_POSITION_CONFIRMED", 0) + c.get("SHORT_TRADE_OPENED", 0)
        denied = c.get("SHORT_GATE_DENIED", 0) + c.get("SHORT_DECISION_ENFORCEMENT_DENIED", 0)
        sym_reasons = Counter(r.get("reason") for r in block_rows if r.get("symbol") == sym and r.get("side") == "SHORT" and r.get("reason"))
        top_reason = sym_reasons.most_common(1)[0][0] if sym_reasons else None
        symbol_rows.append({
            "symbol": sym,
            "predict_calls": c.get("SIGNAL_RECEIVED", 0),
            "short_signal_events": c.get("SHORT_EVENTS", 0),
            "strict_phase_o_short_events": sum(v for k, v in c.items() if k.startswith("STRICT_PHASE_O_SHORT_")),
            "phase_o_guard_modes_applied": c.get("PHASE_O_GUARD_MODES_APPLIED", 0),
            "hold_count": c.get("SHORT_GATE_DENIED", 0),
            "enter_candidates": c.get("SHORT_GATE_ALLOWED", 0) + order_sub,
            "phase_o_short_denied_by_guard": denied,
            "short_gate_denied": c.get("SHORT_SHORT_GATE", 0),
            "secondary_guard_denied": c.get("SHORT_SECONDARY_GUARD", 0),
            "hard_safety_denied": c.get("SHORT_HARD_SAFETY", 0),
            "order_submitted": order_sub,
            "strict_phase_o_order_submitted": c.get("STRICT_PHASE_O_SHORT_ORDER_SUBMITTED", 0),
            "position_confirmed": pos,
            "strict_phase_o_position_confirmed": c.get("STRICT_PHASE_O_SHORT_POSITION_CONFIRMED", 0) + c.get("STRICT_PHASE_O_SHORT_TRADE_OPENED", 0),
            "order_failed": c.get("SHORT_ORDER_FAILED", 0) + c.get("SHORT_ORDER_REJECTED", 0),
            "top_block_reason": top_reason,
            "stage": classify_stage(c, top_reason),
        })
    top_signals = sorted(top_signals, key=lambda r: r.get("score") or 0, reverse=True)[:20]
    return {
        "files_scanned": files_scanned,
        "truncated_files": truncated_files,
        "global_counts": dict(global_counts),
        "guard_counts": dict(guard_counts),
        "reason_counts": dict(reasons.most_common(50)),
        "symbol_event_rows": symbol_rows,
        "block_rows": block_rows[-5000:],
        "event_rows": event_rows,
        "top_signals": top_signals,
        "funnel": build_funnel(global_counts, symbol_rows, block_rows),
    }


def classify_stage(c: Counter, top_reason: str | None) -> str:
    if c.get("SIGNAL_RECEIVED", 0) == 0:
        return "STAGE_NO_PREDICT_CALLS"
    if c.get("SHORT_ORDER_SUBMITTED", 0) > 0 and c.get("SHORT_POSITION_CONFIRMED", 0) + c.get("SHORT_TRADE_OPENED", 0) == 0:
        return "STAGE_ORDER_REJECTED"
    if top_reason and any(x in top_reason for x in ["max_phase_o", "max_open", "daily_loss", "bracket", "min_notional", "insufficient_balance"]):
        return "STAGE_HARD_SAFETY_BLOCKED"
    if c.get("SHORT_SHORT_GATE", 0) > 0 or c.get("SHORT_SECONDARY_GUARD", 0) > 0:
        return "STAGE_TS_RECOGNIZED_BUT_GUARD_BLOCKED"
    if c.get("SHORT_HARD_SAFETY", 0) > 0:
        return "STAGE_HARD_SAFETY_BLOCKED"
    if c.get("SHORT_GATE_DENIED", 0) > 0:
        return "STAGE_PREDICT_HOLD_ONLY"
    return "STAGE_WAITING_FOR_SIGNAL"


def build_funnel(global_counts: Counter, symbol_rows: list[dict[str, Any]], blocks: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "scans_or_signal_received": global_counts.get("SIGNAL_RECEIVED", 0),
        "short_events": global_counts.get("SHORT_EVENTS", 0),
        "phase_o_marker_events": global_counts.get("PHASE_O_MARKER_EVENTS", 0),
        "phase_o_guard_modes_applied": global_counts.get("PHASE_O_GUARD_MODES_APPLIED", 0),
        "gate_denied": global_counts.get("GATE_DENIED", 0),
        "short_gate_denied": global_counts.get("SHORT_SHORT_GATE", 0),
        "short_secondary_guard_denied": global_counts.get("SHORT_SECONDARY_GUARD", 0),
        "short_hard_safety_denied": global_counts.get("SHORT_HARD_SAFETY", 0),
        "entry_policy_decisions": global_counts.get("ENTRY_POLICY_DECISION", 0),
        "order_submitted": global_counts.get("ORDER_SUBMITTED", 0),
        "short_order_submitted_total": global_counts.get("SHORT_ORDER_SUBMITTED", 0),
        "strict_phase_o_short_order_submitted": global_counts.get("STRICT_PHASE_O_SHORT_ORDER_SUBMITTED", 0),
        "brackets_confirmed": global_counts.get("BRACKETS_CONFIRMED", 0),
        "position_confirmed": global_counts.get("POSITION_CONFIRMED", 0) + global_counts.get("TRADE_OPENED", 0),
        "short_position_confirmed_total": global_counts.get("SHORT_POSITION_CONFIRMED", 0) + global_counts.get("SHORT_TRADE_OPENED", 0),
        "strict_phase_o_short_position_confirmed": global_counts.get("STRICT_PHASE_O_SHORT_POSITION_CONFIRMED", 0) + global_counts.get("STRICT_PHASE_O_SHORT_TRADE_OPENED", 0),
        "link_entry_attempts": sum(1 for b in blocks if b.get("symbol") == "LINKUSDT" and b.get("event") in {"ORDER_SUBMITTED", "POSITION_CONFIRMED", "TRADE_OPENED"} and b.get("side") == "SHORT"),
    }


def diagnose(manifests: list[dict[str, Any]], snapshots: list[dict[str, Any]], predicts: list[dict[str, Any]], log_audit: dict[str, Any]) -> dict[str, Any]:
    evidence = []
    entry_predicts = [p for p in predicts if p["symbol"] in ENTRY_SYMBOLS]
    drift = [m for m in manifests if m["symbol"] in ENTRY_SYMBOLS and m["status"] != "PHASE_O_MANIFEST_OK"]
    if drift:
        evidence.append(f"{len(drift)} entry manifests are not fully OK")
        return {"root_cause": "ROOT_CAUSE_PHASE_O_MANIFEST_DRIFT", "confidence": "HIGH", "evidence": evidence, "recommended_next_action": "Run Phase O overlay/re-promote audit before tuning thresholds."}

    bad_snap = [s for s in snapshots if s["symbol"] in ENTRY_SYMBOLS and s.get("status") in {"SNAPSHOT_STALE", "SNAPSHOT_MISSING", "SNAPSHOT_LOAD_ERROR"}]
    if bad_snap and len(bad_snap) >= 3:
        evidence.append(f"{len(bad_snap)} entry snapshots stale/missing/load-error")
        return {"root_cause": "ROOT_CAUSE_SNAPSHOT_STALE_OR_MISSING", "confidence": "HIGH", "evidence": evidence, "recommended_next_action": "Fix refreshers/snapshot freshness before model threshold changes."}

    pred_errors = [p for p in entry_predicts if p.get("status") in {"PREDICT_TIMEOUT", "PREDICT_ERROR"}]
    if len(pred_errors) >= max(2, len(entry_predicts)//3):
        evidence.append(f"{len(pred_errors)} predict errors/timeouts in smoke")
        return {"root_cause": "ROOT_CAUSE_PREDICT_TIMEOUT_OR_ERROR", "confidence": "HIGH", "evidence": evidence, "recommended_next_action": "Stabilize API/refresh contention and retry smoke."}

    no_meta = [p for p in entry_predicts if p.get("status") == "PREDICT_PHASE_O_METADATA_MISSING"]
    if no_meta:
        evidence.append(f"{len(no_meta)} predicts missing Phase O metadata")
        return {"root_cause": "ROOT_CAUSE_TS_METADATA_NOT_RECOGNIZED", "confidence": "MEDIUM", "evidence": evidence, "recommended_next_action": "Fix API metadata/TS recognition path."}

    funnel = log_audit.get("funnel") or {}
    current_short_predicts = [p for p in entry_predicts if str(p.get("action")).upper() == "SHORT"]
    short_gate_blocks = [b for b in log_audit.get("block_rows", []) if b.get("side") == "SHORT" and b.get("guard_family") == "short_gate"]
    secondary_blocks = [b for b in log_audit.get("block_rows", []) if b.get("side") == "SHORT" and b.get("strict_phase_o") and b.get("guard_family") == "secondary_guard"]
    hard_blocks = [b for b in log_audit.get("block_rows", []) if b.get("side") == "SHORT" and b.get("guard_family") == "hard_safety"]

    aggregate_short_gate_blocks = int(funnel.get("short_gate_denied", 0) or 0)
    aggregate_secondary_blocks = int(funnel.get("short_secondary_guard_denied", 0) or 0)
    if current_short_predicts and funnel.get("phase_o_guard_modes_applied", 0) == 0 and (aggregate_short_gate_blocks > 0 or aggregate_secondary_blocks > 0):
        evidence.append(f"Current smoke returns SHORT for {len(current_short_predicts)} entry symbols, but parsed TS events show short_gate_denied={aggregate_short_gate_blocks}, short_secondary_guard_denied={aggregate_secondary_blocks}, and 0 phase_o_short_guard_modes_applied markers")
        evidence.append("This suggests TS is evaluating SHORT candidates without recognizing/applying Phase O SHORT guard-scope metadata, so legacy short/secondary guards remain enforced.")
        return {"root_cause": "ROOT_CAUSE_TS_METADATA_NOT_RECOGNIZED", "confidence": "HIGH", "evidence": evidence, "recommended_next_action": "Audit TradingService isPhaseOShortLiveSignal/metadata path and ensure Phase O SHORT candidates receive SHADOW guard modes before short_gate/secondary enforcement."}

    if secondary_blocks:
        evidence.append(f"{len(secondary_blocks)} strict Phase O SHORT secondary guard block rows found")
        return {"root_cause": "ROOT_CAUSE_GUARD_STILL_BLOCKING", "confidence": "HIGH", "evidence": evidence, "recommended_next_action": "Verify Phase O SHORT guard-mode scoping in TS; secondary guards should be SHADOW for Phase O."}

    if hard_blocks:
        evidence.append(f"{len(hard_blocks)} SHORT hard-safety block rows found")
        return {"root_cause": "ROOT_CAUSE_HARD_SAFETY_BLOCKING", "confidence": "MEDIUM", "evidence": evidence, "recommended_next_action": "Inspect hard-safety block rows and current account state."}

    if not current_short_predicts and funnel.get("strict_phase_o_short_order_submitted", 0) == 0:
        reasons = Counter(str(p.get("reason")) for p in entry_predicts).most_common(5)
        evidence.append(f"Current smoke has 0 SHORT decisions across {len(entry_predicts)} entry symbols; top reasons={reasons}")
        evidence.append(f"Strict Phase O SHORT orders={funnel.get('strict_phase_o_short_order_submitted')} confirmed={funnel.get('strict_phase_o_short_position_confirmed')}")
        return {"root_cause": "ROOT_CAUSE_NO_SHORT_EDGE", "confidence": "HIGH", "evidence": evidence, "recommended_next_action": "Review Phase O thresholds/model agreement and recent short scores; do not change hard safety first."}

    if current_short_predicts and funnel.get("short_events", 0) == 0:
        evidence.append(f"Current smoke returns SHORT for {len(current_short_predicts)} symbols, but parsed TS logs have no SHORT events")
        return {"root_cause": "ROOT_CAUSE_NO_SCAN_COVERAGE", "confidence": "MEDIUM", "evidence": evidence, "recommended_next_action": "Verify Trading Bot scan loop covers all Phase O entry symbols and consumes /ml-v2/predict responses."}

    evidence.append("No dominant cause matched")
    return {"root_cause": "ROOT_CAUSE_UNKNOWN", "confidence": "LOW", "evidence": evidence, "recommended_next_action": "Run deeper TS/API trace around next candidate."}


def write_markdown(path: Path, payload: dict[str, Any]) -> None:
    diag = payload["diagnosis"]
    lines = [
        "# Phase O SHORT No-Trade Audit",
        "",
        "## Safety",
        "- read-only",
        "- no live changes",
        "- no active_manifest writes",
        "- no YAML writes",
        "- no PM2 restart",
        "- no orders",
        "",
        "## Executive Summary",
        f"- Root cause: `{diag['root_cause']}`",
        f"- Confidence: `{diag['confidence']}`",
        f"- Phase O config enabled: `{payload['config'].get('phase_o_enabled')}` allow_orders=`{payload['config'].get('allow_orders')}` brackets=`{payload['config'].get('require_brackets')}`",
        f"- max_open_phase_o_positions: `{payload['config'].get('max_open_phase_o_positions')}`",
        f"- max_phase_o_trades_per_day: `{payload['config'].get('max_phase_o_trades_per_day')}`",
        "",
        "## Evidence",
    ]
    for item in diag.get("evidence", []):
        lines.append(f"- {item}")
    lines += ["", "## Symbol Table", "| Symbol | Manifest | Phase paths | Snapshot bad | Predict | Decision | Reason | Orders | Positions | Top block |", "| --- | --- | ---: | ---: | --- | --- | --- | ---: | ---: | --- |"]
    manifests = {r["symbol"]: r for r in payload["manifests"]}
    predicts = {r["symbol"]: r for r in payload["predicts"]}
    event_rows = {r["symbol"]: r for r in payload["log_audit"]["symbol_event_rows"]}
    snap_bad = defaultdict(int)
    for s in payload["snapshots"]:
        if s.get("status") != "SNAPSHOT_OK":
            snap_bad[s["symbol"]] += 1
    for sym in payload["symbols"]:
        m = manifests.get(sym, {})
        p = predicts.get(sym, {})
        e = event_rows.get(sym, {})
        lines.append(f"| {sym} | {m.get('status')} | {m.get('phase_o_short_path_count')} | {snap_bad[sym]} | {p.get('status')} | {p.get('action')} | {p.get('reason')} | {e.get('order_submitted',0)} | {e.get('position_confirmed',0)} | {e.get('top_block_reason')} |")
    lines += ["", "## Funnel"]
    for k, v in payload["log_audit"].get("funnel", {}).items():
        lines.append(f"- {k}: `{v}`")
    lines += ["", "## Top Block Reasons"]
    for reason, count in list((payload["log_audit"].get("reason_counts") or {}).items())[:20]:
        lines.append(f"- `{reason}`: {count}")
    lines += ["", "## Top Signals"]
    for row in payload["log_audit"].get("top_signals", [])[:20]:
        lines.append(f"- {row.get('timestamp')} {row.get('symbol')} score={row.get('score')} event={row.get('event')} reason={row.get('reason')}")
    lines += ["", "## LINK Safety"]
    link = predicts.get("LINKUSDT", {})
    lines.append(f"- LINK predict status `{link.get('status')}`, action `{link.get('action')}`, reason `{link.get('reason')}`, avoid_only `{link.get('phase_o_link_avoid_only')}`, entry_enabled `{link.get('phase_o_link_entry_enabled')}`.")
    lines += ["", "## Recommendation", f"- {diag.get('recommended_next_action')}"]
    lines += ["", "## Confirmations"]
    for key, val in payload["confirmations"].items():
        lines.append(f"- {key}: `{val}`")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_reports(payload: dict[str, Any], out_dir: Path) -> dict[str, str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = utc_stamp()
    base = out_dir / f"aegis_phase_o_short_no_trades_{ts}"
    paths = {
        "md": str(base.with_suffix(".md")),
        "json": str(base.with_suffix(".json")),
        "symbols_csv": str(out_dir / f"aegis_phase_o_short_no_trades_symbols_{ts}.csv"),
        "predicts_csv": str(out_dir / f"aegis_phase_o_short_no_trades_predicts_{ts}.csv"),
        "blocks_csv": str(out_dir / f"aegis_phase_o_short_no_trades_blocks_{ts}.csv"),
        "events_csv": str(out_dir / f"aegis_phase_o_short_no_trades_events_{ts}.csv"),
        "snapshots_csv": str(out_dir / f"aegis_phase_o_short_no_trades_snapshots_{ts}.csv"),
    }
    payload["reports"] = paths
    Path(paths["json"]).write_text(json.dumps(json_safe(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_markdown(Path(paths["md"]), payload)
    symbol_rows = []
    by_manifest = {r["symbol"]: r for r in payload["manifests"]}
    by_predict = {r["symbol"]: r for r in payload["predicts"]}
    by_events = {r["symbol"]: r for r in payload["log_audit"]["symbol_event_rows"]}
    for sym in payload["symbols"]:
        symbol_rows.append({"symbol": sym, **{f"manifest_{k}": v for k, v in by_manifest.get(sym, {}).items() if k in ["status", "phase_o_short_path_count", "phase_o_short_keys"]}, **{f"predict_{k}": v for k, v in by_predict.get(sym, {}).items() if k in ["status", "action", "reason", "score"]}, **by_events.get(sym, {})})
    write_csv(Path(paths["symbols_csv"]), symbol_rows)
    write_csv(Path(paths["predicts_csv"]), payload["predicts"])
    write_csv(Path(paths["blocks_csv"]), payload["log_audit"].get("block_rows", []))
    write_csv(Path(paths["events_csv"]), payload["log_audit"].get("event_rows", []))
    write_csv(Path(paths["snapshots_csv"]), payload["snapshots"])
    return paths


def audit(args: argparse.Namespace) -> dict[str, Any]:
    symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    start = parse_dt(args.from_ts) or datetime(2026, 6, 1, tzinfo=timezone.utc)
    end = parse_dt(args.to) if args.to != "now" else datetime.now(timezone.utc)
    if end is None:
        end = datetime.now(timezone.utc)
    manifests = audit_manifests(symbols) if args.include_manifests else []
    config = audit_config()
    snapshots = audit_snapshots(symbols) if args.include_snapshots else []
    predicts = [post_predict(s) for s in symbols] if args.predict_smoke else []
    log_audit = audit_logs(symbols, start, end, args.max_log_bytes_per_file) if (args.include_logs or args.include_ts_events) else {"symbol_event_rows": [], "block_rows": [], "event_rows": [], "funnel": {}, "reason_counts": {}, "top_signals": []}
    pm2 = run_cmd(["pm2", "status"], timeout=8) if args.include_pm2 else {}
    diagnosis = diagnose(manifests, snapshots, predicts, log_audit)
    return {
        "schema_version": "phase_o_short_no_trades_audit_v1",
        "created_at": now_iso(),
        "mode": "READ_ONLY",
        "from": start.isoformat(),
        "to": end.isoformat(),
        "symbols": symbols,
        "manifests": manifests,
        "config": config,
        "snapshots": snapshots,
        "predicts": predicts,
        "log_audit": log_audit,
        "pm2_status": pm2.get("stdout"),
        "diagnosis": diagnosis,
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


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--from", dest="from_ts", default="2026-06-01T00:00:00Z")
    p.add_argument("--to", default="now")
    p.add_argument("--out-dir", default="/home/jasan/Develop")
    p.add_argument("--symbols", default=",".join(ALL_SYMBOLS))
    p.add_argument("--predict-smoke", action="store_true")
    p.add_argument("--include-pm2", action="store_true")
    p.add_argument("--include-logs", action="store_true")
    p.add_argument("--include-manifests", action="store_true")
    p.add_argument("--include-snapshots", action="store_true")
    p.add_argument("--include-ts-events", action="store_true")
    p.add_argument("--max-log-bytes-per-file", type=int, default=200_000_000)
    args = p.parse_args()
    payload = audit(args)
    paths = write_reports(payload, Path(args.out_dir))
    print(json.dumps({"reports": paths, "diagnosis": payload["diagnosis"], "funnel": payload["log_audit"].get("funnel")}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
