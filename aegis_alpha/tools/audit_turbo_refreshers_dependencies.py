#!/usr/bin/env python3
"""Read-only dependency audit for Turbo PM2 refreshers.

This tool inspects PM2 metadata, logs, code references, snapshot outputs, and
current /ml-v2/predict behavior. It does not restart, stop, delete, or signal
processes and does not write active manifests, YAML, models, or orders.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.error import URLError
from urllib.request import Request, urlopen


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT_DIR = Path("/home/jasan/Develop")
SERVICE_NAMES = (
    "04-Aegis-Turbo-Refresh-A",
    "05-Aegis-Turbo-Refresh-B",
    "06-Aegis-Turbo-Refresh-C",
    "07-Aegis-Turbo-Retrain",
)
REFRESH_SYMBOL_GROUPS = {
    "04-Aegis-Turbo-Refresh-A": ("ETHUSDT", "BTCUSDT", "SOLUSDT"),
    "05-Aegis-Turbo-Refresh-B": ("BNBUSDT", "XRPUSDT", "DOGEUSDT", "ADAUSDT"),
    "06-Aegis-Turbo-Refresh-C": ("AVAXUSDT", "LINKUSDT", "SUIUSDT", "LTCUSDT"),
}
SMOKE_SYMBOLS = ("AVAXUSDT", "BNBUSDT", "SOLUSDT", "LINKUSDT")
CODE_PATTERNS = (
    "refresh",
    "snapshot",
    "turbo_shadow",
    "turbo_signal",
    "active_manifest",
    "phase_o",
    "ml-v2",
    "predict",
    "OHLCV",
    "binance_candles",
    "DatabaseManager",
    "get_ohlcv",
    "write",
    "jsonl",
    "sqlite",
    "POSITION_CONFIRMED",
    "SIGNAL_RECEIVED",
)
OUTPUT_FILE_RE = re.compile(r"(?P<path>/[^\s\"']+(?:\.npz|\.json|\.jsonl|\.db|active_manifest\.json))")


def utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [json_safe(v) for v in value]
    return value


def run_cmd(cmd: list[str], timeout: int = 10) -> dict[str, Any]:
    if not cmd or shutil.which(cmd[0]) is None:
        return {"available": False, "cmd": cmd, "returncode": None, "stdout": "", "stderr": "command_not_found"}
    try:
        proc = subprocess.run(cmd, cwd=REPO_ROOT, text=True, capture_output=True, timeout=timeout, check=False)
        return {
            "available": True,
            "cmd": cmd,
            "returncode": proc.returncode,
            "stdout": proc.stdout,
            "stderr": proc.stderr,
        }
    except subprocess.TimeoutExpired as exc:
        return {
            "available": True,
            "cmd": cmd,
            "returncode": None,
            "stdout": exc.stdout or "",
            "stderr": f"timeout:{timeout}s",
        }
    except Exception as exc:  # pragma: no cover - defensive host probe
        return {"available": True, "cmd": cmd, "returncode": None, "stdout": "", "stderr": repr(exc)}


def read_text(path: Path, limit: int | None = None) -> str:
    if not path.exists():
        return ""
    text = path.read_text(encoding="utf-8", errors="ignore")
    return text[:limit] if limit else text


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | tuple[str, ...] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(fieldnames or (list(rows[0].keys()) if rows else ["empty"]))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(value, sort_keys=True) if isinstance(value, (dict, list)) else value for key, value in row.items()})


def load_pm2_jlist(out_dir: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    path = out_dir / "pm2_jlist.json"
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8")), {"source": str(path), "fresh": False}
        except Exception:
            pass
    result = run_cmd(["pm2", "jlist"], timeout=10)
    try:
        return json.loads(result.get("stdout") or "[]"), {"source": "pm2 jlist", "fresh": True, "result": result}
    except Exception:
        return [], {"source": "pm2 jlist", "fresh": True, "result": result, "parse_error": True}


def parse_pm2_services(pm2_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    services: list[dict[str, Any]] = []
    for name in SERVICE_NAMES:
        proc = next((row for row in pm2_rows if row.get("name") == name), None)
        env = (proc or {}).get("pm2_env") or {}
        monit = (proc or {}).get("monit") or {}
        args = env.get("args") or []
        args_text = " ".join(str(item) for item in args)
        script = env.get("pm_exec_path") or env.get("script") or ""
        role = "scheduled_model_retrain" if "run_turbo_scheduled_retrain" in args_text else "feature_snapshot_refresh"
        services.append(
            {
                "name": name,
                "pm2_status": env.get("status") or "unknown",
                "pid": proc.get("pid") if proc else None,
                "restart_count": env.get("restart_time"),
                "cpu": monit.get("cpu"),
                "memory_bytes": monit.get("memory"),
                "memory_mb": round(float(monit.get("memory") or 0) / 1024 / 1024, 2),
                "uptime_ms": env.get("pm_uptime"),
                "cwd": env.get("pm_cwd"),
                "exec": script,
                "args": args,
                "args_text": args_text,
                "role_inferred": role,
                "symbols": parse_symbols_from_args(args_text),
            }
        )
    return services


def parse_symbols_from_args(args_text: str) -> list[str]:
    match = re.search(r"--symbols\s+([A-Z0-9,/_-]+)", args_text)
    if not match:
        return []
    return [item.replace("/", "").upper() for item in match.group(1).split(",") if item.strip()]


def code_reference_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    roots = [REPO_ROOT / "aegis_alpha", REPO_ROOT / "binance-futures-bot-ts"]
    allowed_suffixes = {".py", ".ts", ".tsx", ".js", ".json", ".yaml", ".yml", ".md"}
    pattern_re = re.compile("|".join(re.escape(item) for item in CODE_PATTERNS), re.IGNORECASE)
    for root in roots:
        if not root.exists():
            continue
        for path in root.rglob("*"):
            if not path.is_file() or path.suffix not in allowed_suffixes:
                continue
            if any(part in {".git", "node_modules", "dist", "__pycache__", "logs", "models", "data"} for part in path.parts):
                continue
            try:
                for lineno, line in enumerate(path.read_text(encoding="utf-8", errors="ignore").splitlines(), start=1):
                    matches = sorted({m.group(0).lower() for m in pattern_re.finditer(line)})
                    if matches:
                        rows.append(
                            {
                                "file": str(path.relative_to(REPO_ROOT)),
                                "line": lineno,
                                "terms": ",".join(matches),
                                "text": line.strip()[:500],
                            }
                        )
            except Exception:
                continue
    return rows


def output_inventory() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    roots = [
        REPO_ROOT / "data" / "processed",
        REPO_ROOT / "aegis_alpha" / "logs",
        REPO_ROOT / "binance-futures-bot-ts" / "logs",
        REPO_ROOT / "data",
    ]
    name_re = re.compile(r"(turbo_recent_.*\.npz|turbo_snapshot_refresh_.*\.json|turbo_recent_dataset_.*\.json|turbo_shadow_.*\.jsonl|turbo_signals_.*\.jsonl|turbo_trade_events_.*\.jsonl|account_snapshots_.*\.jsonl|binance_candles\.db)$")
    for root in roots:
        if not root.exists():
            continue
        for path in root.rglob("*"):
            if not path.is_file():
                continue
            if not name_re.search(path.name):
                continue
            try:
                stat = path.stat()
                rows.append(
                    {
                        "path": str(path),
                        "name": path.name,
                        "size_bytes": stat.st_size,
                        "mtime": datetime.fromtimestamp(stat.st_mtime, timezone.utc).isoformat(),
                        "kind": output_kind(path),
                    }
                )
            except OSError:
                continue
    rows.sort(key=lambda row: str(row.get("mtime") or ""), reverse=True)
    return rows[:1000]


def output_kind(path: Path) -> str:
    name = path.name
    if name == "binance_candles.db":
        return "sqlite_ohlcv_database"
    if name.startswith("turbo_recent_") and name.endswith(".npz"):
        return "turbo_feature_snapshot_npz"
    if name.startswith("turbo_snapshot_refresh_"):
        return "refresher_report_json"
    if name.startswith("turbo_recent_dataset_"):
        return "dataset_report_json"
    if name.startswith("turbo_shadow_"):
        return "python_turbo_shadow_jsonl"
    if name.startswith("turbo_signals_"):
        return "ts_turbo_signals_jsonl"
    if name.startswith("turbo_trade_events_"):
        return "ts_turbo_trade_events_jsonl"
    if name.startswith("account_snapshots_"):
        return "ts_account_snapshots_jsonl"
    return "unknown"


def analyze_log_text(text: str) -> dict[str, Any]:
    lower = text.lower()
    lines = text.splitlines()
    output_mentions = sorted({m.group("path") for m in OUTPUT_FILE_RE.finditer(text)})
    iterations = len(re.findall(r"turbo_snapshot_refresh_|refresh|files_written|dataset_reports|symbols_requested", lower))
    sleep_mentions = len(re.findall(r"\bsleep\b|interval|already_up_to_date|duration_seconds", lower))
    cpu_hot_hint = iterations >= 20 and sleep_mentions == 0
    return {
        "error_count": len(re.findall(r"\bERROR\b|error", text, flags=re.IGNORECASE)),
        "exception_count": len(re.findall(r"Exception|exception", text)),
        "traceback_count": len(re.findall(r"Traceback", text)),
        "econnreset_count": lower.count("econnreset"),
        "timeout_count": lower.count("timeout"),
        "sqlite_locked_count": lower.count("database is locked") + lower.count("sqlite locked"),
        "retry_count": lower.count("retry"),
        "sleep_or_interval_count": sleep_mentions,
        "refresh_complete_count": len(re.findall(r'"success":\s*true|refresh complete|finished_at', lower)),
        "signal_received_count": lower.count("signal_received"),
        "would_execute_count": lower.count("would_execute"),
        "active_manifest_touched_count": lower.count("active_manifest"),
        "phase_o_touched_count": lower.count("phase_o"),
        "hot_loop_cpu_hint": cpu_hot_hint,
        "line_count": len(lines),
        "last_error": last_matching_line(lines, ("error", "exception", "traceback", "database is locked", "timeout")),
        "last_success": last_matching_line(lines, ('"success": true', "refresh complete", "finished_at")),
        "output_files_mentioned": output_mentions[:50],
    }


def last_matching_line(lines: list[str], needles: tuple[str, ...]) -> str | None:
    lowered = tuple(item.lower() for item in needles)
    for line in reversed(lines):
        if any(item in line.lower() for item in lowered):
            return line.strip()[:500]
    return None


def service_log_findings(out_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    file_map = {
        "04-Aegis-Turbo-Refresh-A": out_dir / "pm2_04_refresh_a.log",
        "05-Aegis-Turbo-Refresh-B": out_dir / "pm2_05_refresh_b.log",
        "06-Aegis-Turbo-Refresh-C": out_dir / "pm2_06_refresh_c.log",
        "07-Aegis-Turbo-Retrain": out_dir / "pm2_07_retrain.log",
        "02-Aegis-API": out_dir / "pm2_02_api.log",
        "01-Trading-Bot": out_dir / "pm2_01_bot.log",
    }
    for name, path in file_map.items():
        finding = analyze_log_text(read_text(path))
        finding["service"] = name
        finding["path"] = str(path)
        rows.append(finding)
    return rows


def consumer_rows(code_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    consumers: list[dict[str, Any]] = []
    consumer_rules = [
        ("turbo_feature_snapshot_npz", "aegis_alpha/turbo/turbo_signal.py", "runtime turbo scoring reads freshest snapshot and blocks stale/missing snapshots"),
        ("turbo_feature_snapshot_npz", "aegis_alpha/inference/server.py", "/ml-v2/predict returns turbo_snapshot_status and calls turbo shadow block"),
        ("turbo_feature_snapshot_npz", "aegis_alpha/entry_quality/feature_builder.py", "entry-quality runtime features use latest Turbo snapshot as input"),
        ("ml_v2_predict_payload", "binance-futures-bot-ts/src/infra/adapters/AegisMLAdapter.ts", "TS bot calls /ml-v2/predict"),
        ("ml_v2_predict_payload", "binance-futures-bot-ts/src/app/services/TradingService.ts", "TradingService consumes freshness/turbo metadata from predict response"),
        ("turbo_logs_jsonl", "binance-futures-bot-ts/tools/analyze_aegis_entry_quality.py", "offline local analysis consumes turbo_signals/trades/account snapshots"),
        ("retrain_active_manifest", "aegis_alpha/tools/run_turbo_scheduled_retrain.py", "retrain can promote model manifests and now should reapply Phase O overlay"),
    ]
    existing = {row["file"] for row in code_rows}
    for output, file, reason in consumer_rules:
        consumers.append(
            {
                "output": output,
                "consumer_file": file,
                "consumer_found": file in existing,
                "consumer_reason": reason,
            }
        )
    return consumers


def classify_phase_dependency(service: dict[str, Any], consumers: list[dict[str, Any]] | None = None) -> str:
    role = service.get("role_inferred")
    name = str(service.get("name") or "")
    if role == "scheduled_model_retrain":
        return "INDIRECT_REQUIRED"
    if name in REFRESH_SYMBOL_GROUPS:
        symbols = set(service.get("symbols") or REFRESH_SYMBOL_GROUPS[name])
        entry_symbols = {"LTCUSDT", "AVAXUSDT", "ETHUSDT", "SUIUSDT", "ADAUSDT", "DOGEUSDT", "BTCUSDT", "BNBUSDT", "XRPUSDT", "SOLUSDT"}
        if symbols & entry_symbols:
            return "INDIRECT_REQUIRED"
    return "UNKNOWN"


def classify_health(service: dict[str, Any], log_finding: dict[str, Any]) -> str:
    status = str(service.get("pm2_status") or "").lower()
    cpu = float(service.get("cpu") or 0)
    if log_finding.get("traceback_count") or log_finding.get("exception_count"):
        return "ERRORED"
    if log_finding.get("sqlite_locked_count"):
        return "SQLITE_CONTENTION_RISK"
    if cpu >= 85 or log_finding.get("hot_loop_cpu_hint"):
        return "HOT_LOOP_CPU"
    if status == "stopped":
        return "STOPPED_BY_DESIGN"
    if log_finding.get("error_count", 0) > 20:
        return "LOG_SPAM"
    if status == "online":
        return "OK"
    return "UNKNOWN"


def classify_pause_safety(service: dict[str, Any], phase_dependency: str, health: str) -> str:
    if service.get("role_inferred") == "scheduled_model_retrain" and str(service.get("pm2_status")).lower() == "stopped":
        return "SAFE_TO_PAUSE"
    if phase_dependency == "DIRECT_REQUIRED":
        return "DO_NOT_PAUSE"
    if phase_dependency == "INDIRECT_REQUIRED":
        return "RISKY_TO_PAUSE"
    if health in {"ERRORED", "HOT_LOOP_CPU", "SQLITE_CONTENTION_RISK"}:
        return "UNKNOWN"
    return "UNKNOWN"


def classify_recommendation(service: dict[str, Any], phase_dependency: str, safety: str, health: str) -> str:
    if health in {"ERRORED", "HOT_LOOP_CPU", "SQLITE_CONTENTION_RISK"}:
        return "FIX_REQUIRED"
    if phase_dependency == "DIRECT_REQUIRED":
        return "KEEP_LIVE_REQUIRED"
    if phase_dependency == "INDIRECT_REQUIRED":
        return "KEEP_BUT_REDUCE_FREQUENCY"
    if safety == "SAFE_TO_PAUSE":
        return "PAUSE_SAFE"
    return "UNKNOWN_NEEDS_MANUAL_REVIEW"


def enrich_services(services: list[dict[str, Any]], log_rows: list[dict[str, Any]], consumers: list[dict[str, Any]]) -> list[dict[str, Any]]:
    log_by_service = {row["service"]: row for row in log_rows}
    enriched: list[dict[str, Any]] = []
    for service in services:
        log = log_by_service.get(service["name"], {})
        phase_dep = classify_phase_dependency(service, consumers)
        health = classify_health(service, log)
        pause_safety = classify_pause_safety(service, phase_dep, health)
        recommendation = classify_recommendation(service, phase_dep, pause_safety, health)
        writes, reads = infer_service_io(service)
        enriched.append(
            {
                **service,
                "writes": writes,
                "reads": reads,
                "consumers": [row for row in consumers if row.get("consumer_found")],
                "phase_o_dependency": phase_dep,
                "live_safety_if_paused": pause_safety,
                "health": health,
                "recommendation": recommendation,
                "log_summary": log,
            }
        )
    return enriched


def infer_service_io(service: dict[str, Any]) -> tuple[list[str], list[str]]:
    role = service.get("role_inferred")
    if role == "scheduled_model_retrain":
        return (
            [
                "aegis_alpha/models/turbo/<SYMBOL>/active_manifest.json when promotion is enabled",
                "aegis_alpha/models/turbo/<SYMBOL> model artifacts",
                "aegis_alpha/logs/turbo_retrain/*",
            ],
            [
                "data/binance_candles.db",
                "data/processed/turbo/<SYMBOL>/turbo_recent_<lookback>d.npz",
                "phase_o overlay manifests/pointers",
            ],
        )
    symbols = service.get("symbols") or []
    return (
        [
            f"data/processed/turbo/{symbol}/turbo_recent_<lookback>d.npz" for symbol in symbols
        ]
        + [
            "aegis_alpha/logs/turbo_snapshot_refresh_<timestamp>.json",
            "data/binance_candles.db via incremental OHLCV inserts",
        ],
        [
            "Binance public futures klines",
            "data/binance_candles.db current OHLCV",
            "aegis_alpha/configs/turbo.yaml runtime defaults",
        ],
    )


def db_inventory() -> dict[str, Any]:
    db_path = REPO_ROOT / "data" / "binance_candles.db"
    info: dict[str, Any] = {"path": str(db_path), "exists": db_path.exists()}
    if db_path.exists():
        stat = db_path.stat()
        info.update({"size_bytes": stat.st_size, "mtime": datetime.fromtimestamp(stat.st_mtime, timezone.utc).isoformat()})
    result = run_cmd(
        [
            sys.executable,
            "-c",
            (
                "import sqlite3,json; "
                f"con=sqlite3.connect({str(db_path)!r}); "
                "cur=con.cursor(); "
                "tabs=[r[0] for r in cur.execute(\"select name from sqlite_master where type='table'\")]; "
                "out={'tables':tabs}; "
                "rows=[]; "
                "\nfor t in tabs:\n"
                "    try:\n"
                "        cols=[r[1] for r in cur.execute(f'pragma table_info({t})')]\n"
                "        cnt=cur.execute(f'select count(*) from {t}').fetchone()[0]\n"
                "        rows.append({'table':t,'count':cnt,'columns':cols[:12]})\n"
                "    except Exception as e: rows.append({'table':t,'error':repr(e)})\n"
                "out['table_rows']=rows; print(json.dumps(out))"
            ),
        ],
        timeout=10,
    )
    try:
        info["sqlite"] = json.loads(result.get("stdout") or "{}")
    except Exception:
        info["sqlite_probe_error"] = result.get("stderr")
    return info


def smoke_predict(symbols: tuple[str, ...] = SMOKE_SYMBOLS, base_url: str = "http://127.0.0.1:8001") -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for symbol in symbols:
        started = time.perf_counter()
        payload = json.dumps({"symbol": symbol}).encode("utf-8")
        req = Request(base_url.rstrip("/") + "/ml-v2/predict", data=payload, headers={"Content-Type": "application/json"}, method="POST")
        try:
            with urlopen(req, timeout=8) as response:
                body = response.read().decode("utf-8", errors="replace")
                data = json.loads(body)
                aegis = data.get("aegis") or {}
                turbo = aegis.get("turbo") or {}
                phase = turbo.get("phase_o") if isinstance(turbo.get("phase_o"), dict) else turbo
                rows.append(
                    {
                        "symbol": symbol,
                        "http_status": response.status,
                        "ok": response.status == 200,
                        "latency_ms": round((time.perf_counter() - started) * 1000, 2),
                        "action": turbo.get("action") or (turbo.get("raw") or {}).get("action"),
                        "reason": turbo.get("reason") or (turbo.get("raw") or {}).get("reason"),
                        "freshness_is_fresh": ((turbo.get("freshness") or {}).get("is_fresh")),
                        "phase_o_live_enabled": phase.get("phase_o_live_enabled") if isinstance(phase, dict) else None,
                        "phase_o_link_avoid_only": phase.get("phase_o_link_avoid_only") if isinstance(phase, dict) else None,
                        "phase_o_link_entry_enabled": phase.get("phase_o_link_entry_enabled") if isinstance(phase, dict) else None,
                    }
                )
        except (URLError, TimeoutError, json.JSONDecodeError, Exception) as exc:  # pragma: no cover - host dependent
            rows.append(
                {
                    "symbol": symbol,
                    "http_status": None,
                    "ok": False,
                    "latency_ms": round((time.perf_counter() - started) * 1000, 2),
                    "error": repr(exc),
                }
            )
    return rows


def write_markdown(path: Path, payload: dict[str, Any]) -> None:
    services = payload["services"]
    lines = [
        "# OPS-REFRESH-A Turbo Refreshers Dependency Audit",
        "",
        "## Safety",
        "- read-only audit",
        "- no PM2 restart/stop/delete",
        "- no process kill",
        "- no YAML, active_manifest, .env, model, or live code changes",
        "- no orders",
        "",
        "## Executive Summary",
    ]
    for service in services:
        lines.append(
            "- {name}: status `{pm2_status}`, role `{role_inferred}`, health `{health}`, Phase O dependency `{phase_o_dependency}`, recommendation `{recommendation}`.".format(
                **service
            )
        )
    lines.extend(
        [
            "",
            "## Findings",
            "- Refreshers 04/05/06 run `refresh_turbo_snapshots.py --mode features-only` for different symbol groups.",
            "- The refresher script updates `data/binance_candles.db` from Binance public klines and writes `data/processed/turbo/<SYMBOL>/turbo_recent_<lookback>d.npz` snapshots.",
            "- `/ml-v2/predict` calls Turbo runtime code that selects the freshest snapshot and blocks stale/missing snapshots.",
            "- Phase O SHORT model manifests and overlay are not produced by refreshers 04/05/06, but Phase O runtime scoring still depends indirectly on fresh Turbo snapshots.",
            "- 07 retrain can promote active manifests when enabled; it is stopped in the captured state and is not needed for immediate Phase O predict, but it matters for future retrain/promotion workflows.",
            "",
            "## Services",
            "| Service | PM2 | Role | Symbols | Health | Phase O dependency | Pause safety | Recommendation |",
            "| --- | --- | --- | --- | --- | --- | --- | --- |",
        ]
    )
    for service in services:
        lines.append(
            "| {name} | {pm2_status} | {role_inferred} | {symbols} | {health} | {phase_o_dependency} | {live_safety_if_paused} | {recommendation} |".format(
                **{**service, "symbols": ",".join(service.get("symbols") or [])}
            )
        )
    lines.extend(["", "## Smoke Predict"])
    for row in payload.get("smoke_predict", []):
        lines.append(
            f"- {row.get('symbol')}: HTTP {row.get('http_status')} ok={row.get('ok')} action={row.get('action')} reason={row.get('reason')} phase_o={row.get('phase_o_live_enabled')} link_avoid={row.get('phase_o_link_avoid_only')}"
        )
    lines.extend(
        [
            "",
            "## OPS-REFRESH-B Proposal",
            "- Do not delete any refresher yet.",
            "- Use a controlled pause test only after confirming no open-position dependency and with before/after `/ml-v2/predict` smoke.",
            "- Candidate order: validate already-stopped service behavior first, then test the hottest refresher with a short rollback window.",
            "- Rollback would be `pm2 restart <service> --update-env`; this audit did not execute it.",
            "",
            "## Reports",
        ]
    )
    for key, value in sorted(payload.get("paths", {}).items()):
        lines.append(f"- {key}: `{value}`")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def audit(out_dir: Path, smoke: bool = True) -> dict[str, Any]:
    pm2_rows, pm2_source = load_pm2_jlist(out_dir)
    services = parse_pm2_services(pm2_rows)
    code_rows = code_reference_rows()
    (out_dir / "code_refs_refreshers.txt").write_text(
        "\n".join(f"{row['file']}:{row['line']}: {row['text']}" for row in code_rows) + "\n",
        encoding="utf-8",
    )
    outputs = output_inventory()
    log_rows = service_log_findings(out_dir)
    consumers = consumer_rows(code_rows)
    enriched = enrich_services(services, log_rows, consumers)
    smoke_rows = smoke_predict() if smoke else []
    payload = {
        "schema_version": "ops_refresh_a_v1",
        "created_at": now_iso(),
        "repo_root": str(REPO_ROOT),
        "mode": "READ_ONLY",
        "pm2_source": pm2_source,
        "services": enriched,
        "code_refs_count": len(code_rows),
        "code_refs": code_rows[:5000],
        "outputs": outputs,
        "consumers": consumers,
        "log_findings": log_rows,
        "db_inventory": db_inventory(),
        "smoke_predict": smoke_rows,
        "safety_confirmations": {
            "pm2_restarted": False,
            "services_paused": False,
            "processes_killed": False,
            "live_changes": False,
            "yaml_changed": False,
            "active_manifest_changed": False,
            "orders_sent": False,
            "env_changed": False,
            "push": False,
            "commit": False,
        },
    }
    return payload


def write_reports(payload: dict[str, Any], out_dir: Path) -> dict[str, str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "summary_md": out_dir / "ops_refresh_a_summary.md",
        "summary_json": out_dir / "ops_refresh_a_summary.json",
        "services_csv": out_dir / "ops_refresh_a_services.csv",
        "code_refs_csv": out_dir / "ops_refresh_a_code_refs.csv",
        "outputs_csv": out_dir / "ops_refresh_a_outputs.csv",
        "consumers_csv": out_dir / "ops_refresh_a_consumers.csv",
        "log_findings_csv": out_dir / "ops_refresh_a_log_findings.csv",
        "smoke_predict_json": out_dir / "smoke_predict_current_refresh_state.json",
        "recent_outputs_txt": out_dir / "recent_turbo_snapshot_files.txt",
    }
    payload["paths"] = {key: str(value) for key, value in paths.items()}
    paths["summary_json"].write_text(json.dumps(json_safe(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_markdown(paths["summary_md"], payload)
    write_csv(paths["services_csv"], payload["services"])
    write_csv(paths["code_refs_csv"], payload["code_refs"], ("file", "line", "terms", "text"))
    write_csv(paths["outputs_csv"], payload["outputs"])
    write_csv(paths["consumers_csv"], payload["consumers"])
    write_csv(paths["log_findings_csv"], payload["log_findings"])
    paths["smoke_predict_json"].write_text(json.dumps(payload["smoke_predict"], indent=2, sort_keys=True) + "\n", encoding="utf-8")
    paths["recent_outputs_txt"].write_text(
        "\n".join(f"{row.get('mtime')} {row.get('size_bytes')} {row.get('path')}" for row in payload["outputs"][:200]) + "\n",
        encoding="utf-8",
    )
    return {key: str(value) for key, value in paths.items()}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--no-smoke", action="store_true")
    args = parser.parse_args()
    out_dir = Path(args.out_dir)
    payload = audit(out_dir, smoke=not args.no_smoke)
    paths = write_reports(payload, out_dir)
    print(
        json.dumps(
            {
                "reports": paths,
                "services": [
                    {
                        "name": row["name"],
                        "pm2_status": row["pm2_status"],
                        "health": row["health"],
                        "phase_o_dependency": row["phase_o_dependency"],
                        "recommendation": row["recommendation"],
                    }
                    for row in payload["services"]
                ],
                "smoke_ok": all(row.get("ok") for row in payload["smoke_predict"]),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
