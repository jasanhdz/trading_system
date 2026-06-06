#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import sys
import tempfile
from pathlib import Path

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from aegis_alpha.tools.audit_turbo_refreshers_dependencies import (
    analyze_log_text,
    classify_health,
    classify_phase_dependency,
    consumer_rows,
    enrich_services,
    parse_symbols_from_args,
    write_csv,
)


def test_classifies_indirect_required_for_phase_o_entry_refresh_symbol() -> None:
    service = {
        "name": "06-Aegis-Turbo-Refresh-C",
        "role_inferred": "feature_snapshot_refresh",
        "symbols": ["AVAXUSDT", "LINKUSDT", "SUIUSDT", "LTCUSDT"],
    }

    assert classify_phase_dependency(service, []) == "INDIRECT_REQUIRED"


def test_classifies_unknown_when_no_phase_o_symbol_or_consumer() -> None:
    service = {
        "name": "custom-refresh",
        "role_inferred": "feature_snapshot_refresh",
        "symbols": ["FOOUSDT"],
    }

    assert classify_phase_dependency(service, []) == "UNKNOWN"


def test_detects_sqlite_contention_in_logs() -> None:
    finding = analyze_log_text("ERROR sqlite3.OperationalError: database is locked\n")
    service = {"pm2_status": "online", "cpu": 0}

    assert finding["sqlite_locked_count"] == 1
    assert classify_health(service, finding) == "SQLITE_CONTENTION_RISK"


def test_detects_traceback_as_errored() -> None:
    finding = analyze_log_text("Traceback (most recent call last):\nException: boom\n")
    service = {"pm2_status": "online", "cpu": 0}

    assert finding["traceback_count"] == 1
    assert classify_health(service, finding) == "ERRORED"


def test_detects_hot_loop_cpu_from_pm2_cpu() -> None:
    finding = analyze_log_text("refresh\n")
    service = {"pm2_status": "online", "cpu": 99.0}

    assert classify_health(service, finding) == "HOT_LOOP_CPU"


def test_parse_symbols_from_pm2_args() -> None:
    args = "aegis_alpha/tools/refresh_turbo_snapshots.py --mode features-only --symbols ETHUSDT,BTCUSDT,SOLUSDT"

    assert parse_symbols_from_args(args) == ["ETHUSDT", "BTCUSDT", "SOLUSDT"]


def test_consumer_rows_mark_known_consumers_found() -> None:
    rows = [
        {"file": "aegis_alpha/turbo/turbo_signal.py", "line": 1, "terms": "snapshot", "text": "load_turbo_snapshot_status"},
        {"file": "aegis_alpha/inference/server.py", "line": 1, "terms": "ml-v2", "text": "/ml-v2/predict"},
    ]

    consumers = consumer_rows(rows)

    assert any(row["consumer_file"] == "aegis_alpha/turbo/turbo_signal.py" and row["consumer_found"] for row in consumers)


def test_enrich_services_serializes_to_csv_and_json() -> None:
    service = {
        "name": "04-Aegis-Turbo-Refresh-A",
        "pm2_status": "online",
        "pid": 123,
        "restart_count": 1,
        "cpu": 0,
        "memory_bytes": 1,
        "memory_mb": 0.01,
        "uptime_ms": 1000,
        "cwd": "/repo",
        "exec": "python",
        "args": [],
        "args_text": "",
        "role_inferred": "feature_snapshot_refresh",
        "symbols": ["ETHUSDT"],
    }
    logs = [{"service": "04-Aegis-Turbo-Refresh-A", **analyze_log_text('"success": true\n')}]

    enriched = enrich_services([service], logs, [])

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "services.csv"
        write_csv(path, enriched)
        text = path.read_text()
        assert "04-Aegis-Turbo-Refresh-A" in text
        rows = list(csv.DictReader(path.open()))
        assert rows[0]["phase_o_dependency"] == "INDIRECT_REQUIRED"
        json.dumps(enriched)


def test_no_pm2_logs_missing_does_not_fail() -> None:
    finding = analyze_log_text("")
    service = {"pm2_status": "stopped", "cpu": 0}

    assert classify_health(service, finding) == "STOPPED_BY_DESIGN"


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
    print("audit_turbo_refreshers_dependencies tests passed")
