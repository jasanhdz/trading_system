#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from aegis_alpha.tools.audit_forward_live_events_a import (
    build_trade_rows,
    classify_event,
    discover_log_files,
    iter_log_events,
    run_audit,
)


def test_no_logs_status() -> None:
    with tempfile.TemporaryDirectory() as td:
        result = run_audit(argparse.Namespace(out_dir=td, log_roots=[td]))
        assert result["status"] == "NO_LIVE_LOGS_AVAILABLE"
        assert Path(result["outputs"]["json"]).exists()


def test_parse_open_closed_fixture() -> None:
    with tempfile.TemporaryDirectory() as td:
        log = Path(td) / "forward.jsonl"
        log.write_text(
            "\n".join([
                json.dumps({"timestamp": "2026-07-09T00:00:00Z", "event": "TRADE_OPEN", "trade_id": "t1", "symbol": "ADAUSDT", "side": "SHORT", "entry_price": 1.0, "qty": 10}),
                json.dumps({"timestamp": "2026-07-09T00:10:00Z", "event": "TRADE_CLOSED", "trade_id": "t1", "symbol": "ADAUSDT", "exit_price": 0.98, "realized_pnl": 12.3, "reason": "tp"}),
            ]) + "\n",
            encoding="utf-8",
        )
        files = discover_log_files([Path(td)])
        events = iter_log_events(files)
        trades = build_trade_rows(events)
        assert len(events) == 2
        assert events[0]["event_type"] == "TRADE_OPEN"
        assert trades[0]["is_closed"] is True
        assert trades[0]["realized_pnl"] == 12.3


def test_classifies_events() -> None:
    assert classify_event({"message": "signal emitted"}, "") == "SIGNAL"
    assert classify_event({"message": "order error rejected"}, "") == "ORDER_ERROR"
    assert classify_event({"message": "bracket confirmed"}, "") == "BRACKET_CONFIRMED"


def test_serializes_outputs() -> None:
    with tempfile.TemporaryDirectory() as td:
        result = run_audit(argparse.Namespace(out_dir=td, log_roots=[td]))
        with Path(result["outputs"]["events_csv"]).open(newline="", encoding="utf-8") as f:
            assert csv.DictReader(f).fieldnames is not None


if __name__ == "__main__":
    test_no_logs_status()
    test_parse_open_closed_fixture()
    test_classifies_events()
    test_serializes_outputs()
    print("test_audit_forward_live_events_a: OK")
