#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from aegis_alpha.tools.audit_phase_o_short_live_entries import (  # noqa: E402
    classify_entry,
    classify_manifest_row,
    classify_signal_match,
    classify_snapshot,
    classify_system,
    detect_machine_gun,
    entry_audit_score,
    quality_label,
    signal_model_consistent,
    validate_guards,
    validate_hard_safety,
    validate_sizing,
    write_csv,
)


def phase_o_trade(symbol: str = "ETHUSDT", ts: str = "2026-06-07T01:00:00Z", trade_id: str = "T1") -> dict:
    return {
        "trade_id": trade_id,
        "symbol": symbol,
        "side": "SHORT",
        "strategy": "AEGIS_TURBO",
        "opened_at": ts,
        "entry_price": 100.0,
        "quantity": 1.0,
        "leverage": 20,
        "position_fraction": 0.35,
        "brackets_confirmed": True,
        "turbo_score": 0.72,
        "metadata": {
            "entryPolicy": {
                "guards": {
                    "short_gate": {"mode": "SHADOW", "enforced": False},
                    "clean_entry": {"mode": "SHADOW", "enforced": False},
                    "event_risk": {"mode": "SHADOW", "enforced": False},
                    "entry_quality": {"mode": "SHADOW", "enforced": False},
                    "decision_brain": {"mode": "SHADOW", "enforced": False},
                    "regime": {"mode": "SHADOW", "enforced": False},
                }
            }
        },
    }


def short_signal(symbol: str = "ETHUSDT", ts: str = "2026-06-07T00:59:50Z") -> dict:
    return {
        "signal_id": "S1",
        "symbol": symbol,
        "timestamp": ts,
        "raw_action": "SHORT",
        "gated_action": "SHORT",
        "final_action": "SHORT",
        "reason": "raw_recent_short_agreement_2_of_3",
        "turbo_score": 0.72,
        "position_fraction": 0.35,
        "leverage": 20,
        "freshness": {
            "exists": True,
            "is_fresh": True,
            "stale": False,
            "feature_age_seconds": 120,
            "max_feature_age_seconds": 900,
        },
    }


def good_checks() -> dict:
    return {
        "manifest_status": "MODEL_MANIFEST_OK",
        "snapshot_status": "SNAPSHOT_OK_FRESH",
        "signal_status": "SIGNAL_MATCH_OK",
        "model_status": "MODEL_DECISION_CONSISTENT",
        "sizing_status": "MODEL_DECISION_CONSISTENT",
        "guard_status": "GUARDS_EXPECTED_PHASE_O_SHADOW",
        "hard_safety_status": "HARD_SAFETY_OK",
        "machine_gun_status": "MACHINE_GUN_NONE",
    }


def test_trade_with_matching_phase_o_signal_valid() -> None:
    trade = phase_o_trade()
    signal = short_signal()
    assert classify_signal_match(trade, signal, datetime(2026, 6, 7, 1, 0, tzinfo=timezone.utc)) == "SIGNAL_MATCH_OK"
    checks = good_checks()
    score = entry_audit_score(checks)
    assert score == 100
    assert classify_entry(checks, score) == "VALID_PHASE_O_SHORT_ENTRY"


def test_trade_with_missing_signal_questionable() -> None:
    checks = good_checks()
    checks["signal_status"] = "SIGNAL_MISSING"
    score = entry_audit_score(checks)
    assert score < 70
    assert classify_entry(checks, score) == "QUESTIONABLE_PHASE_O_SHORT_ENTRY"


def test_raw_hold_but_ordered_is_model_conflict() -> None:
    signal = short_signal()
    signal["raw_action"] = "HOLD"
    signal["final_action"] = "SHORT"
    assert signal_model_consistent(signal) == "MODEL_HOLD_BUT_ORDERED"


def test_stale_snapshot_entry() -> None:
    signal = short_signal()
    signal["freshness"]["stale"] = True
    assert classify_snapshot(signal) == "SNAPSHOT_STALE"
    checks = good_checks()
    checks["snapshot_status"] = "SNAPSHOT_STALE"
    score = entry_audit_score(checks)
    assert classify_entry(checks, score) == "STALE_DATA_ENTRY"


def test_link_entry_is_bug() -> None:
    trade = phase_o_trade("LINKUSDT")
    assert validate_hard_safety(trade, []) == "LINK_ENTRY_BUG"


def test_duplicate_same_symbol_within_window() -> None:
    trades = [
        phase_o_trade("ETHUSDT", "2026-06-07T01:00:00Z", "T1"),
        phase_o_trade("ETHUSDT", "2026-06-07T01:01:00Z", "T2"),
    ]
    result = detect_machine_gun(trades, 300)
    assert result["classification"] == "MACHINE_GUN_DUPLICATE_SYMBOL"


def test_multi_symbol_burst_detected() -> None:
    trades = [
        phase_o_trade("ETHUSDT", "2026-06-07T01:00:00Z", "T1"),
        phase_o_trade("BTCUSDT", "2026-06-07T01:00:10Z", "T2"),
        phase_o_trade("SOLUSDT", "2026-06-07T01:00:20Z", "T3"),
        phase_o_trade("ADAUSDT", "2026-06-07T01:00:30Z", "T4"),
    ]
    result = detect_machine_gun(trades, 300)
    assert result["classification"] == "MACHINE_GUN_MULTISYMBOL_BURST"


def test_manifest_path_mismatch() -> None:
    assert classify_manifest_row("ETHUSDT", {"ETHUSDT": {"status": "PHASE_O_MANIFEST_DRIFTED"}}, short_signal()) == "MANIFEST_OR_MODEL_MISMATCH"


def test_hard_safety_bypass_classifies_issue() -> None:
    trade = phase_o_trade()
    trade["brackets_confirmed"] = False
    assert validate_hard_safety(trade, []) == "BRACKETS_MISSING"
    checks = good_checks()
    checks["hard_safety_status"] = "BRACKETS_MISSING"
    assert classify_entry(checks, entry_audit_score(checks)) == "HARD_SAFETY_ISSUE"


def test_entry_audit_score_low_with_multiple_failures() -> None:
    checks = good_checks()
    checks.update({
        "snapshot_status": "SNAPSHOT_STALE",
        "model_status": "MODEL_HOLD_BUT_ORDERED",
        "machine_gun_status": "MACHINE_GUN_DUPLICATE_SYMBOL",
    })
    assert entry_audit_score(checks) < 50
    assert quality_label(entry_audit_score(checks)) == "BAD_ENTRY"


def test_guard_and_sizing_ok() -> None:
    trade = phase_o_trade()
    assert validate_guards(trade) == "GUARDS_EXPECTED_PHASE_O_SHADOW"
    assert validate_sizing(trade) == "MODEL_DECISION_CONSISTENT"


def test_system_watch_when_machine_gun_only() -> None:
    rows = [{"classification": "DUPLICATE_OR_MACHINE_GUN_ENTRY", "entry_audit_score": 75}]
    assert classify_system(rows, {"classification": "MACHINE_GUN_MULTISYMBOL_BURST"}) == "PHASE_O_SHORT_LIVE_WATCH_CLOSELY"


def test_json_csv_serializes() -> None:
    row = {"symbol": "ETHUSDT", "nested": {"x": 1}}
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "rows.csv"
        write_csv(path, [row])
        assert "ETHUSDT" in path.read_text()
        json.dumps(row)


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
    print("audit_phase_o_short_live_entries tests passed")
