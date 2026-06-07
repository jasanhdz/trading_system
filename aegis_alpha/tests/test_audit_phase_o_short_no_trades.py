#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
import tempfile
from collections import Counter
from pathlib import Path

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from aegis_alpha.tools.audit_phase_o_short_no_trades import (  # noqa: E402
    build_funnel,
    classify_manifest,
    classify_predict,
    classify_stage,
    diagnose,
    write_csv,
)


def test_manifest_ok_detected() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        phase_dir = Path(tmp) / "phase_o_stamp"
        phase_dir.mkdir()
        model = phase_dir / "turbo_short_edge_14d_phase_o_stamp.joblib"
        model.write_text("x")
        manifest = {"phase_o_live_enabled": True, "phase_o_overlay_persistence_enabled": True}
        paths = {"short_14d": str(model)}
        assert classify_manifest("ETHUSDT", manifest, True, paths) == "PHASE_O_MANIFEST_OK"


def test_manifest_drift_detected() -> None:
    manifest = {"phase_o_live_enabled": True, "phase_o_overlay_persistence_enabled": True}
    paths = {"short_14d": "/tmp/base_model.joblib"}
    assert classify_manifest("ETHUSDT", manifest, True, paths) == "PHASE_O_MANIFEST_DRIFTED"


def test_link_avoid_only_ok_detected() -> None:
    manifest = {"phase_o_avoid_only": True, "phase_o_live_enabled": False, "phase_o_link_entry_enabled": False}
    assert classify_manifest("LINKUSDT", manifest, True, {}) == "LINK_AVOID_ONLY_OK"


def test_predict_hold_classifies_no_edge() -> None:
    phase = {"phase_o_live_enabled": True, "phase_o_link_avoid_only": False, "phase_o_link_entry_enabled": False}
    assert classify_predict("ETHUSDT", 200, "HOLD", "insufficient_recent_model_agreement", phase, None) == "PREDICT_HOLD_NO_EDGE"


def test_predict_timeout_classifies_timeout() -> None:
    assert classify_predict("ETHUSDT", None, None, None, {}, "TimeoutError('timed out')") == "PREDICT_TIMEOUT"


def test_stage_guard_blocked() -> None:
    c = Counter({"SIGNAL_RECEIVED": 5, "SHORT_GATE_DENIED": 5, "SHORT_SECONDARY_GUARD": 5})
    assert classify_stage(c, "clean_entry_blocked") == "STAGE_TS_RECOGNIZED_BUT_GUARD_BLOCKED"


def test_stage_order_submitted_without_position_is_order_issue() -> None:
    c = Counter({"SIGNAL_RECEIVED": 1, "SHORT_ORDER_SUBMITTED": 1})
    assert classify_stage(c, None) == "STAGE_ORDER_REJECTED"


def test_root_cause_no_short_edge_when_all_hold() -> None:
    manifests = [{"symbol": "ETHUSDT", "status": "PHASE_O_MANIFEST_OK"}]
    snapshots = [{"symbol": "ETHUSDT", "status": "SNAPSHOT_OK"}]
    predicts = [{"symbol": "ETHUSDT", "status": "PREDICT_HOLD_NO_EDGE", "action": "HOLD", "reason": "insufficient_recent_model_agreement"}]
    log_audit = {"funnel": {"strict_phase_o_short_order_submitted": 0, "strict_phase_o_short_position_confirmed": 0}, "block_rows": []}
    assert diagnose(manifests, snapshots, predicts, log_audit)["root_cause"] == "ROOT_CAUSE_NO_SHORT_EDGE"


def test_root_cause_manifest_drift() -> None:
    manifests = [{"symbol": "ETHUSDT", "status": "PHASE_O_MANIFEST_DRIFTED"}]
    assert diagnose(manifests, [], [], {"funnel": {}, "block_rows": []})["root_cause"] == "ROOT_CAUSE_PHASE_O_MANIFEST_DRIFT"


def test_root_cause_guard_still_blocking_if_clean_entry_blocks_phase_o() -> None:
    manifests = [{"symbol": "ETHUSDT", "status": "PHASE_O_MANIFEST_OK"}]
    predicts = [{"symbol": "ETHUSDT", "status": "PREDICT_PHASE_O_OK", "action": "SHORT", "reason": "raw_recent_short_agreement_2_of_3"}]
    log_audit = {"funnel": {"phase_o_guard_modes_applied": 1}, "block_rows": [{"side": "SHORT", "strict_phase_o": True, "guard_family": "secondary_guard", "reason": "clean_entry_blocked"}]}
    assert diagnose(manifests, [], predicts, log_audit)["root_cause"] == "ROOT_CAUSE_GUARD_STILL_BLOCKING"


def test_root_cause_ts_metadata_not_recognized_if_short_gate_blocks_without_phase_o_marker() -> None:
    manifests = [{"symbol": "ETHUSDT", "status": "PHASE_O_MANIFEST_OK"}]
    predicts = [{"symbol": "ETHUSDT", "status": "PREDICT_PHASE_O_OK", "action": "SHORT", "reason": "raw_recent_short_agreement_2_of_3"}]
    log_audit = {"funnel": {"phase_o_guard_modes_applied": 0, "short_gate_denied": 1}, "block_rows": [{"side": "SHORT", "strict_phase_o": False, "guard_family": "short_gate", "reason": "short_score_below_premium_threshold"}]}
    assert diagnose(manifests, [], predicts, log_audit)["root_cause"] == "ROOT_CAUSE_TS_METADATA_NOT_RECOGNIZED"


def test_root_cause_hard_safety_blocking() -> None:
    manifests = [{"symbol": "ETHUSDT", "status": "PHASE_O_MANIFEST_OK"}]
    snapshots = [{"symbol": "ETHUSDT", "status": "SNAPSHOT_OK"}]
    predicts = [{"symbol": "ETHUSDT", "status": "PREDICT_PHASE_O_OK", "action": "SHORT"}]
    log_audit = {"funnel": {"strict_phase_o_short_order_submitted": 0}, "block_rows": [{"side": "SHORT", "guard_family": "hard_safety", "reason": "risk_guard_max_phase_o_trades_per_day"}]}
    assert diagnose(manifests, snapshots, predicts, log_audit)["root_cause"] == "ROOT_CAUSE_HARD_SAFETY_BLOCKING"



def test_link_avoid_only_blocks_do_not_count_as_entry_attempts() -> None:
    funnel = build_funnel(Counter({"SIGNAL_RECEIVED": 1}), [], [{"symbol": "LINKUSDT", "event": "GATE_DENIED", "side": "SHORT", "reason": "phase_o_link_avoid_only_no_entry"}])
    assert funnel["link_entry_attempts"] == 0


def test_json_csv_serializes() -> None:
    rows = [{"symbol": "ETHUSDT", "nested": {"x": 1}}]
    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp) / "x.csv"
        write_csv(p, rows)
        assert "ETHUSDT" in p.read_text()
        json.dumps(rows)


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
    print("audit_phase_o_short_no_trades tests passed")
