from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[2]))

from aegis_alpha.tools import audit_phase_o_runtime_safety as audit


def assert_true(value, message="assertion failed"):
    if not value:
        raise AssertionError(message)


def test_classify_safe_passive():
    row = {"model_paths_point_to_phase_o": False, "active_manifest_has_phase_o_fields": False, "runtime_uses_model_paths_directly": True, "yaml_guard_detected": False, "affects_orders": False, "affects_gating": False, "affects_sizing": False, "symbol": "LTCUSDT"}
    assert_true(audit.classify_phase_o_runtime_risk(row) == "SAFE_PASSIVE")


def test_classify_live_path_replaced():
    row = {"model_paths_point_to_phase_o": True, "runtime_uses_model_paths_directly": True, "yaml_guard_detected": False, "affects_orders": False, "affects_gating": False, "affects_sizing": False, "symbol": "LTCUSDT"}
    assert_true(audit.classify_phase_o_runtime_risk(row) == "LIVE_PATH_REPLACED")


def test_classify_dangerous_link_entry_and_affects_orders():
    assert_true(audit.classify_phase_o_runtime_risk({"symbol": "LINKUSDT", "entry_enabled": True}) == "DANGEROUS")
    assert_true(audit.classify_phase_o_runtime_risk({"symbol": "LTCUSDT", "affects_orders": True}) == "DANGEROUS")


def test_parser_detects_model_paths_and_phase_o():
    manifest = {"model_paths": {"short_7d": "/x/active/phase_o_20260101/model.joblib", "long_7d": "/x/long.joblib"}}
    paths = audit.parse_short_model_paths(manifest)
    assert_true(paths == {"short_7d": "/x/active/phase_o_20260101/model.joblib"})
    assert_true(audit.is_phase_o_path(paths["short_7d"]))


def test_global_risk_needs_fix_when_live_replaced():
    rows = [{"risk_class": "SAFE_PASSIVE"}, {"risk_class": "LIVE_PATH_REPLACED"}]
    assert_true(audit.classify_global_risk(rows) == "NEEDS_FIX_BEFORE_YAML")


def test_report_serializes_json_csv():
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        payload = {
            "created_stamp": "20260101T000000Z",
            "global_risk": "NEEDS_FIX_BEFORE_YAML",
            "biggest_risk": "test",
            "phase_o_already_affects_runtime": True,
            "code_findings": {
                "turbo_signal_loads_active_manifest": True,
                "turbo_signal_uses_model_paths_directly": True,
                "turbo_signal_phase_o_aware": False,
                "server_calls_evaluate_turbo_shadow": True,
                "yaml_guard_detected": False,
            },
            "symbol_rows": [
                {
                    "symbol": "LTCUSDT",
                    "active_manifest_has_phase_o_fields": True,
                    "model_paths_point_to_phase_o": True,
                    "shadow_type": "entry_model",
                    "risk_class": "LIVE_PATH_REPLACED",
                    "risk_reason": "test",
                    "phase_o_prod_ready": True,
                    "yaml_shadow_expected": True,
                    "phase_o_artifact_stamp": "s",
                    "has_phase_o_symbols": True,
                    "phase_o_model_path_keys": ["short_14d"],
                    "has_backup_previous_paths": False,
                    "entry_enabled": True,
                    "avoid_only": False,
                    "affects_orders": False,
                    "affects_gating": False,
                    "affects_sizing": False,
                    "affects_decision": True,
                    "runtime_can_read_phase_o_without_yaml": True,
                },
                {
                    "symbol": "LINKUSDT",
                    "active_manifest_has_phase_o_fields": True,
                    "model_paths_point_to_phase_o": False,
                    "shadow_type": "avoid_only_filter",
                    "risk_class": "PASSIVE_BUT_AMBIGUOUS",
                    "risk_reason": "test",
                    "entry_enabled": False,
                    "avoid_only": True,
                    "link_runtime_confusion_risk": False,
                },
            ],
            "code_refs": [{"file": "x.py", "line": 1, "pattern": "model_paths", "text": "model_paths"}],
            "model_path_rows": [{"symbol": "LTCUSDT", "source": "active", "key": "short_14d", "path": "/x/active/phase_o_s/m.joblib", "points_to_phase_o": True, "exists": False, "would_runtime_load": True}],
            "recommendations": ["fix manifests first"],
        }
        paths = audit.write_reports(tmp, payload)
        assert_true(Path(paths["json"]).exists())
        assert_true(Path(paths["symbols_csv"]).exists())
        loaded = json.loads(Path(paths["json"]).read_text())
        assert_true(loaded["global_risk"] == "NEEDS_FIX_BEFORE_YAML")


def test_auditor_does_not_write_manifests_and_no_model_file_requirement():
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        manifest = tmp / "active_manifest.json"
        manifest.write_text(json.dumps({"model_paths": {"short_7d": str(tmp / "missing.joblib")}}), encoding="utf-8")
        before = manifest.read_text()
        parsed = audit.parse_short_model_paths(json.loads(before))
        assert_true("short_7d" in parsed)
        assert_true(manifest.read_text() == before)


if __name__ == "__main__":
    test_classify_safe_passive()
    test_classify_live_path_replaced()
    test_classify_dangerous_link_entry_and_affects_orders()
    test_parser_detects_model_paths_and_phase_o()
    test_global_risk_needs_fix_when_live_replaced()
    test_report_serializes_json_csv()
    test_auditor_does_not_write_manifests_and_no_model_file_requirement()
    print("manual_audit_phase_o_runtime_safety_tests_passed")
