from __future__ import annotations

import json
import hashlib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
EVIDENCE = ROOT / "reports/compatibility_replay/aegis-gen2-compatibility-replay-v1"


def _load(name: str) -> dict:
    return json.loads((EVIDENCE / name).read_text())


def _sha256(name: str) -> str:
    return hashlib.sha256((EVIDENCE / name).read_bytes()).hexdigest()


def test_stage_1b_official_report_is_bound_to_both_attempts() -> None:
    report = _load("official_followup_report.json")["stage_1b"]
    first = _load("stage_1b/attempt_1/result.json")
    second = _load("stage_1b/attempt_2/result.json")
    assert first["scientific_hash"] == second["scientific_hash"] == report["scientific_hash"]
    assert first["trade_keys_hash"] == second["trade_keys_hash"] == report["trade_keys_hash"]
    assert first["dataset_hash"] == second["dataset_hash"] == report["dataset_hash"]
    assert report["minimum_trades_each_fold_satisfied"] is True
    assert all(item["trades"] >= 100 for item in report["folds"].values())
    assert report["answer"] == "EDGE_ABSENT_ON_HOURLY"


def test_stage_4b_official_report_is_bound_to_closed_variants() -> None:
    report = _load("official_followup_report.json")["stage_4b"]
    source = _load("stage_4b.json")
    assert [item["stage"] for item in source["variants"]] == ["STAGE_4B_A", "STAGE_4B_B", "STAGE_4B_C"]
    for item in source["variants"]:
        frozen = report["variants"][item["stage"]]
        assert item["scientific_hash"] == frozen["scientific_hash"]
        assert item["metrics"]["trades"] == frozen["trades"]
        assert item["metrics"]["net_profit_factor"] == frozen["profit_factor"]
        assert item["metrics"]["net_expectancy"] == frozen["net_expectancy"]
        assert item["determinism"]["canonical_identical"] is True
        assert item["determinism"]["trade_keys_identical"] is True


def test_followup_evidence_records_no_prohibited_execution() -> None:
    report = _load("official_followup_report.json")
    assert report["verdict"] == "READY_FOR_E3_VALIDATION"
    assert set(report["safety"].values()) == {False}
    authority = json.loads((ROOT / "reports/experiments/lockbox_semi_blind_20260427_20260711.json").read_text())
    assert authority["status"] == "NOT_CONSUMED"
    assert authority["consumed_queries"] == []


def test_official_followup_manifest_binds_physical_and_scientific_results() -> None:
    manifest = _load("official_followup_manifest.json")
    assert manifest["stage_1b"]["attempt_1_physical_sha256"] == _sha256(
        "stage_1b/attempt_1/result.json"
    )
    assert manifest["stage_1b"]["attempt_2_physical_sha256"] == _sha256(
        "stage_1b/attempt_2/result.json"
    )
    assert manifest["stage_4b"]["summary_physical_sha256"] == _sha256("stage_4b.json")
    assert manifest["reports"]["json_physical_sha256"] == _sha256("official_followup_report.json")
    assert manifest["verdict"] == "READY_FOR_E3_VALIDATION"
    assert manifest["stage_1b"]["deterministic"] is True
    assert manifest["stage_4b"]["deterministic"] is True
