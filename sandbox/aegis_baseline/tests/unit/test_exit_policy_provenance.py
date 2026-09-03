from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
AUDIT = ROOT / "reports/governance/exit_policy_provenance_p1"


def _json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_provenance_manifest_binds_all_audit_artifacts() -> None:
    manifest = _json(AUDIT / "provenance_manifest.json")
    assert manifest["classification"] == "READ_ONLY_HISTORICAL_PROVENANCE_AUDIT"
    for reference in manifest["artifacts"]:
        path = ROOT / reference["path"]
        assert path.is_file()
        assert _sha256(path) == reference["sha256"]


def test_provenance_keeps_frozen_science_and_lockbox_intact() -> None:
    manifest = _json(AUDIT / "provenance_manifest.json")
    for reference in manifest["frozen_evidence"]:
        path = ROOT / reference["path"]
        assert path.is_file()
        assert _sha256(path) == reference["sha256"]

    authority = _json(ROOT / "reports/experiments/lockbox_semi_blind_20260427_20260711.json")
    assert authority["status"] == "NOT_CONSUMED"
    assert authority["consumed_queries"] == []
    assert authority["maximum_queries_total"] == 1


def test_inventory_recovers_composite_policy_without_inventing_risk() -> None:
    summary = _json(AUDIT / "provenance_summary.json")
    inventory = _json(AUDIT / "phase_o_exit_rules_inventory.json")
    risk = _json(AUDIT / "risk_unit_analysis.json")
    rules = {rule["id"]: rule for rule in inventory["rules"]}

    assert summary["provenance_verdict"] == "PHASE_O_POLICY_PARTIALLY_RECOVERABLE"
    assert summary["e4_recommendation"] == "E4_REQUIRES_OWNER_RISK_DEFINITION"
    assert inventory["architectural_ownership"]["standalone_phase_o_exit_policy_found"] is False
    assert rules["INITIAL_STOP"]["value"] == -0.4
    assert rules["FIXED_TAKE_PROFIT"]["value"] == 0.5
    assert rules["TRAILING_ACTIVATION"]["value"] == 0.15
    assert rules["PARTIAL_CLOSE"]["present"] is False
    assert rules["PYTHON_EXITS_BLOCK"]["classification"] == "EXPERIMENTAL_OR_ABANDONED"
    assert risk["historical_r_definition_found"] is False
    assert risk["owner_definition_required"] is True


def test_architecture_review_is_pre_d1a_and_independently_traced() -> None:
    manifest = _json(AUDIT / "provenance_manifest.json")
    summary = _json(AUDIT / "provenance_summary.json")
    cutoff = manifest["temporal_cutoff"]["d1a_commit_timestamp_utc"]

    with (AUDIT / "evidence_table.csv").open(newline="", encoding="utf-8") as handle:
        rows = {row["evidence_id"]: row for row in csv.DictReader(handle)}

    review = rows["PRE_D1A_ARCH_REVIEW"]
    assert review["classification"] == "HISTORICALLY_DOCUMENTED"
    assert review["commit_date_utc"] < cutoff
    assert summary["post_d1a_evidence_used_as_authority"] is False
    assert summary["post_d1a_or_contaminated_evidence_found"] is False


def test_d1a_compatibility_fails_closed_on_missing_replay_inputs() -> None:
    assessment = _json(AUDIT / "d1a_compatibility_assessment.json")
    missing = assessment["missing_for_exact_historical_policy"]
    assert assessment["exact_application_without_new_decisions"] is False
    assert all(missing.values())
    assert assessment["d1a_source"]["trades"] == 1292
    assert assessment["d1a_source"]["bars_per_trade"] == 12


def test_operational_execution_remains_disabled_and_short_only() -> None:
    config = (ROOT / "binance-futures-bot-ts/config/regimen.config.yaml").read_text(encoding="utf-8")
    assert "enabledByConfig: false" in config
    assert "allowedSides:\n    - SHORT" in config
