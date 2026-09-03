from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
CLOSURE = ROOT / "reports/governance/aegis-short-candidate-e3/closure"


def _json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_e3_closure_manifest_binds_every_tracked_artifact() -> None:
    manifest = _json(CLOSURE / "closure_manifest.json")
    for reference in manifest["closure_artifacts"].values():
        path = ROOT / reference["path"]
        assert path.is_file()
        assert _sha256(path) == reference["physical_sha256"]
    summary = ROOT / manifest["validation_evidence"]["summary_path"]
    assert _sha256(summary) == manifest["validation_evidence"]["summary_physical_sha256"]


def test_e3_frozen_contracts_and_final_disposition_are_immutable() -> None:
    manifest = _json(CLOSURE / "closure_manifest.json")
    frozen = manifest["frozen_contracts"]
    assert _sha256(ROOT / "config/experiments/aegis_short_candidate_e1.yaml") == frozen["e1_physical_sha256"]
    assert _sha256(ROOT / "config/experiments/aegis_short_candidate_e2.yaml") == frozen["e2_physical_sha256"]
    assert _sha256(ROOT / "config/experiments/aegis_short_candidate_e3.yaml") == frozen["e3_physical_sha256"]
    assert _sha256(ROOT / "config/scientific_competition_v1.yaml") == frozen["competition_v1_physical_sha256"]
    assert _sha256(ROOT / "config/scientific_competition_v2.yaml") == frozen["competition_v2_physical_sha256"]

    disposition = _json(CLOSURE / "final_disposition.json")
    assert disposition["formal_verdict"] == "E3_REJECTED_PRE_LOCKBOX"
    assert disposition["closed_permanently"] is True
    assert disposition["repeat_authorized"] is False
    assert set(disposition["rejection_classification"]) == {"PREDICTIVE", "ECONOMIC", "NOT_TECHNICAL"}


def test_e3_closure_preserves_lockbox_and_forbids_publication() -> None:
    manifest = _json(CLOSURE / "closure_manifest.json")
    authority = _json(ROOT / manifest["lockbox"]["authority_path"])
    assert _sha256(ROOT / manifest["lockbox"]["authority_path"]) == manifest["lockbox"]["authority_physical_sha256"]
    assert authority["status"] == "NOT_CONSUMED"
    assert authority["consumed_queries"] == []
    assert authority["maximum_queries_total"] == 1
    assert manifest["lockbox"]["budget_remaining"] == 1
    assert set(manifest["final_controls"].values()) == {False}

    for attempt in ("attempt_1", "attempt_2"):
        run = ROOT / (
            "reports/experiments/e3_validation_official/"
            f"{attempt}/aegis-short-candidate-e3/runs/d742d9bc0ae867bb"
        )
        for forbidden in ("lockbox_lease.json", "selection_policy.json", "system_freeze.json", "final_report.json"):
            assert not (run / forbidden).exists()


def test_e3_closure_keeps_operational_execution_disabled_and_short_only() -> None:
    config = (ROOT / "binance-futures-bot-ts/config/regimen.config.yaml").read_text(encoding="utf-8")
    assert "enabledByConfig: false" in config
    assert "allowedSides:\n    - SHORT" in config
