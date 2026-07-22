from __future__ import annotations

import json
import stat
from pathlib import Path

import pytest

from aegis.prospective.evidence_audit import EvidenceAuditError, audit_evidence, write_report


def _signal(identity: str, cycle: str, symbol: str, selected: bool, score: float) -> dict:
    action = "ENTER_NOW" if selected else "DO_NOT_ENTER"
    components = {
        "d3": {"status": "PASS", "output": {"decision": action, "regime": "RANGE"}},
        "rv2": {"status": "PASS", "output": {"tail_risk_probability": 0.4}},
        "trrm": {"status": "PASS", "output": {"passed": True}},
        "qmae": {"status": "PASS", "output": {"valid": True}},
        "eqm": {"status": "PASS", "output": {"eligible": True}},
        "econ1": {
            "status": "PASS",
            "output": {"eligible": True, "expected_return": score * 2, "calibrated_score": score},
        },
    }
    return {
        "prospective_signal_id": identity,
        "evaluation_id": f"{cycle}:00",
        "signal_timestamp_utc": "2026-07-21T18:00:00Z",
        "symbol": symbol,
        "component_evidence": components,
        "final_decision": {"action": action},
        "upstream_model": {
            "short_probability": 0.99,
            "quality_probability": score,
            "tail_risk_probability": 0.4,
        },
    }


def _outcome(identity: str, value: float) -> dict:
    return {
        "prospective_signal_id": identity,
        "net_return_fraction": value,
        "tail_event": int(value < 0),
        "mfe_fraction": max(value, 0),
        "mae_fraction": max(-value, 0),
    }


def _jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def test_audit_joins_evidence_and_reports_observational_metrics(tmp_path: Path) -> None:
    signals = tmp_path / "signals.jsonl"
    outcomes = tmp_path / "outcomes.jsonl"
    _jsonl(signals, [
        _signal("a", "cycle-a", "AVAXUSDT", True, 0.2),
        _signal("b", "cycle-a", "ETHUSDT", False, 0.1),
        _signal("c", "cycle-b", "AVAXUSDT", True, 0.3),
        _signal("d", "cycle-b", "ETHUSDT", False, 0.05),
    ])
    _jsonl(outcomes, [
        _outcome("a", 0.01),
        _outcome("b", -0.01),
        _outcome("c", -0.02),
        _outcome("d", 0.005),
    ])

    report = audit_evidence(signals, outcomes, bootstrap_repetitions=20, seed=7)
    repeated = audit_evidence(signals, outcomes, bootstrap_repetitions=20, seed=7)

    assert report == repeated
    assert report["runtime_effect"] == "NONE_OBSERVATIONAL_ONLY"
    assert report["source"]["matured_join_count"] == 4
    assert report["decision_counts"] == {"selected": 2, "rejected": 2}
    assert report["by_symbol"]["AVAXUSDT"]["selected"]["count"] == 2
    assert report["variability"]["unique_short_probabilities"] == 1
    assert "BASE_DIRECTIONAL_SHORT_PROBABILITY_CONSTANT" in report["warnings"]
    assert "D3_DECISION_FIELD_NOT_INDEPENDENT_OF_FINAL_SELECTION" in report["warnings"]
    assert report["stage_economics"]["selected"]["passed"]["count"] == 2
    assert len(report["calibrated_score_ranking"]["deciles"]) == 4
    assert report["execution_recommendation"] == "NO_AUTOMATIC_RUNTIME_CHANGE"


def test_report_is_written_atomically_with_private_permissions(tmp_path: Path) -> None:
    output = tmp_path / "report.json"
    write_report({"status": "PASS"}, output)
    assert json.loads(output.read_text(encoding="utf-8")) == {"status": "PASS"}
    assert stat.S_IMODE(output.stat().st_mode) == 0o600


def test_duplicate_evidence_identity_fails_closed(tmp_path: Path) -> None:
    signals = tmp_path / "signals.jsonl"
    outcomes = tmp_path / "outcomes.jsonl"
    duplicate = _signal("same", "cycle-a", "AVAXUSDT", True, 0.2)
    _jsonl(signals, [duplicate, duplicate])
    _jsonl(outcomes, [_outcome("same", 0.01)])

    with pytest.raises(EvidenceAuditError, match="EVIDENCE_DUPLICATE_IDENTITY"):
        audit_evidence(signals, outcomes, bootstrap_repetitions=10)


def test_auditor_source_has_no_runtime_or_exchange_dependency() -> None:
    source = Path("src/aegis/prospective/evidence_audit.py").read_text(encoding="utf-8")
    forbidden = (
        "aegis.live_decision",
        "aegis.prospective.shadow_bridge",
        "binance",
        "requests",
        "subprocess",
        "pm2",
    )
    assert all(value not in source.lower() for value in forbidden)
