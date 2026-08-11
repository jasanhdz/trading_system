from __future__ import annotations

from aegis.research.tail_aware_entry_v8_gate import evaluate_v8_gate


def valid_side() -> dict[str, object]:
    return {
        "validation_pass": True,
        "passing_folds": 3,
        "router_skilled_folds": 3,
        "late_detector_skilled_folds": 3,
        "worst_fold_non_negative": True,
        "leave_one_symbol_out": {"passed": True},
    }


def test_v8_gate_passes_only_complete_research_evidence() -> None:
    validation = {
        "verdict": "ELIGIBLE_FOR_SEPARATELY_AUTHORIZED_SHADOW",
        "sides": {"LONG": valid_side(), "SHORT": valid_side()},
        "exchange_calls": 0,
        "exchange_mutations": 0,
        "shadow_runtime_enabled": False,
        "model_exported": False,
    }
    assert evaluate_v8_gate(validation)["passed"] is True


def test_v8_gate_fails_closed_on_late_detector_or_runtime_effect() -> None:
    short = valid_side()
    short["late_detector_skilled_folds"] = 2
    validation = {
        "verdict": "ELIGIBLE_FOR_SEPARATELY_AUTHORIZED_SHADOW",
        "sides": {"LONG": valid_side(), "SHORT": short},
        "exchange_calls": 0,
        "exchange_mutations": 0,
        "shadow_runtime_enabled": True,
        "model_exported": False,
    }
    result = evaluate_v8_gate(validation)
    assert result["passed"] is False
    assert "SHORT_LATE_DETECTOR_SKILL_INSUFFICIENT" in result["blockers"]
    assert "SHADOW_ALREADY_ENABLED" in result["blockers"]
