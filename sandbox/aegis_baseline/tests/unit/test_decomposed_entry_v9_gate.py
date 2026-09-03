from __future__ import annotations

from aegis.research.decomposed_entry_v9_gate import evaluate_v9_gate


def side() -> dict[str, object]:
    return {
        "validation_pass": True,
        "passing_folds": 3,
        "direction_skilled_folds": 3,
        "timing_skilled_folds": 3,
        "trajectory_skilled_folds": 3,
        "worst_fold_non_negative": True,
        "leave_one_symbol_out": {"passed": True},
    }


def test_v9_gate_requires_all_three_components() -> None:
    validation = {
        "verdict": "ELIGIBLE_FOR_SEPARATELY_AUTHORIZED_SHADOW",
        "sides": {"LONG": side(), "SHORT": side()},
        "exchange_calls": 0,
        "exchange_mutations": 0,
        "shadow_runtime_enabled": False,
        "model_exported": False,
    }
    assert evaluate_v9_gate(validation)["passed"] is True
    validation["sides"]["SHORT"]["timing_skilled_folds"] = 2
    result = evaluate_v9_gate(validation)
    assert result["passed"] is False
    assert "SHORT_TIMING_SKILL_INSUFFICIENT" in result["blockers"]
