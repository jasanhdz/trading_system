from __future__ import annotations

from aegis.research.regime_entry_exit_v7_gate import evaluate_v7_gate


def validation(passed: bool) -> dict[str, object]:
    side = {
        "validation_pass": passed,
        "passing_folds": 3 if passed else 0,
        "router_skilled_folds": 3 if passed else 0,
        "worst_fold_non_negative": passed,
        "leave_one_symbol_out": {"passed": passed},
    }
    return {
        "verdict": (
            "ELIGIBLE_FOR_SEPARATELY_AUTHORIZED_SHADOW"
            if passed
            else "RESEARCH_ONLY_NOT_PROMOTABLE"
        ),
        "sides": {"LONG": side, "SHORT": side},
        "exchange_calls": 0,
        "exchange_mutations": 0,
        "shadow_runtime_enabled": False,
        "model_exported": False,
    }


def test_gate_passes_only_complete_historical_evidence() -> None:
    assert evaluate_v7_gate(validation(True))["passed"] is True
    blocked = evaluate_v7_gate(validation(False))
    assert blocked["passed"] is False
    assert "LONG_VALIDATION_FAILED" in blocked["blockers"]
