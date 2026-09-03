from __future__ import annotations

from aegis.research.regime_aware_directional_v6_gate import assess_shadow_gate


def report() -> dict[str, object]:
    side = {
        "validation_pass": True,
        "evaluated_folds": 4,
        "passing_folds": 3,
        "regime_router_skilled_folds": 3,
        "minimum_regime_router_skilled_folds": 3,
        "worst_fold_non_negative": True,
        "leave_one_symbol_out": {"passed": True},
    }
    return {
        "schema_id": "aegis-regime-aware-directional-v6-validation-v1",
        "mode": "RESEARCH_SHADOW",
        "shadow_runtime_enabled": False,
        "model_exported": False,
        "selection_effect": "NONE",
        "exchange_calls": 0,
        "exchange_mutations": 0,
        "validation_pass": True,
        "verdict": "ELIGIBLE_FOR_SEPARATE_PROSPECTIVE_SHADOW_ACTIVATION",
        "sides": {"LONG": dict(side), "SHORT": dict(side)},
    }


def test_gate_accepts_complete_historical_evidence_without_activating() -> None:
    decision = assess_shadow_gate(report())
    assert decision.eligible is True
    assert decision.blockers == ()


def test_gate_fails_closed_on_directional_failure_or_exchange_effect() -> None:
    value = report()
    value["exchange_mutations"] = 1
    value["sides"]["LONG"]["validation_pass"] = False  # type: ignore[index]
    decision = assess_shadow_gate(value)
    assert decision.eligible is False
    assert "EXCHANGE_MUTATIONS_NONZERO" in decision.blockers
    assert "LONG_VALIDATION_FAILED" in decision.blockers
