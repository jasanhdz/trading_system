"""Fail-closed promotion gate for decomposed V9 evidence."""

from __future__ import annotations

from typing import Any, Mapping


def evaluate_v9_gate(validation: Mapping[str, Any]) -> Mapping[str, Any]:
    blockers = []
    if validation.get("verdict") != "ELIGIBLE_FOR_SEPARATELY_AUTHORIZED_SHADOW":
        blockers.append("HISTORICAL_VALIDATION_FAILED")
    for side in ("LONG", "SHORT"):
        result = validation.get("sides", {}).get(side, {})
        checks = {
            "VALIDATION_FAILED": bool(result.get("validation_pass")),
            "POSITIVE_FOLDS_INSUFFICIENT": int(result.get("passing_folds", 0)) >= 3,
            "DIRECTION_SKILL_INSUFFICIENT": int(
                result.get("direction_skilled_folds", 0)
            )
            >= 3,
            "TIMING_SKILL_INSUFFICIENT": int(result.get("timing_skilled_folds", 0))
            >= 3,
            "TRAJECTORY_SKILL_INSUFFICIENT": int(
                result.get("trajectory_skilled_folds", 0)
            )
            >= 3,
            "WORST_FOLD_NEGATIVE": bool(result.get("worst_fold_non_negative")),
            "LEAVE_ONE_SYMBOL_OUT_FAILED": bool(
                result.get("leave_one_symbol_out", {}).get("passed")
            ),
        }
        blockers.extend(
            f"{side}_{name}" for name, passed in checks.items() if not passed
        )
    if int(validation.get("exchange_calls", -1)) != 0:
        blockers.append("EXCHANGE_CALL_COUNTER_NONZERO")
    if int(validation.get("exchange_mutations", -1)) != 0:
        blockers.append("EXCHANGE_MUTATION_COUNTER_NONZERO")
    if bool(validation.get("shadow_runtime_enabled")):
        blockers.append("SHADOW_ALREADY_ENABLED")
    if bool(validation.get("model_exported")):
        blockers.append("MODEL_ALREADY_EXPORTED")
    return {
        "passed": not blockers,
        "blockers": blockers,
        "decision": (
            "ELIGIBLE_FOR_SEPARATE_SHADOW_AUTHORIZATION"
            if not blockers
            else "RESEARCH_ONLY_NOT_PROMOTABLE"
        ),
        "runtime_effect": "NONE",
        "exchange_calls": 0,
        "exchange_mutations": 0,
    }
