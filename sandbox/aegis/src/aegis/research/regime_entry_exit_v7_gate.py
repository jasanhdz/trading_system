"""Fail-closed deployment gate for V7 research evidence."""

from __future__ import annotations

from typing import Any, Mapping


def evaluate_v7_gate(validation: Mapping[str, Any]) -> Mapping[str, Any]:
    blockers = []
    if validation.get("verdict") != "ELIGIBLE_FOR_SEPARATELY_AUTHORIZED_SHADOW":
        blockers.append("HISTORICAL_VALIDATION_FAILED")
    for side in ("LONG", "SHORT"):
        value = validation.get("sides", {}).get(side, {})
        if not bool(value.get("validation_pass")):
            blockers.append(f"{side}_VALIDATION_FAILED")
        if int(value.get("passing_folds", 0)) < 3:
            blockers.append(f"{side}_POSITIVE_FOLDS_INSUFFICIENT")
        if int(value.get("router_skilled_folds", 0)) < 3:
            blockers.append(f"{side}_REGIME_ROUTER_SKILL_INSUFFICIENT")
        if not bool(value.get("worst_fold_non_negative")):
            blockers.append(f"{side}_WORST_FOLD_NEGATIVE")
        if not bool(value.get("leave_one_symbol_out", {}).get("passed")):
            blockers.append(f"{side}_LEAVE_ONE_SYMBOL_OUT_FAILED")
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
