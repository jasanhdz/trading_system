"""Fail-closed historical gate for regime-aware directional v6 research."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping


@dataclass(frozen=True)
class ShadowGateDecision:
    eligible: bool
    status: str
    blockers: tuple[str, ...]


def assess_shadow_gate(report: Mapping[str, Any]) -> ShadowGateDecision:
    """Assess eligibility without activating, exporting, or selecting anything."""

    blockers: list[str] = []
    if report.get("schema_id") != "aegis-regime-aware-directional-v6-validation-v1":
        blockers.append("VALIDATION_SCHEMA_MISMATCH")
    if report.get("mode") != "RESEARCH_SHADOW":
        blockers.append("MODE_NOT_RESEARCH_SHADOW")
    if report.get("shadow_runtime_enabled") is not False:
        blockers.append("SHADOW_RUNTIME_ALREADY_ENABLED")
    if report.get("model_exported") is not False:
        blockers.append("MODEL_EXPORT_ALREADY_OCCURRED")
    if report.get("selection_effect") != "NONE":
        blockers.append("SELECTION_EFFECT_NOT_NONE")
    if int(report.get("exchange_calls", -1)) != 0:
        blockers.append("EXCHANGE_CALLS_NONZERO")
    if int(report.get("exchange_mutations", -1)) != 0:
        blockers.append("EXCHANGE_MUTATIONS_NONZERO")
    if report.get("validation_pass") is not True:
        blockers.append("HISTORICAL_VALIDATION_FAILED")
    if report.get("verdict") != "ELIGIBLE_FOR_SEPARATE_PROSPECTIVE_SHADOW_ACTIVATION":
        blockers.append("VALIDATION_VERDICT_NOT_ELIGIBLE")
    sides = report.get("sides")
    if not isinstance(sides, Mapping) or set(sides) != {"LONG", "SHORT"}:
        blockers.append("DIRECTIONAL_REPORT_INCOMPLETE")
    else:
        for side in ("LONG", "SHORT"):
            side_report = sides[side]
            if not isinstance(side_report, Mapping):
                blockers.append(f"{side}_REPORT_INVALID")
                continue
            if side_report.get("validation_pass") is not True:
                blockers.append(f"{side}_VALIDATION_FAILED")
            if int(side_report.get("evaluated_folds", -1)) != 4:
                blockers.append(f"{side}_FOLD_COVERAGE_INCOMPLETE")
            if int(side_report.get("passing_folds", -1)) < 3:
                blockers.append(f"{side}_POSITIVE_FOLDS_INSUFFICIENT")
            if int(side_report.get("regime_router_skilled_folds", -1)) < int(
                side_report.get("minimum_regime_router_skilled_folds", 3)
            ):
                blockers.append(f"{side}_REGIME_ROUTER_SKILL_INSUFFICIENT")
            if side_report.get("worst_fold_non_negative") is not True:
                blockers.append(f"{side}_WORST_FOLD_NEGATIVE")
            loso = side_report.get("leave_one_symbol_out")
            if not isinstance(loso, Mapping) or loso.get("passed") is not True:
                blockers.append(f"{side}_LEAVE_ONE_SYMBOL_OUT_FAILED")
    unique = tuple(dict.fromkeys(blockers))
    return ShadowGateDecision(
        eligible=not unique,
        status=(
            "READY_FOR_SEPARATELY_AUTHORIZED_PROSPECTIVE_SHADOW"
            if not unique
            else "RESEARCH_ONLY_NOT_PROMOTABLE"
        ),
        blockers=unique,
    )
