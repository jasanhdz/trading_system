from __future__ import annotations

from dataclasses import asdict, dataclass, field, is_dataclass
from typing import Any, Literal


DecisionBrainDecision = Literal["ENTER_NOW", "WAIT_CONFIRMATION", "MANUAL_ONLY", "DO_NOT_ENTER", "UNKNOWN"]
DecisionBrainRecommendation = Literal[
    "ENTER_NOW_SHADOW",
    "WAIT_CONFIRMATION_SHADOW",
    "MANUAL_ONLY_SHADOW",
    "DO_NOT_ENTER_SHADOW",
    "INSUFFICIENT_DATA",
    "MODEL_ERROR",
]


@dataclass
class DecisionBrainShadowOutput:
    mode: Literal["SHADOW"] = "SHADOW"
    execute: bool = False
    production_allowed: bool = False
    status: str = "RESEARCH_CANDIDATE_NOT_LIVE"
    model_version: str = "v010"
    symbol: str = ""
    side: str | None = None
    decision: DecisionBrainDecision = "UNKNOWN"
    enter_now_prob: float = 0.0
    wait_confirmation_prob: float = 0.0
    manual_only_prob: float = 0.0
    do_not_enter_prob: float = 0.0
    recommendation: DecisionBrainRecommendation = "INSUFFICIENT_DATA"
    reason: str = "decision_brain_insufficient_features"
    feature_status: Literal["ok", "partial", "insufficient"] = "insufficient"
    feature_parity_pct: float = 0.0
    missing_features_count: int = 0
    missing_features: list[str] = field(default_factory=list)
    critical_missing_groups: list[str] = field(default_factory=list)
    available_feature_groups: list[str] = field(default_factory=list)
    approximated_features: list[str] = field(default_factory=list)
    missing_features_by_group: dict[str, list[str]] = field(default_factory=dict)
    feature_group_coverage_pct: dict[str, float] = field(default_factory=dict)
    feature_warnings: list[str] = field(default_factory=list)
    feature_build_latency_ms: float = 0.0
    model_latency_ms: float = 0.0
    total_latency_ms: float = 0.0
    latency_ms: float = 0.0


def model_dump(payload: Any) -> dict[str, Any]:
    if is_dataclass(payload):
        return asdict(payload)
    if hasattr(payload, "model_dump"):
        return payload.model_dump()
    if hasattr(payload, "dict"):
        return payload.dict()
    return dict(payload or {})


def base_shadow_output(symbol: str, side: str | None, model_version: str = "v010") -> DecisionBrainShadowOutput:
    return DecisionBrainShadowOutput(symbol=symbol, side=side, model_version=model_version)
