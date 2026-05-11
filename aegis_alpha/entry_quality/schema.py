from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal


MODE = "SHADOW"
STATUS = "RESEARCH_CANDIDATE_NOT_LIVE"
MODEL_VERSION = "v020"
QUALITY_MIN = 0.60
TAIL_MAX = 0.50

Recommendation = Literal["ALLOW_SHADOW", "BLOCK_SHADOW", "INSUFFICIENT_DATA", "MODEL_ERROR"]
FeatureStatus = Literal["ok", "partial", "insufficient"]
ModelScope = Literal["symbol", "global", "none"]


@dataclass(frozen=True)
class EntryQualityShadowResult:
    symbol: str
    model_version: str = MODEL_VERSION
    entry_quality_score: float | None = None
    tail_risk_score: float | None = None
    recommendation: Recommendation = "INSUFFICIENT_DATA"
    reason: str = "insufficient_entry_quality_features"
    feature_status: FeatureStatus = "insufficient"
    missing_features: list[str] = field(default_factory=list)
    feature_parity_pct: float | None = None
    missing_features_count: int | None = None
    approximated_features: list[str] = field(default_factory=list)
    critical_missing_groups: list[str] = field(default_factory=list)
    feature_build_latency_ms: float | None = None
    model_latency_ms: float | None = None
    total_latency_ms: float | None = None
    model_scope: ModelScope = "none"
    latency_ms: float = 0.0
    mode: str = MODE
    execute: bool = False
    production_allowed: bool = False
    status: str = STATUS
    thresholds: dict[str, float] = field(default_factory=lambda: {"quality_min": QUALITY_MIN, "tail_max": TAIL_MAX})

    def to_dict(self) -> dict[str, Any]:
        return {
            "mode": MODE,
            "execute": False,
            "production_allowed": False,
            "status": STATUS,
            "symbol": self.symbol,
            "model_version": self.model_version,
            "entry_quality_score": self.entry_quality_score,
            "tail_risk_score": self.tail_risk_score,
            "recommendation": self.recommendation,
            "reason": self.reason,
            "thresholds": dict(self.thresholds),
            "feature_status": self.feature_status,
            "missing_features": list(self.missing_features),
            "feature_parity_pct": self.feature_parity_pct,
            "missing_features_count": self.missing_features_count if self.missing_features_count is not None else len(self.missing_features),
            "approximated_features": list(self.approximated_features),
            "critical_missing_groups": list(self.critical_missing_groups),
            "feature_build_latency_ms": self.feature_build_latency_ms,
            "model_latency_ms": self.model_latency_ms,
            "total_latency_ms": self.total_latency_ms if self.total_latency_ms is not None else self.latency_ms,
            "model_scope": self.model_scope,
            "latency_ms": self.latency_ms,
        }


def shadow_result(
    *,
    symbol: str,
    entry_quality_score: float | None = None,
    tail_risk_score: float | None = None,
    recommendation: Recommendation = "INSUFFICIENT_DATA",
    reason: str = "insufficient_entry_quality_features",
    feature_status: FeatureStatus = "insufficient",
    missing_features: list[str] | None = None,
    feature_parity_pct: float | None = None,
    missing_features_count: int | None = None,
    approximated_features: list[str] | None = None,
    critical_missing_groups: list[str] | None = None,
    feature_build_latency_ms: float | None = None,
    model_latency_ms: float | None = None,
    total_latency_ms: float | None = None,
    model_scope: ModelScope = "none",
    latency_ms: float = 0.0,
    model_version: str = MODEL_VERSION,
) -> dict[str, Any]:
    return EntryQualityShadowResult(
        symbol=symbol,
        model_version=model_version,
        entry_quality_score=entry_quality_score,
        tail_risk_score=tail_risk_score,
        recommendation=recommendation,
        reason=reason,
        feature_status=feature_status,
        missing_features=missing_features or [],
        feature_parity_pct=round(float(feature_parity_pct), 2) if feature_parity_pct is not None else None,
        missing_features_count=missing_features_count,
        approximated_features=approximated_features or [],
        critical_missing_groups=critical_missing_groups or [],
        feature_build_latency_ms=round(float(feature_build_latency_ms), 3) if feature_build_latency_ms is not None else None,
        model_latency_ms=round(float(model_latency_ms), 3) if model_latency_ms is not None else None,
        total_latency_ms=round(float(total_latency_ms), 3) if total_latency_ms is not None else None,
        model_scope=model_scope,
        latency_ms=round(float(latency_ms), 3),
    ).to_dict()
