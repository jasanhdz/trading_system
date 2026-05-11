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
        model_scope=model_scope,
        latency_ms=round(float(latency_ms), 3),
    ).to_dict()
