from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class PredictRequest(BaseModel):
    symbol: str = Field(default="ETHUSDT")


class Probabilities(BaseModel):
    idle: float
    long: float
    short: float
    close: float = 0.0


class SignalQuality(BaseModel):
    has_conviction: bool
    reason: str


class RiskContext(BaseModel):
    leverage: float
    position_fraction: float
    recommended_risk: float


class RegimePayload(BaseModel):
    type: str
    confidence: float


class AegisPredictResponse(BaseModel):
    model_name: str
    model_version: str
    symbol: str
    raw_action: str
    gated_action: str
    probs: Probabilities
    top_prob: float
    long_short_gap: float
    signal_quality: SignalQuality
    risk_context: RiskContext
    regime: RegimePayload
    features: dict[str, float] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)


class LegacyMlV2Response(BaseModel):
    symbol: str
    long_prob: float
    short_prob: float
    neutral_prob: float
    close_prob: float = 0.0
    consensus_level: float
    meta_verdict: str
    smart_leverage: float
    features: dict[str, float]
    aegis: AegisPredictResponse


class ExitSignalRequest(BaseModel):
    symbol: str
    entry_price: float
    current_pnl: float
    mfe: float = 0.0
    mae: float = 0.0
    duration_minutes: float = 0.0
    leverage: float = 5.0


class ExitSignalResponse(BaseModel):
    action: str
    confidence: float
    reason: str = "aegis_defensive_default"
