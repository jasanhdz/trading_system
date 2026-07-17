"""Shared domain types for the scientific decision flow."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Mapping


class TradeSide(str, Enum):
    LONG = "LONG"
    SHORT = "SHORT"
    NO_TRADE = "NO_TRADE"


class DecisionStatus(str, Enum):
    SELECTED = "SELECTED"
    NO_TRADE = "NO_TRADE"
    ERROR = "ERROR"


class ValidationStatus(str, Enum):
    VALID = "VALID"
    NO_TRADE_DATA_INSUFFICIENT = "NO_TRADE_DATA_INSUFFICIENT"
    NO_TRADE_DATA_STALE = "NO_TRADE_DATA_STALE"
    NO_TRADE_UNIVERSE_MISMATCH = "NO_TRADE_UNIVERSE_MISMATCH"
    ERROR_CONTRACT = "ERROR_CONTRACT"
    ERROR_MODEL_BUNDLE = "ERROR_MODEL_BUNDLE"


class ScientificLayerName(str, Enum):
    D3 = "D3"
    RV2 = "RV2"
    TRRM = "TRRM"
    QMAE = "QMAE"
    EQM = "EQM"
    ECON1 = "ECON1"


@dataclass(frozen=True)
class Candle:
    open_time: datetime
    close_time: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float
    is_closed: bool
    source: str
    sequence: str | None = None


@dataclass(frozen=True)
class SymbolSeries:
    symbol: str
    candles: tuple[Candle, ...]
    last_confirmed_close: datetime
    feed_quality: Mapping[str, str | float | int | bool] = field(default_factory=dict)


@dataclass(frozen=True)
class PortfolioContext:
    blocked_symbols: tuple[str, ...] = ()
    occupied_symbols: tuple[str, ...] = ()
    available_slots: int = 0
    long_exposure_count: int = 0
    short_exposure_count: int = 0
    active_cooldowns: Mapping[str, datetime] = field(default_factory=dict)
    accepted_decision_ids: tuple[str, ...] = ()
    operational_time: datetime | None = None


@dataclass(frozen=True)
class MarketSnapshot:
    closed_at: datetime
    timeframe: str
    symbol_set_hash: str
    series: tuple[SymbolSeries, ...]
    portfolio: PortfolioContext


@dataclass(frozen=True)
class DecisionRequest:
    request_id: str
    decision_cycle_id: str
    schema_version: str
    contract_version: str
    config_version: str
    snapshot: MarketSnapshot


@dataclass(frozen=True)
class FeatureBatch:
    schema_version: str
    feature_names: tuple[str, ...]
    feature_hash: str
    row_count: int
    values_by_symbol: Mapping[str, tuple[tuple[float, ...], ...]]


@dataclass(frozen=True)
class ModelPrediction:
    model_id: str
    symbol: str
    horizon: str
    direction: TradeSide
    score: float
    confidence: float
    uncertainty: float
    expected_return: float | None = None


@dataclass(frozen=True)
class ModelPredictions:
    bundle_id: str
    feature_hash: str
    predictions: tuple[ModelPrediction, ...]


@dataclass(frozen=True)
class ScientificContext:
    request_id: str
    decision_cycle_id: str
    closed_at: datetime
    timeframe: str
    portfolio: PortfolioContext


@dataclass(frozen=True)
class LayerOutputs:
    ordered_layers: tuple[ScientificLayerName, ...]
    values_by_symbol: Mapping[str, Mapping[str, str | float | int | bool]]
    warnings: tuple[str, ...] = ()


@dataclass(frozen=True)
class RiskIntent:
    stop_distance_fraction: float | None = None
    volatility_multiple: float | None = None
    target_risk_ratio: float | None = None
    maximum_holding_bars: int | None = None
    scientific_invalidation: str | None = None
    relative_priority: float | None = None


@dataclass(frozen=True)
class Candidate:
    candidate_id: str
    symbol: str
    side: TradeSide
    raw_score: float
    calibrated_score: float
    confidence: float
    uncertainty: float
    regime: str
    compatibility: float
    expected_return: float | None
    horizon: str
    risk_intent: RiskIntent
    positive_reasons: tuple[str, ...] = ()
    rejection_reasons: tuple[str, ...] = ()
    candidate_hash: str = ""


@dataclass(frozen=True)
class CandidateSet:
    decision_cycle_id: str
    candidates: tuple[Candidate, ...]


@dataclass(frozen=True)
class FrozenDecision:
    decision_id: str
    decision_cycle_id: str
    generated_at: datetime
    expires_at: datetime
    status: DecisionStatus
    selected: tuple[Candidate, ...]
    ranked_candidates: tuple[Candidate, ...]
    evidence_hash: str
    decision_hash: str


@dataclass(frozen=True)
class DecisionResponse:
    contract_version: str
    decision_id: str
    decision_cycle_id: str
    generated_at: datetime
    expires_at: datetime
    status: DecisionStatus
    universe_id: str
    symbol_set_hash: str
    config_version: str
    model_bundle_id: str
    feature_schema_version: str
    evidence_hash: str
    selected: tuple[Candidate, ...]
    ranking_summary: tuple[str, ...]
    warnings: tuple[str, ...] = ()


@dataclass(frozen=True)
class BrainManifest:
    contract_version: str
    universe_id: str
    symbols: tuple[str, ...]
    symbol_set_hash: str
    timeframe: str
    config_version: str
    model_bundle_id: str
    feature_schema_version: str
    capabilities: tuple[str, ...]
    build_id: str
    ready: bool


@dataclass(frozen=True)
class DecisionOutcome:
    decision_id: str
    decision_cycle_id: str
    candidate_hash: str | None
    accepted: bool
    executed: bool
    reason_codes: tuple[str, ...]
    occurred_at: datetime
    normalized_details: Mapping[str, str | float | int | bool | None] = field(default_factory=dict)


@dataclass(frozen=True)
class ScientificEvidenceEvent:
    event_id: str
    decision_id: str
    decision_cycle_id: str
    event_type: str
    occurred_at: datetime
    payload: Mapping[str, object]
    previous_event_hash: str | None = None
