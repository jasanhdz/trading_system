"""Immutable domain contracts shared by scientific inference and training."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Mapping, Sequence


class DomainValidationError(ValueError):
    pass


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
    ERROR_DATA_INVALID = "ERROR_DATA_INVALID"


class ScientificLayerName(str, Enum):
    D3 = "D3"
    RV2 = "RV2"
    TRRM = "TRRM"
    QMAE = "QMAE"
    EQM = "EQM"
    ECON1 = "ECON1"


class Regime(str, Enum):
    BULL_TREND = "BULL_TREND"
    BEAR_TREND = "BEAR_TREND"
    RANGE = "RANGE"
    HIGH_VOLATILITY = "HIGH_VOLATILITY"
    TRANSITION = "TRANSITION"
    UNKNOWN = "UNKNOWN"


class ReasonCode(str, Enum):
    ELIGIBLE = "ELIGIBLE"
    NO_TRADE_NO_CANDIDATE = "NO_TRADE_NO_CANDIDATE"
    NO_TRADE_THRESHOLD = "NO_TRADE_THRESHOLD"
    NO_TRADE_REGIME = "NO_TRADE_REGIME"
    NO_TRADE_UNCERTAINTY = "NO_TRADE_UNCERTAINTY"
    NO_TRADE_PORTFOLIO_CONFLICT = "NO_TRADE_PORTFOLIO_CONFLICT"
    NO_TRADE_DATA_QUALITY = "NO_TRADE_DATA_QUALITY"
    NO_TRADE_STALE = "NO_TRADE_STALE"
    NO_TRADE_CONFIG_MISMATCH = "NO_TRADE_CONFIG_MISMATCH"
    MODEL_NO_DIRECTION = "MODEL_NO_DIRECTION"
    TRRM_TAIL_RISK_VETO = "TRRM_TAIL_RISK_VETO"
    QMAE_ADVERSE_EXCURSION_HIGH = "QMAE_ADVERSE_EXCURSION_HIGH"
    EQM_QUALITY_LOW = "EQM_QUALITY_LOW"
    ECON1_EDGE_BELOW_COST = "ECON1_EDGE_BELOW_COST"
    SYMBOL_BLOCKED = "SYMBOL_BLOCKED"
    SYMBOL_OCCUPIED = "SYMBOL_OCCUPIED"
    ACTIVE_COOLDOWN = "ACTIVE_COOLDOWN"
    NO_AVAILABLE_SLOT = "NO_AVAILABLE_SLOT"
    INPUT_INVALID = "INPUT_INVALID"
    MODEL_UNAVAILABLE = "MODEL_UNAVAILABLE"
    INFERENCE_FAILURE = "INFERENCE_FAILURE"


class OutcomeExecutionStatus(str, Enum):
    NOT_EXECUTED = "NOT_EXECUTED"
    FILLED = "FILLED"
    PARTIALLY_FILLED = "PARTIALLY_FILLED"
    REJECTED = "REJECTED"
    INCIDENT = "INCIDENT"


class EvidenceMode(str, Enum):
    OPERATIONAL = "OPERATIONAL"
    PAPER = "PAPER"
    SHADOW = "SHADOW"
    REPLAY = "REPLAY"


def _aware_utc(value: datetime, field_name: str) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise DomainValidationError(f"{field_name} must be timezone-aware")
    return value.astimezone(timezone.utc)


def _finite(value: float, field_name: str) -> float:
    numeric = float(value)
    if not math.isfinite(numeric):
        raise DomainValidationError(f"{field_name} must be finite")
    return numeric


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

    def __post_init__(self) -> None:
        object.__setattr__(self, "open_time", _aware_utc(self.open_time, "open_time"))
        object.__setattr__(self, "close_time", _aware_utc(self.close_time, "close_time"))
        if self.close_time <= self.open_time:
            raise DomainValidationError("close_time must be after open_time")
        for name in ("open", "high", "low", "close", "volume"):
            object.__setattr__(self, name, _finite(getattr(self, name), name))
        if min(self.open, self.high, self.low, self.close) <= 0:
            raise DomainValidationError("OHLC values must be positive")
        if self.volume < 0:
            raise DomainValidationError("volume cannot be negative")
        if not self.source:
            raise DomainValidationError("source is required")


@dataclass(frozen=True)
class FeedQuality:
    missing_bars: int = 0
    duplicate_bars: int = 0
    source_lag_ms: int = 0

    def __post_init__(self) -> None:
        if min(self.missing_bars, self.duplicate_bars, self.source_lag_ms) < 0:
            raise DomainValidationError("feed quality counters cannot be negative")


@dataclass(frozen=True)
class SymbolSeries:
    symbol: str
    candles: tuple[Candle, ...]
    last_confirmed_close: datetime
    feed_quality: FeedQuality = field(default_factory=FeedQuality)

    def __post_init__(self) -> None:
        if not self.symbol:
            raise DomainValidationError("symbol is required")
        object.__setattr__(self, "last_confirmed_close", _aware_utc(self.last_confirmed_close, "last_confirmed_close"))


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

    def __post_init__(self) -> None:
        if min(self.available_slots, self.long_exposure_count, self.short_exposure_count) < 0:
            raise DomainValidationError("portfolio counters cannot be negative")
        if len(set(self.blocked_symbols)) != len(self.blocked_symbols):
            raise DomainValidationError("blocked_symbols contains duplicates")
        if len(set(self.occupied_symbols)) != len(self.occupied_symbols):
            raise DomainValidationError("occupied_symbols contains duplicates")
        normalized = {symbol: _aware_utc(ts, f"cooldown:{symbol}") for symbol, ts in self.active_cooldowns.items()}
        object.__setattr__(self, "active_cooldowns", normalized)
        if self.operational_time is not None:
            object.__setattr__(self, "operational_time", _aware_utc(self.operational_time, "operational_time"))


@dataclass(frozen=True)
class MarketSnapshot:
    closed_at: datetime
    timeframe: str
    symbol_set_hash: str
    series: tuple[SymbolSeries, ...]
    portfolio: PortfolioContext

    def __post_init__(self) -> None:
        object.__setattr__(self, "closed_at", _aware_utc(self.closed_at, "closed_at"))
        if not self.timeframe or not self.symbol_set_hash:
            raise DomainValidationError("timeframe and symbol_set_hash are required")


@dataclass(frozen=True)
class DecisionRequest:
    request_id: str
    decision_cycle_id: str
    schema_version: str
    contract_version: str
    config_version: str
    snapshot: MarketSnapshot

    def __post_init__(self) -> None:
        if not all((self.request_id, self.decision_cycle_id, self.schema_version, self.contract_version, self.config_version)):
            raise DomainValidationError("decision request identifiers and versions are required")


@dataclass(frozen=True)
class FeatureQuality:
    missing_values: int
    clipped_values: int
    finite: bool
    history_rows: int

    def __post_init__(self) -> None:
        if min(self.missing_values, self.clipped_values, self.history_rows) < 0:
            raise DomainValidationError("feature quality counters cannot be negative")


@dataclass(frozen=True)
class FeatureRow:
    symbol: str
    raw_values: tuple[float, ...]
    normalized_values: tuple[float, ...]
    quality: FeatureQuality


@dataclass(frozen=True)
class FeatureBatch:
    schema_version: str
    feature_names: tuple[str, ...]
    feature_hash: str
    rows: tuple[FeatureRow, ...]

    def __post_init__(self) -> None:
        expected = len(self.feature_names)
        if not expected or len(set(self.feature_names)) != expected:
            raise DomainValidationError("feature names must be non-empty and unique")
        if len({row.symbol for row in self.rows}) != len(self.rows):
            raise DomainValidationError("feature rows contain duplicate symbols")
        for row in self.rows:
            if len(row.raw_values) != expected or len(row.normalized_values) != expected:
                raise DomainValidationError(f"feature dimension mismatch for {row.symbol}")
            if not all(math.isfinite(value) for value in (*row.raw_values, *row.normalized_values)):
                raise DomainValidationError(f"non-finite feature for {row.symbol}")

    def row_for(self, symbol: str) -> FeatureRow:
        for row in self.rows:
            if row.symbol == symbol:
                return row
        raise KeyError(symbol)


@dataclass(frozen=True)
class ModelPrediction:
    model_id: str
    symbol: str
    horizon_bars: int
    side: TradeSide
    long_probability: float
    short_probability: float
    neutral_probability: float
    expected_return: float
    tail_risk_probability: float
    qmae_q90: float
    quality_probability: float
    uncertainty: float

    def __post_init__(self) -> None:
        for name in (
            "long_probability",
            "short_probability",
            "neutral_probability",
            "tail_risk_probability",
            "quality_probability",
            "uncertainty",
        ):
            value = _finite(getattr(self, name), name)
            if not 0.0 <= value <= 1.0:
                raise DomainValidationError(f"{name} must be within [0, 1]")
        _finite(self.expected_return, "expected_return")
        if _finite(self.qmae_q90, "qmae_q90") < 0:
            raise DomainValidationError("qmae_q90 cannot be negative")
        probability_sum = self.long_probability + self.short_probability + self.neutral_probability
        if not math.isclose(probability_sum, 1.0, rel_tol=1e-9, abs_tol=1e-9):
            raise DomainValidationError("direction probabilities must sum to one")


@dataclass(frozen=True)
class ModelPredictions:
    bundle_id: str
    feature_hash: str
    predictions: tuple[ModelPrediction, ...]

    def for_symbol(self, symbol: str) -> tuple[ModelPrediction, ...]:
        return tuple(prediction for prediction in self.predictions if prediction.symbol == symbol)


@dataclass(frozen=True)
class ScientificContext:
    request_id: str
    decision_cycle_id: str
    closed_at: datetime
    timeframe: str
    portfolio: PortfolioContext
    features: FeatureBatch


@dataclass(frozen=True)
class LayerResult:
    symbol: str
    side: TradeSide
    regime: Regime
    d3_confidence: float
    rv2_tail_risk: float
    trrm_compatibility: float
    qmae_q90: float
    qmae_quality: float
    eqm_score: float
    model_disagreement: float
    econ_edge: float
    calibrated_score: float
    eligible: bool
    reason_codes: tuple[ReasonCode, ...]
    diagnostics: tuple[tuple[str, float | str | bool], ...] = ()

    def __post_init__(self) -> None:
        bounded = ("d3_confidence", "rv2_tail_risk", "trrm_compatibility", "qmae_quality",
                   "model_disagreement", "calibrated_score")
        for name in bounded:
            value = _finite(getattr(self, name), name)
            if not 0.0 <= value <= 1.0:
                raise DomainValidationError(f"{name} must be within [0, 1]")
        if _finite(self.qmae_q90, "qmae_q90") < 0:
            raise DomainValidationError("qmae_q90 cannot be negative")
        _finite(self.eqm_score, "eqm_score")
        _finite(self.econ_edge, "econ_edge")


@dataclass(frozen=True)
class LayerOutputs:
    ordered_layers: tuple[ScientificLayerName, ...]
    results: tuple[LayerResult, ...]
    warnings: tuple[str, ...] = ()

    def for_symbol(self, symbol: str) -> LayerResult:
        for result in self.results:
            if result.symbol == symbol:
                return result
        raise KeyError(symbol)


@dataclass(frozen=True)
class RiskIntent:
    stop_distance_fraction: float | None = None
    target_distance_fraction: float | None = None
    volatility_multiple: float | None = None
    target_risk_ratio: float | None = None
    maximum_holding_bars: int | None = None
    scientific_invalidation: str | None = None
    relative_priority: float | None = None

    def __post_init__(self) -> None:
        for name in ("stop_distance_fraction", "target_distance_fraction", "volatility_multiple",
                     "target_risk_ratio", "relative_priority"):
            value = getattr(self, name)
            if value is not None and _finite(value, name) < 0:
                raise DomainValidationError(f"{name} cannot be negative")
        if self.maximum_holding_bars is not None and self.maximum_holding_bars <= 0:
            raise DomainValidationError("maximum_holding_bars must be positive")


@dataclass(frozen=True)
class Candidate:
    candidate_id: str
    symbol: str
    side: TradeSide
    raw_score: float
    calibrated_score: float
    confidence: float
    uncertainty: float
    regime: Regime
    compatibility: float
    expected_return: float
    horizon_bars: int
    risk_intent: RiskIntent
    reason_codes: tuple[ReasonCode, ...]
    evidence_references: tuple[str, ...]
    model_bundle_id: str
    feature_hash: str
    candidate_hash: str
    eligible: bool

    def __post_init__(self) -> None:
        for name in ("raw_score", "calibrated_score", "confidence", "uncertainty", "compatibility"):
            value = _finite(getattr(self, name), name)
            if not 0.0 <= value <= 1.0:
                raise DomainValidationError(f"{name} must be within [0, 1]")
        _finite(self.expected_return, "expected_return")
        if self.horizon_bars <= 0:
            raise DomainValidationError("horizon_bars must be positive")
        if not all((self.candidate_id, self.symbol, self.model_bundle_id, self.feature_hash, self.candidate_hash)):
            raise DomainValidationError("candidate identifiers and hashes are required")


@dataclass(frozen=True)
class RankedCandidate:
    rank: int
    symbol: str
    candidate_hash: str
    score: float
    eligible: bool
    reason_codes: tuple[ReasonCode, ...]


@dataclass(frozen=True)
class CandidateSet:
    decision_cycle_id: str
    candidates: tuple[Candidate, ...]


@dataclass(frozen=True)
class SelectionResult:
    status: DecisionStatus
    selected: tuple[Candidate, ...]
    ranking: tuple[RankedCandidate, ...]
    reason_codes: tuple[ReasonCode, ...]


@dataclass(frozen=True)
class FrozenDecision:
    decision_id: str
    decision_cycle_id: str
    generated_at: datetime
    expires_at: datetime
    status: DecisionStatus
    selected: tuple[Candidate, ...]
    ranking: tuple[RankedCandidate, ...]
    reason_codes: tuple[ReasonCode, ...]
    model_bundle_id: str
    feature_hash: str
    config_hash: str
    evidence_hash: str
    decision_hash: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "generated_at", _aware_utc(self.generated_at, "generated_at"))
        object.__setattr__(self, "expires_at", _aware_utc(self.expires_at, "expires_at"))
        if self.expires_at <= self.generated_at:
            raise DomainValidationError("frozen decision expiry must follow generation")


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
    ranking: tuple[RankedCandidate, ...]
    reason_codes: tuple[ReasonCode, ...]
    warnings: tuple[str, ...] = ()


@dataclass(frozen=True)
class BrainManifest:
    contract_version: str
    universe_id: str
    symbols: tuple[str, ...]
    symbol_set_hash: str
    timeframe: str
    config_version: str
    config_hash: str
    model_bundle_id: str
    feature_schema_version: str
    feature_hash: str
    capabilities: tuple[str, ...]
    build_id: str
    ready: bool


@dataclass(frozen=True)
class FillOutcome:
    status: OutcomeExecutionStatus
    filled_quantity: float | None = None
    average_entry_price: float | None = None
    filled_at: datetime | None = None

    def __post_init__(self) -> None:
        if self.filled_quantity is not None and _finite(self.filled_quantity, "filled_quantity") < 0:
            raise DomainValidationError("filled_quantity cannot be negative")
        if self.average_entry_price is not None and _finite(self.average_entry_price, "average_entry_price") <= 0:
            raise DomainValidationError("average_entry_price must be positive")
        if self.filled_at is not None:
            object.__setattr__(self, "filled_at", _aware_utc(self.filled_at, "filled_at"))


@dataclass(frozen=True)
class DecisionOutcome:
    decision_id: str
    decision_cycle_id: str
    candidate_hash: str | None
    accepted: bool
    executed: bool
    rejection_reason: str | None
    fill: FillOutcome
    closed_at: datetime | None
    realized_pnl: float | None
    close_reason: str | None
    incidents: tuple[str, ...]
    reconciled: bool
    occurred_at: datetime
    execution_mode: EvidenceMode = EvidenceMode.OPERATIONAL
    hypothetical_details: Mapping[str, float | str | bool] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "occurred_at", _aware_utc(self.occurred_at, "occurred_at"))
        if self.closed_at is not None:
            object.__setattr__(self, "closed_at", _aware_utc(self.closed_at, "closed_at"))
        if self.realized_pnl is not None:
            _finite(self.realized_pnl, "realized_pnl")
        if self.executed and not self.accepted:
            raise DomainValidationError("an executed outcome must have been accepted")
        if self.execution_mode in (EvidenceMode.SHADOW, EvidenceMode.REPLAY) and self.executed:
            raise DomainValidationError("shadow/replay outcomes cannot be marked executed")
        for key, value in self.hypothetical_details.items():
            if isinstance(value, float):
                _finite(value, f"hypothetical_details:{key}")


@dataclass(frozen=True)
class ScientificEvidenceEvent:
    event_id: str
    decision_id: str
    decision_cycle_id: str
    event_type: str
    occurred_at: datetime
    payload: Mapping[str, Any]
    previous_event_hash: str | None = None
    event_hash: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "occurred_at", _aware_utc(self.occurred_at, "occurred_at"))


def _parse_datetime(value: str | datetime, field_name: str) -> datetime:
    if isinstance(value, datetime):
        return _aware_utc(value, field_name)
    parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    return _aware_utc(parsed, field_name)


def decision_request_from_dict(payload: Mapping[str, Any]) -> DecisionRequest:
    snapshot_data = payload["snapshot"]
    portfolio_data = snapshot_data.get("portfolio", {})
    portfolio = PortfolioContext(
        blocked_symbols=tuple(portfolio_data.get("blocked_symbols", ())),
        occupied_symbols=tuple(portfolio_data.get("occupied_symbols", ())),
        available_slots=int(portfolio_data.get("available_slots", 0)),
        long_exposure_count=int(portfolio_data.get("long_exposure_count", 0)),
        short_exposure_count=int(portfolio_data.get("short_exposure_count", 0)),
        active_cooldowns={key: _parse_datetime(value, f"cooldown:{key}") for key, value in portfolio_data.get("active_cooldowns", {}).items()},
        accepted_decision_ids=tuple(portfolio_data.get("accepted_decision_ids", ())),
        operational_time=_parse_datetime(portfolio_data["operational_time"], "operational_time") if portfolio_data.get("operational_time") else None,
    )
    series = []
    for item in snapshot_data["series"]:
        candles = tuple(
            Candle(
                open_time=_parse_datetime(candle["open_time"], "open_time"),
                close_time=_parse_datetime(candle["close_time"], "close_time"),
                open=float(candle["open"]),
                high=float(candle["high"]),
                low=float(candle["low"]),
                close=float(candle["close"]),
                volume=float(candle["volume"]),
                is_closed=bool(candle["is_closed"]),
                source=str(candle["source"]),
                sequence=str(candle["sequence"]) if candle.get("sequence") is not None else None,
            )
            for candle in item["candles"]
        )
        quality = item.get("feed_quality", {})
        series.append(
            SymbolSeries(
                symbol=str(item["symbol"]),
                candles=candles,
                last_confirmed_close=_parse_datetime(item["last_confirmed_close"], "last_confirmed_close"),
                feed_quality=FeedQuality(
                    missing_bars=int(quality.get("missing_bars", 0)),
                    duplicate_bars=int(quality.get("duplicate_bars", 0)),
                    source_lag_ms=int(quality.get("source_lag_ms", 0)),
                ),
            )
        )
    return DecisionRequest(
        request_id=str(payload["request_id"]),
        decision_cycle_id=str(payload["decision_cycle_id"]),
        schema_version=str(payload["schema_version"]),
        contract_version=str(payload["contract_version"]),
        config_version=str(payload["config_version"]),
        snapshot=MarketSnapshot(
            closed_at=_parse_datetime(snapshot_data["closed_at"], "closed_at"),
            timeframe=str(snapshot_data["timeframe"]),
            symbol_set_hash=str(snapshot_data["symbol_set_hash"]),
            series=tuple(series),
            portfolio=portfolio,
        ),
    )


def decision_outcome_from_dict(payload: Mapping[str, Any]) -> DecisionOutcome:
    fill_data = payload.get("fill", {})
    return DecisionOutcome(
        decision_id=str(payload["decision_id"]),
        decision_cycle_id=str(payload["decision_cycle_id"]),
        candidate_hash=str(payload["candidate_hash"]) if payload.get("candidate_hash") is not None else None,
        accepted=bool(payload["accepted"]),
        executed=bool(payload["executed"]),
        rejection_reason=str(payload["rejection_reason"]) if payload.get("rejection_reason") is not None else None,
        fill=FillOutcome(
            status=OutcomeExecutionStatus(fill_data.get("status", "NOT_EXECUTED")),
            filled_quantity=float(fill_data["filled_quantity"]) if fill_data.get("filled_quantity") is not None else None,
            average_entry_price=float(fill_data["average_entry_price"]) if fill_data.get("average_entry_price") is not None else None,
            filled_at=_parse_datetime(fill_data["filled_at"], "filled_at") if fill_data.get("filled_at") else None,
        ),
        closed_at=_parse_datetime(payload["closed_at"], "closed_at") if payload.get("closed_at") else None,
        realized_pnl=float(payload["realized_pnl"]) if payload.get("realized_pnl") is not None else None,
        close_reason=str(payload["close_reason"]) if payload.get("close_reason") is not None else None,
        incidents=tuple(str(item) for item in payload.get("incidents", ())),
        reconciled=bool(payload.get("reconciled", False)),
        occurred_at=_parse_datetime(payload["occurred_at"], "occurred_at"),
        execution_mode=EvidenceMode(payload.get("execution_mode", "OPERATIONAL")),
        hypothetical_details={str(key): value for key, value in payload.get("hypothetical_details", {}).items()},
    )
