"""Frozen domain objects for causal snapshots and feature ownership."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Iterable, Mapping

from aegis_strategy_router.domain.serialization import canonical_json_bytes, content_hash, frozen_pairs, utc_datetime


class DataStatus(str, Enum):
    AVAILABLE = "AVAILABLE"
    UNKNOWN = "UNKNOWN"
    INVALID = "INVALID"


class Side(str, Enum):
    LONG = "LONG"
    SHORT = "SHORT"


class Timeframe(str, Enum):
    M1 = "1m"
    M5 = "5m"
    M15 = "15m"
    H1 = "1h"
    H4 = "4h"
    D1 = "1d"

    @property
    def minutes(self) -> int:
        return {
            Timeframe.M1: 1,
            Timeframe.M5: 5,
            Timeframe.M15: 15,
            Timeframe.H1: 60,
            Timeframe.H4: 240,
            Timeframe.D1: 1_440,
        }[self]

    @property
    def duration(self) -> timedelta:
        return timedelta(minutes=self.minutes)

    @property
    def structural_lookback(self) -> int | None:
        return {
            Timeframe.M1: None,
            Timeframe.M5: None,
            Timeframe.M15: 96,
            Timeframe.H1: 120,
            Timeframe.H4: 90,
            Timeframe.D1: 60,
        }[self]


Scalar = float | int | str | bool


@dataclass(frozen=True, slots=True)
class Candle:
    open_at: datetime
    close_at: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float
    taker_buy_volume: float | None
    available_at: datetime
    source_id: str
    complete: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(self, "open_at", utc_datetime(self.open_at))
        object.__setattr__(self, "close_at", utc_datetime(self.close_at))
        object.__setattr__(self, "available_at", utc_datetime(self.available_at))
        prices = (self.open, self.high, self.low, self.close)
        if self.open_at >= self.close_at:
            raise ValueError("candle open_at must precede close_at")
        if self.available_at < self.close_at:
            raise ValueError("closed candle cannot be available before close_at")
        if not all(math.isfinite(value) and value > 0 for value in prices):
            raise ValueError("OHLC prices must be finite and positive")
        if self.high < max(self.open, self.close) or self.low > min(self.open, self.close):
            raise ValueError("OHLC bounds are incoherent")
        if self.low > self.high or not math.isfinite(self.volume) or self.volume < 0:
            raise ValueError("candle range/volume is invalid")
        if self.taker_buy_volume is not None and (
            not math.isfinite(self.taker_buy_volume) or self.taker_buy_volume < 0
        ):
            raise ValueError("taker_buy_volume is invalid")

    def to_primitive(self) -> dict[str, Any]:
        return {
            "open_at": self.open_at,
            "close_at": self.close_at,
            "open": self.open,
            "high": self.high,
            "low": self.low,
            "close": self.close,
            "volume": self.volume,
            "taker_buy_volume": self.taker_buy_volume,
            "available_at": self.available_at,
            "source_id": self.source_id,
            "complete": self.complete,
        }


@dataclass(frozen=True, slots=True)
class FeatureObservation:
    name: str
    value: Scalar | None
    observed_at: datetime | None
    available_at: datetime | None
    owner: str
    source_version: str
    status: DataStatus
    reason: str | None = None

    def __post_init__(self) -> None:
        if not self.name or not self.owner or not self.source_version:
            raise ValueError("feature identity fields cannot be empty")
        if self.observed_at is not None:
            object.__setattr__(self, "observed_at", utc_datetime(self.observed_at))
        if self.available_at is not None:
            object.__setattr__(self, "available_at", utc_datetime(self.available_at))
        if self.status is DataStatus.AVAILABLE:
            if self.value is None or self.observed_at is None or self.available_at is None:
                raise ValueError("available feature requires value and timestamps")
            if isinstance(self.value, float) and not math.isfinite(self.value):
                raise ValueError("available feature value must be finite")
            if self.available_at < self.observed_at:
                raise ValueError("feature cannot be available before it was observed")
        elif self.value is not None:
            raise ValueError("unknown/invalid features cannot expose a value")
        if self.status is not DataStatus.AVAILABLE and not self.reason:
            raise ValueError("unknown/invalid feature requires a reason")

    def assert_available_by(self, cutoff: datetime) -> None:
        boundary = utc_datetime(cutoff)
        if self.status is DataStatus.AVAILABLE and self.available_at and self.available_at > boundary:
            raise FutureLeakageError(f"{self.name} is available after snapshot cutoff")

    def to_primitive(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "value": self.value,
            "observed_at": self.observed_at,
            "available_at": self.available_at,
            "owner": self.owner,
            "source_version": self.source_version,
            "status": self.status,
            "reason": self.reason,
        }


class FutureLeakageError(ValueError):
    """Raised when information is newer than its decision boundary."""


class UndeclaredFeatureAccess(KeyError):
    """Raised when a consumer reads outside its declared allowlist."""


@dataclass(frozen=True, slots=True)
class FeatureSet:
    observations: tuple[FeatureObservation, ...]
    schema_hash: str

    def __post_init__(self) -> None:
        ordered = tuple(sorted(self.observations, key=lambda item: item.name))
        if len({item.name for item in ordered}) != len(ordered):
            raise ValueError("feature names must be unique")
        object.__setattr__(self, "observations", ordered)

    def view(self, allowlist: Iterable[str]) -> "FeatureView":
        return FeatureView(self, frozenset(allowlist))

    def to_primitive(self) -> dict[str, Any]:
        return {
            "schema_hash": self.schema_hash,
            "observations": [item.to_primitive() for item in self.observations],
        }


@dataclass(frozen=True, slots=True)
class FeatureView:
    feature_set: FeatureSet
    allowlist: frozenset[str]

    def get(self, name: str) -> FeatureObservation:
        if name not in self.allowlist:
            raise UndeclaredFeatureAccess(name)
        for item in self.feature_set.observations:
            if item.name == name:
                return item
        raise KeyError(name)


class PivotKind(str, Enum):
    HIGH = "HIGH"
    LOW = "LOW"


@dataclass(frozen=True, slots=True)
class ConfirmedPivot:
    kind: PivotKind
    price: float
    bar_index: int
    pivot_at: datetime
    available_at: datetime

    def __post_init__(self) -> None:
        object.__setattr__(self, "pivot_at", utc_datetime(self.pivot_at))
        object.__setattr__(self, "available_at", utc_datetime(self.available_at))
        if self.available_at < self.pivot_at:
            raise ValueError("pivot availability cannot precede pivot")

    def to_primitive(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "price": self.price,
            "bar_index": self.bar_index,
            "pivot_at": self.pivot_at,
            "available_at": self.available_at,
        }


@dataclass(frozen=True, slots=True)
class StructuralLevel:
    level_id: str
    timeframe: Timeframe
    kind: PivotKind
    price: float
    touch_count: int
    pivot_indices: tuple[int, ...]
    pivot_prices: tuple[float, ...]
    first_touch_at: datetime
    last_touch_at: datetime
    available_at: datetime

    def __post_init__(self) -> None:
        object.__setattr__(self, "first_touch_at", utc_datetime(self.first_touch_at))
        object.__setattr__(self, "last_touch_at", utc_datetime(self.last_touch_at))
        object.__setattr__(self, "available_at", utc_datetime(self.available_at))
        if not self.level_id or not math.isfinite(self.price) or self.price <= 0:
            raise ValueError("structural level identity/price is invalid")
        if self.touch_count < 2 or self.touch_count != len(self.pivot_indices):
            raise ValueError("structural level requires at least two matching touches")
        if self.touch_count != len(self.pivot_prices):
            raise ValueError("pivot price/count mismatch")
        if tuple(sorted(self.pivot_indices)) != self.pivot_indices:
            raise ValueError("pivot indices must be ordered")
        if any(right - left < 3 for left, right in zip(self.pivot_indices, self.pivot_indices[1:])):
            raise ValueError("structural touches must be separated by at least three bars")
        if self.first_touch_at > self.last_touch_at or self.available_at < self.last_touch_at:
            raise ValueError("structural level timestamps are incoherent")

    def to_primitive(self) -> dict[str, Any]:
        return {
            "level_id": self.level_id,
            "timeframe": self.timeframe,
            "kind": self.kind,
            "price": self.price,
            "touch_count": self.touch_count,
            "pivot_indices": self.pivot_indices,
            "pivot_prices": self.pivot_prices,
            "first_touch_at": self.first_touch_at,
            "last_touch_at": self.last_touch_at,
            "available_at": self.available_at,
        }


@dataclass(frozen=True, slots=True)
class LevelDistance:
    level_id: str
    price: float
    distance_atr: float
    distance_bps: float

    def __post_init__(self) -> None:
        values = (self.price, self.distance_atr, self.distance_bps)
        if not self.level_id or not all(math.isfinite(value) for value in values):
            raise ValueError("level distance is invalid")
        if self.price <= 0 or self.distance_atr < 0 or self.distance_bps < 0:
            raise ValueError("level distance must be non-negative")

    def to_primitive(self) -> dict[str, Any]:
        return {
            "level_id": self.level_id,
            "price": self.price,
            "distance_atr": self.distance_atr,
            "distance_bps": self.distance_bps,
        }


@dataclass(frozen=True, slots=True)
class StructuralContext:
    status: DataStatus
    pivots: tuple[ConfirmedPivot, ...] = ()
    levels: tuple[StructuralLevel, ...] = ()
    atr14: float | None = None
    cluster_tolerance: float | None = None
    reference_price: float | None = None
    nearest_below: LevelDistance | None = None
    nearest_above: LevelDistance | None = None
    reason: str | None = None

    def __post_init__(self) -> None:
        if self.status is not DataStatus.AVAILABLE and not self.reason:
            raise ValueError("unavailable structural context requires reason")
        if self.status is DataStatus.AVAILABLE:
            values = (self.atr14, self.cluster_tolerance, self.reference_price)
            if any(value is None or not math.isfinite(value) or value <= 0 for value in values):
                raise ValueError("available structural context requires ATR, tolerance, and reference price")
        ordered = tuple(sorted(self.levels, key=lambda level: (level.price, level.kind.value, level.level_id)))
        if len({level.level_id for level in ordered}) != len(ordered):
            raise ValueError("structural level IDs must be unique")
        object.__setattr__(self, "levels", ordered)

    def to_primitive(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "reason": self.reason,
            "pivots": [pivot.to_primitive() for pivot in self.pivots],
            "levels": [level.to_primitive() for level in self.levels],
            "atr14": self.atr14,
            "cluster_tolerance": self.cluster_tolerance,
            "reference_price": self.reference_price,
            "nearest_below": self.nearest_below.to_primitive() if self.nearest_below else None,
            "nearest_above": self.nearest_above.to_primitive() if self.nearest_above else None,
        }


@dataclass(frozen=True, slots=True)
class TimeframeSnapshot:
    timeframe: Timeframe
    status: DataStatus
    candle_count: int
    required_warmup_bars: int
    latest_closed_at: datetime | None
    latest_candle: Candle | None
    features: FeatureSet
    structural: StructuralContext | None
    reason: str | None = None

    def __post_init__(self) -> None:
        if self.latest_closed_at is not None:
            object.__setattr__(self, "latest_closed_at", utc_datetime(self.latest_closed_at))
        if self.latest_candle is not None and self.latest_closed_at != self.latest_candle.close_at:
            raise ValueError("latest_candle must match latest_closed_at")
        if self.status is not DataStatus.AVAILABLE and not self.reason:
            raise ValueError("unavailable timeframe requires reason")

    def assert_causal(self, cutoff: datetime) -> None:
        boundary = utc_datetime(cutoff)
        if self.latest_closed_at is not None and self.latest_closed_at > boundary:
            raise FutureLeakageError(f"{self.timeframe.value} close is after cutoff")
        if self.latest_candle is not None and self.latest_candle.available_at > boundary:
            raise FutureLeakageError(f"{self.timeframe.value} candle is after cutoff")
        for feature in self.features.observations:
            feature.assert_available_by(boundary)
        if self.structural:
            for pivot in self.structural.pivots:
                if pivot.available_at > boundary:
                    raise FutureLeakageError("pivot is not confirmed at snapshot cutoff")
            for level in self.structural.levels:
                if level.available_at > boundary:
                    raise FutureLeakageError("structural level is not available at snapshot cutoff")

    def to_primitive(self) -> dict[str, Any]:
        return {
            "timeframe": self.timeframe,
            "status": self.status,
            "candle_count": self.candle_count,
            "required_warmup_bars": self.required_warmup_bars,
            "latest_closed_at": self.latest_closed_at,
            "latest_candle": self.latest_candle.to_primitive() if self.latest_candle else None,
            "features": self.features.to_primitive(),
            "structural": self.structural.to_primitive() if self.structural else None,
            "reason": self.reason,
        }


@dataclass(frozen=True, slots=True)
class MarketSnapshot:
    snapshot_id: str
    schema_version: str
    schema_hash: str
    symbol: str
    decision_at: datetime
    built_at: datetime
    proposed_side: Side | None
    signal_id: str | None
    reference_price: float
    timeframes: tuple[TimeframeSnapshot, ...]
    source_versions: tuple[tuple[str, Any], ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        object.__setattr__(self, "decision_at", utc_datetime(self.decision_at))
        object.__setattr__(self, "built_at", utc_datetime(self.built_at))
        object.__setattr__(self, "source_versions", frozen_pairs(self.source_versions))
        ordered = tuple(sorted(self.timeframes, key=lambda item: item.timeframe.minutes))
        if len({item.timeframe for item in ordered}) != len(ordered):
            raise ValueError("snapshot timeframes must be unique")
        object.__setattr__(self, "timeframes", ordered)
        if not math.isfinite(self.reference_price) or self.reference_price <= 0:
            raise ValueError("reference_price must be finite and positive")
        for timeframe in ordered:
            timeframe.assert_causal(self.decision_at)
        expected = content_hash(self.identity_payload())
        if self.snapshot_id != expected:
            raise ValueError("snapshot_id does not match canonical identity")

    @classmethod
    def create(
        cls,
        *,
        schema_version: str,
        schema_hash: str,
        symbol: str,
        decision_at: datetime,
        built_at: datetime,
        proposed_side: Side | None,
        signal_id: str | None,
        reference_price: float,
        timeframes: Iterable[TimeframeSnapshot],
        source_versions: Mapping[str, Any] | tuple[tuple[str, Any], ...],
    ) -> "MarketSnapshot":
        values = tuple(timeframes)
        versions = frozen_pairs(source_versions)
        prototype = object.__new__(cls)
        object.__setattr__(prototype, "snapshot_id", "")
        object.__setattr__(prototype, "schema_version", schema_version)
        object.__setattr__(prototype, "schema_hash", schema_hash)
        object.__setattr__(prototype, "symbol", symbol)
        object.__setattr__(prototype, "decision_at", utc_datetime(decision_at))
        object.__setattr__(prototype, "built_at", utc_datetime(built_at))
        object.__setattr__(prototype, "proposed_side", proposed_side)
        object.__setattr__(prototype, "signal_id", signal_id)
        object.__setattr__(prototype, "reference_price", float(reference_price))
        object.__setattr__(prototype, "timeframes", tuple(sorted(values, key=lambda item: item.timeframe.minutes)))
        object.__setattr__(prototype, "source_versions", versions)
        snapshot_id = content_hash(prototype.identity_payload())
        return cls(snapshot_id=snapshot_id, schema_version=schema_version, schema_hash=schema_hash,
                   symbol=symbol, decision_at=decision_at, built_at=built_at,
                   proposed_side=proposed_side, signal_id=signal_id,
                   reference_price=reference_price, timeframes=values, source_versions=versions)

    def identity_payload(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "schema_hash": self.schema_hash,
            "symbol": self.symbol,
            "decision_at": self.decision_at,
            "built_at": self.built_at,
            "proposed_side": self.proposed_side,
            "signal_id": self.signal_id,
            "reference_price": self.reference_price,
            "timeframes": [item.to_primitive() for item in self.timeframes],
            "source_versions": dict(self.source_versions),
        }

    def to_primitive(self) -> dict[str, Any]:
        return {"snapshot_id": self.snapshot_id, **self.identity_payload()}

    def canonical_bytes(self) -> bytes:
        return canonical_json_bytes(self.to_primitive())
