from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Literal

Side = Literal["LONG", "SHORT"]
PivotSide = Literal["LOW", "HIGH"]
SignalSide = Literal["LONG", "SHORT", "NONE"]
OutsideDirection = Literal["UP", "DOWN"]


@dataclass(frozen=True, slots=True)
class Candle1m:
    symbol: str
    open_time: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float
    open_source: str | None = None
    high_source: str | None = None
    low_source: str | None = None
    close_source: str | None = None


@dataclass(frozen=True, slots=True)
class Candle5m:
    symbol: str
    open_time: datetime
    available_at: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float
    segment_id: int = 0
    high_source: str | None = None
    low_source: str | None = None


@dataclass(frozen=True, slots=True)
class DataIntegrityEvent:
    symbol: str
    block_open_time: datetime
    reason: str = "DATA_INTEGRITY"


@dataclass(frozen=True, slots=True)
class Pivot:
    symbol: str
    side: PivotSide
    price: float
    source_price: str
    pivot_at: datetime
    available_at: datetime


@dataclass(frozen=True, slots=True)
class Touch:
    touch_at: datetime
    bar_index: int
    level_at_touch: float


@dataclass(slots=True)
class LevelCluster:
    cluster_id: str
    symbol: str
    side: PivotSide
    first_pivot_at: datetime
    pivots: list[Pivot] = field(default_factory=list)
    touches: list[Touch] = field(default_factory=list)
    armed: bool = True

    @property
    def center(self) -> float:
        prices = sorted(p.price for p in self.pivots)
        count = len(prices)
        if count == 0:
            raise ValueError("empty cluster has no center")
        middle = count // 2
        if count % 2:
            return prices[middle]
        return (prices[middle - 1] + prices[middle]) / 2.0

    @property
    def last_touch_at(self) -> datetime | None:
        return self.touches[-1].touch_at if self.touches else None


@dataclass(frozen=True, slots=True)
class RangePair:
    support_cluster_id: str
    resistance_cluster_id: str
    support: float
    resistance: float
    midpoint: float
    amplitude: float
    support_touches: int
    resistance_touches: int
    pair_recency_at: datetime
    pair_first_eligible_at: datetime


@dataclass(frozen=True, slots=True)
class LevelSnapshot:
    decision_at: datetime
    pair: RangePair
    atr14_raw: float
    range_episode_id: str
    range_id: str


@dataclass(frozen=True, slots=True)
class RegimeSnapshot:
    technical_regime: str
    transition_risk: str
    adx: float
    atr_percentile: float
    bollinger_width_percentile: float
    volume_ratio: float
    range_breakout: str
    failed_breakout_count: int
    structure: str
    chop_risk: float
    atr14_raw: float


@dataclass(frozen=True, slots=True)
class Signal:
    side: SignalSide
    decision_at: datetime
    reason: str


@dataclass(frozen=True, slots=True)
class PendingEntry:
    symbol: str
    side: Side
    decision_at: datetime
    entry_available_at: datetime
    range_episode_id: str
    range_id: str
    range_confirmed_at: datetime
    support: float
    resistance: float
    midpoint: float
    atr_entry: float
    regime_at_entry: str
    range_confidence_at_entry: float
    tail_risk_score_at_entry: float | None


@dataclass(slots=True)
class Position:
    symbol: str
    side: Side
    entry_at: datetime
    entry_fill: float
    range_episode_id: str
    range_id: str
    range_confirmed_at: datetime
    support_at_entry: float
    resistance_at_entry: float
    midpoint_at_entry: float
    atr_entry: float
    stop_at_entry: float
    target_at_entry: float
    thesis_serialized: str
    thesis_feature_hash: str
    closed_bars: int = 0
    adverse_breakout_count: int = 0
    pending_exit_reason: str | None = None


@dataclass(frozen=True, slots=True)
class FillEvent:
    symbol: str
    side: Side
    fill_at: datetime
    fill_price: float
    reason: str


@dataclass(slots=True)
class Episode:
    symbol: str
    range_episode_id: str
    range_confirmed_at: datetime
    support_cluster_id: str
    resistance_cluster_id: str
    previous_snapshot: LevelSnapshot
    outside_direction: OutsideDirection | None = None
    outside_count: int = 0
    ended_at: datetime | None = None
    end_reason: str | None = None
