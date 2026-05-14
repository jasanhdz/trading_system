from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal


EventRiskSuggestedMode = Literal["NORMAL", "CAUTION", "RISK_OFF", "MANUAL_ONLY"]


@dataclass
class EventRiskSymbolContext:
    symbol: str
    turbo_action: str | None = None
    turbo_score: float | None = None
    gated_action: str | None = None
    gated_reason: str | None = None
    recent_return_15m: float | None = None
    recent_return_1h: float | None = None
    atr_percentile: float | None = None
    mtf_trend: str | None = None
    freshness: dict[str, Any] = field(default_factory=dict)
    entry_quality_score: float | None = None
    tail_risk_score: float | None = None
    recommendation: str | None = None
    reason: str | None = None


@dataclass
class EventRiskMarketContext:
    alt_hold_count: int = 0
    alt_block_shadow_count: int = 0
    alt_signal_count: int = 0
    stale_symbol_count: int = 0
    runtime_symbol_count: int = 0


@dataclass
class EventRiskAutoInput:
    symbol: str
    btc: EventRiskSymbolContext | None = None
    eth: EventRiskSymbolContext | None = None
    current: EventRiskSymbolContext | None = None
    market: EventRiskMarketContext = field(default_factory=EventRiskMarketContext)
    api_warnings: list[str] = field(default_factory=list)


@dataclass
class EventRiskAutoOutput:
    suggested_mode: EventRiskSuggestedMode
    confidence: float
    reasons: list[str]
    mode: Literal["SHADOW"] = "SHADOW"
    btc_context: dict[str, Any] = field(default_factory=dict)
    eth_context: dict[str, Any] = field(default_factory=dict)
    market_context: dict[str, Any] = field(default_factory=dict)
    execute: bool = False
    production_allowed: bool = False
    does_not_change_event_risk_mode: bool = True
    latency_ms: float = 0.0
    last_update: str | None = None
    cache_status: dict[str, Any] = field(default_factory=dict)
