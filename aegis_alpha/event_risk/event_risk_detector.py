from __future__ import annotations

import time
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from typing import Any

from aegis_alpha.event_risk.schema import (
    EventRiskAutoInput,
    EventRiskAutoOutput,
    EventRiskMarketContext,
    EventRiskSymbolContext,
)


LAST_EVENT_RISK_AUTO: dict[str, Any] = {
    "last_suggested_mode": None,
    "confidence": None,
    "reasons": [],
    "last_update": None,
    "cache_status": {
        "status": "cold",
        "evaluations": 0,
    },
}


def _model_dump(payload: Any) -> dict[str, Any]:
    if is_dataclass(payload):
        return asdict(payload)
    if hasattr(payload, "model_dump"):
        return payload.model_dump()
    if hasattr(payload, "dict"):
        return payload.dict()
    return dict(payload or {})


def _symbol_context(payload: Any) -> EventRiskSymbolContext | None:
    if payload is None or isinstance(payload, EventRiskSymbolContext):
        return payload
    if isinstance(payload, dict):
        return EventRiskSymbolContext(**payload)
    return None


def _market_context(payload: Any) -> EventRiskMarketContext:
    if payload is None:
        return EventRiskMarketContext()
    if isinstance(payload, EventRiskMarketContext):
        return payload
    if isinstance(payload, dict):
        return EventRiskMarketContext(**payload)
    return EventRiskMarketContext()


def _input_payload(payload: EventRiskAutoInput | dict[str, Any]) -> EventRiskAutoInput:
    if isinstance(payload, EventRiskAutoInput):
        return payload
    return EventRiskAutoInput(
        symbol=str(payload.get("symbol", "")),
        btc=_symbol_context(payload.get("btc")),
        eth=_symbol_context(payload.get("eth")),
        current=_symbol_context(payload.get("current")),
        market=_market_context(payload.get("market")),
        api_warnings=list(payload.get("api_warnings") or []),
    )


def _finite_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if number != number or number in (float("inf"), float("-inf")):
        return None
    return number


def _action(ctx: EventRiskSymbolContext | None) -> str:
    return str(ctx.turbo_action or ctx.gated_action or "HOLD").upper() if ctx else "HOLD"


def _score(ctx: EventRiskSymbolContext | None) -> float | None:
    return _finite_float(ctx.turbo_score if ctx else None)


def _is_stale(ctx: EventRiskSymbolContext | None) -> bool:
    if ctx is None:
        return True
    freshness = ctx.freshness or {}
    if freshness.get("is_fresh") is False or freshness.get("stale") is True:
        return True
    age = _finite_float(freshness.get("feature_age_seconds"))
    max_age = _finite_float(freshness.get("max_feature_age_seconds"))
    return bool(age is not None and max_age is not None and age > max_age)


def _negative_return(ctx: EventRiskSymbolContext | None, limit_15m: float, limit_1h: float) -> bool:
    if ctx is None:
        return False
    r15 = _finite_float(ctx.recent_return_15m)
    r1h = _finite_float(ctx.recent_return_1h)
    return bool((r15 is not None and r15 <= limit_15m) or (r1h is not None and r1h <= limit_1h))


def _high_atr(ctx: EventRiskSymbolContext | None, limit: float) -> bool:
    atr = _finite_float(ctx.atr_percentile if ctx else None)
    return bool(atr is not None and atr >= limit)


def _weak(ctx: EventRiskSymbolContext | None) -> bool:
    action = _action(ctx)
    score = _score(ctx)
    if action == "HOLD":
        return True
    return bool(score is not None and score < 0.45)


def _very_weak(ctx: EventRiskSymbolContext | None) -> bool:
    score = _score(ctx)
    return bool(_weak(ctx) and (score is None or score < 0.35))


def _context_dict(ctx: EventRiskSymbolContext | None) -> dict[str, Any]:
    return _model_dump(ctx) if ctx is not None else {}


def evaluate_event_risk_auto(payload: EventRiskAutoInput | dict[str, Any]) -> dict[str, Any]:
    start = time.perf_counter()
    data = _input_payload(payload)
    reasons: list[str] = []
    suggested_mode = "NORMAL"
    confidence = 0.55

    btc = data.btc
    eth = data.eth
    market: EventRiskMarketContext = data.market

    btc_stale = _is_stale(btc)
    eth_stale = _is_stale(eth)
    if btc_stale:
        reasons.append("btc_data_stale_or_missing")
    if eth_stale:
        reasons.append("eth_data_stale_or_missing")
    if data.api_warnings:
        reasons.extend([f"api_warning:{warning}" for warning in data.api_warnings[:3]])

    btc_weak = _weak(btc)
    eth_weak = _weak(eth)
    btc_very_weak = _very_weak(btc)
    eth_very_weak = _very_weak(eth)
    severe_drop = _negative_return(btc, -0.025, -0.045) or _negative_return(eth, -0.025, -0.045)
    extreme_drop = _negative_return(btc, -0.045, -0.075) or _negative_return(eth, -0.045, -0.075)
    high_vol = _high_atr(btc, 0.85) or _high_atr(eth, 0.85)
    extreme_vol = _high_atr(btc, 0.95) or _high_atr(eth, 0.95)
    broad_shadow_blocks = market.alt_signal_count > 0 and market.alt_block_shadow_count / max(market.alt_signal_count, 1) >= 0.5
    broad_holds = market.alt_signal_count > 0 and market.alt_hold_count / max(market.alt_signal_count, 1) >= 0.6

    if btc_weak:
        reasons.append("btc_weak_or_hold")
    if eth_weak:
        reasons.append("eth_weak_or_hold")
    if severe_drop:
        reasons.append("btc_eth_recent_drop")
    if high_vol:
        reasons.append("btc_eth_high_volatility")
    if broad_shadow_blocks:
        reasons.append("many_alt_shadow_blocks")
    if broad_holds:
        reasons.append("many_alt_holds")

    if (btc_stale and eth_stale) or extreme_drop or extreme_vol or data.api_warnings:
        suggested_mode = "MANUAL_ONLY"
        confidence = 0.80 if (extreme_drop or extreme_vol or data.api_warnings) else 0.66
    elif (btc_very_weak and eth_very_weak) or (btc_weak and eth_weak and (severe_drop or high_vol or broad_shadow_blocks)):
        suggested_mode = "RISK_OFF"
        confidence = 0.76
    elif btc_weak or eth_weak or severe_drop or high_vol or broad_shadow_blocks or broad_holds or btc_stale or eth_stale:
        suggested_mode = "CAUTION"
        confidence = 0.68
    else:
        suggested_mode = "NORMAL"
        reasons.append("btc_eth_context_stable")
        confidence = 0.62

    output = EventRiskAutoOutput(
        suggested_mode=suggested_mode,  # type: ignore[arg-type]
        confidence=round(float(max(0.0, min(confidence, 1.0))), 3),
        reasons=reasons,
        btc_context=_context_dict(btc),
        eth_context=_context_dict(eth),
        market_context=_model_dump(market),
        execute=False,
        production_allowed=False,
        does_not_change_event_risk_mode=True,
        latency_ms=round((time.perf_counter() - start) * 1000, 3),
        last_update=datetime.now(timezone.utc).isoformat(),
        cache_status={
            "status": "warm",
            "evaluations": int((LAST_EVENT_RISK_AUTO.get("cache_status") or {}).get("evaluations", 0)) + 1,
        },
    )
    result = _model_dump(output)
    LAST_EVENT_RISK_AUTO.update({
        "last_suggested_mode": result["suggested_mode"],
        "confidence": result["confidence"],
        "reasons": list(result["reasons"]),
        "last_update": result["last_update"],
        "cache_status": dict(result["cache_status"]),
    })
    return result


def event_risk_runtime_status() -> dict[str, Any]:
    return dict(LAST_EVENT_RISK_AUTO)
