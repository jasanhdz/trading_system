"""Preregistered LONG v3.1 timing, context, and committee contracts."""

from __future__ import annotations

import math
from enum import Enum
from typing import Any, Mapping

from ..domain import Candle
from .long_entry_v3_shadow import LongCandidateFamily


class LongEntryTrigger(str, Enum):
    NEXT_BAR_OPEN = "NEXT_BAR_OPEN"
    NEXT_BAR_FOLLOW_THROUGH = "NEXT_BAR_FOLLOW_THROUGH"
    NEXT_BAR_HIGHER_CLOSE = "NEXT_BAR_HIGHER_CLOSE"


class LongV31ShadowError(ValueError):
    pass


def _number(values: Mapping[str, Any], name: str) -> float:
    try:
        value = float(values[name])
    except (KeyError, TypeError, ValueError) as exc:
        raise LongV31ShadowError(f"missing LONG v3.1 input: {name}") from exc
    if not math.isfinite(value):
        raise LongV31ShadowError(f"non-finite LONG v3.1 input: {name}")
    return value


def global_context_gate(
    *,
    base: Mapping[str, Any],
    context: Mapping[str, Any],
    regime: Mapping[str, str],
    config: Mapping[str, Any],
) -> Mapping[str, Any]:
    """Apply only preregistered, causal market-context conditions."""

    hourly_long = max(
        _number(context, "1h_trend_stack_long"),
        _number(context, "15m_trend_stack_long"),
    )
    evidence = {
        "market_breadth": _number(base, "market_breadth_6")
        >= float(config["minimum_market_breadth_6"]),
        "btc_context": _number(base, "btc_trend_proxy")
        >= float(config["minimum_btc_trend_proxy"]),
        "eth_context": _number(base, "eth_trend_proxy")
        >= float(config["minimum_eth_trend_proxy"]),
        "regime_direction": str(regime["direction"])
        not in {str(value) for value in config["prohibited_regime_directions"]},
        "higher_timeframe_support": hourly_long > 0.5,
    }
    return {
        "passed": all(evidence.values()),
        "evidence": evidence,
        "selection_effect": "NONE",
        "exchange_authority": False,
    }


def entry_confirmation(
    *,
    family: str,
    signal: Candle,
    confirmation: Candle | None,
    confirmation_micro: Mapping[str, Any] | None,
    config: Mapping[str, Any],
) -> Mapping[str, Any]:
    """Decide causally whether an opportunity becomes executable."""

    try:
        trigger = LongEntryTrigger(str(config["family_trigger"][family]))
    except (KeyError, ValueError) as exc:
        raise LongV31ShadowError("LONG v3.1 family trigger is invalid") from exc
    if trigger is LongEntryTrigger.NEXT_BAR_OPEN:
        return {
            "passed": True,
            "trigger": trigger.value,
            "entry_offset_bars": 1,
            "evidence": {"opportunity_structure_is_confirmation": True},
            "selection_effect": "NONE",
        }
    if confirmation is None or confirmation_micro is None:
        raise LongV31ShadowError("LONG v3.1 confirmation data is unavailable")
    if signal.close_time != confirmation.open_time or not confirmation.is_closed:
        raise LongV31ShadowError("LONG v3.1 confirmation candle is not causal")
    green = confirmation.close > confirmation.open
    higher_close = confirmation.close > signal.close
    taker_ratio = _number(confirmation_micro, "taker_buy_ratio_1")
    if trigger is LongEntryTrigger.NEXT_BAR_FOLLOW_THROUGH:
        raw = config["next_bar_follow_through"]
        candle_range = max(confirmation.high - confirmation.low, 1e-12)
        close_location = (confirmation.close - confirmation.low) / candle_range
        evidence = {
            "green_candle": green,
            "higher_close": higher_close,
            "close_location": close_location >= float(raw["minimum_close_location"]),
            "taker_support": taker_ratio >= float(raw["minimum_taker_buy_ratio"]),
        }
    else:
        raw = config["next_bar_higher_close"]
        evidence = {
            "green_candle": green,
            "higher_close": higher_close,
            "taker_support": taker_ratio >= float(raw["minimum_taker_buy_ratio"]),
        }
    return {
        "passed": all(evidence.values()),
        "trigger": trigger.value,
        "entry_offset_bars": 2,
        "evidence": evidence,
        "selection_effect": "NONE",
    }


def family_horizon(family: str, config: Mapping[str, Any]) -> int:
    try:
        horizon = int(config["family_horizon_bars"][family])
    except (KeyError, TypeError, ValueError) as exc:
        raise LongV31ShadowError("LONG v3.1 family horizon is invalid") from exc
    if family not in {
        candidate.value
        for candidate in LongCandidateFamily
        if candidate is not LongCandidateFamily.NONE
    } or horizon <= 0:
        raise LongV31ShadowError("LONG v3.1 family horizon is invalid")
    return horizon


def specialist_committee_score_v31(
    opportunity_probability: float,
    timing_probability: float,
    falling_knife_probability: float,
    path_risk_probability: float,
) -> float:
    """Combine calibrated outputs without manufacturing votes or values."""

    values = (
        opportunity_probability,
        timing_probability,
        falling_knife_probability,
        path_risk_probability,
    )
    if not all(math.isfinite(value) and 0.0 <= value <= 1.0 for value in values):
        raise LongV31ShadowError("LONG v3.1 probability is invalid")
    return (
        opportunity_probability
        * timing_probability
        * (1.0 - falling_knife_probability)
        * (1.0 - path_risk_probability)
    )
