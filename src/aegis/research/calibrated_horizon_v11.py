"""Causal regime, clean-entry labels, and utility attribution for V11."""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

from .competing_barrier_v10 import BarrierContract, BarrierOutcome, BarrierResearchError


def _finite(value: Any, name: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise BarrierResearchError(f"non-finite V11 value: {name}")
    return result


def causal_regime(
    context: Mapping[str, Any], config: Mapping[str, Any]
) -> str:
    four_hour = _finite(context["rolling_4h_return"], "rolling_4h_return")
    twelve_hour = _finite(context["rolling_12h_return"], "rolling_12h_return")
    volatility = _finite(context["rolling_4h_volatility"], "rolling_4h_volatility")
    if volatility < 0.0:
        raise BarrierResearchError("negative causal volatility")
    if (
        abs(four_hour) <= float(config["range_absolute_4h_return"])
        and abs(twelve_hour) <= float(config["range_absolute_12h_return"])
    ):
        direction = "RANGE"
    elif four_hour > 0.0 and twelve_hour > 0.0:
        direction = "TREND_UP"
    elif four_hour < 0.0 and twelve_hour < 0.0:
        direction = "TREND_DOWN"
    else:
        direction = "TRANSITION"
    volatility_state = (
        "HIGH_VOL"
        if volatility >= float(config["high_volatility_threshold"])
        else "LOW_VOL"
    )
    identity = f"{direction}_{volatility_state}"
    if identity not in set(config["identities"]):
        raise BarrierResearchError("unregistered V11 causal regime")
    return identity


def clean_entry_diagnostics(
    *,
    side: str,
    entry_price: float,
    future_bars: Sequence[Any],
    primary_outcome: Mapping[str, Any],
    maximum_mae_fraction_of_barrier: float,
    maximum_event_bar: int,
    severe_cost_fraction: float,
) -> Mapping[str, Any]:
    entry = _finite(entry_price, "entry_price")
    horizon = int(primary_outcome["horizon_bars"])
    if side not in {"LONG", "SHORT"} or entry <= 0.0 or len(future_bars) < horizon:
        raise BarrierResearchError("invalid V11 clean-entry path")
    event_raw = primary_outcome.get("event_bar")
    event_bar = int(event_raw) if event_raw is not None else None
    observed_bars = event_bar if event_bar is not None else horizon
    pre_event_mae = 0.0
    maximum_favorable = 0.0
    first_positive: int | None = None
    for index, bar in enumerate(future_bars[:horizon], start=1):
        high = _finite(bar.high, "high")
        low = _finite(bar.low, "low")
        close = _finite(bar.close, "close")
        favorable = high / entry - 1.0 if side == "LONG" else 1.0 - low / entry
        adverse = 1.0 - low / entry if side == "LONG" else high / entry - 1.0
        maximum_favorable = max(maximum_favorable, favorable)
        if index <= observed_bars:
            pre_event_mae = max(pre_event_mae, adverse)
        side_close = close / entry - 1.0 if side == "LONG" else 1.0 - close / entry
        if first_positive is None and side_close > severe_cost_fraction:
            first_positive = index
    barrier = _finite(primary_outcome["adverse_fraction"], "adverse_fraction")
    clean = bool(
        primary_outcome["outcome"] == BarrierOutcome.FAVORABLE_FIRST.value
        and event_bar is not None
        and event_bar <= maximum_event_bar
        and pre_event_mae <= barrier * maximum_mae_fraction_of_barrier
    )
    return {
        "clean_entry": clean,
        "pre_event_mae_fraction": pre_event_mae,
        "pre_event_mae_as_adverse_barrier_fraction": pre_event_mae / barrier,
        "first_positive_after_severe_cost_bar": first_positive,
        "maximum_favorable_excursion_fraction": maximum_favorable,
        "terminal_side_return": _finite(
            primary_outcome["terminal_side_return"], "terminal_side_return"
        ),
        "observed_bars_for_pre_event_mae": observed_bars,
    }


def attributed_utility(
    probabilities: Mapping[str, float],
    contract: BarrierContract,
    *,
    clean_probability: float,
    unknown_penalty_fraction: float,
    clean_bonus_fraction: float,
) -> Mapping[str, float]:
    expected = {outcome.value for outcome in BarrierOutcome}
    if set(probabilities) != expected:
        raise BarrierResearchError("incomplete V11 outcome probabilities")
    values = {name: _finite(value, name) for name, value in probabilities.items()}
    clean = _finite(clean_probability, "clean_probability")
    if (
        any(value < 0.0 or value > 1.0 for value in values.values())
        or not math.isclose(sum(values.values()), 1.0, abs_tol=1e-6)
        or not 0.0 <= clean <= 1.0
        or min(unknown_penalty_fraction, clean_bonus_fraction) < 0.0
    ):
        raise BarrierResearchError("invalid V11 utility inputs")
    favorable = values[BarrierOutcome.FAVORABLE_FIRST.value] * contract.favorable_fraction
    adverse = -values[BarrierOutcome.ADVERSE_FIRST.value] * contract.adverse_fraction
    ambiguous = (
        -values[BarrierOutcome.SAME_BAR_AMBIGUOUS.value]
        * unknown_penalty_fraction
        * contract.adverse_fraction
    )
    unresolved = (
        -values[BarrierOutcome.NEITHER_REACHED.value]
        * unknown_penalty_fraction
        * contract.adverse_fraction
    )
    cost = -contract.severe_cost_fraction
    base = favorable + adverse + ambiguous + unresolved + cost
    bonus = (
        clean * clean_bonus_fraction * contract.favorable_fraction if base > 0.0 else 0.0
    )
    return {
        "favorable_value": favorable,
        "adverse_value": adverse,
        "cost": cost,
        "ambiguous_penalty": ambiguous,
        "unresolved_penalty": unresolved,
        "clean_entry_bonus": bonus,
        "base_utility": base,
        "total_utility": base + bonus,
    }
