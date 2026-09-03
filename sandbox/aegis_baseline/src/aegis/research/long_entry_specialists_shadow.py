"""Causal LONG archetypes and outcome labels for Shadow research."""

from __future__ import annotations

import math
from enum import Enum
from typing import Any, Mapping, Sequence

from ..domain import Candle


class LongEntrySpecialistError(ValueError):
    pass


class LongArchetype(str, Enum):
    TREND_CONTINUATION = "TREND_CONTINUATION"
    CONFIRMED_REVERSAL = "CONFIRMED_REVERSAL"
    OTHER = "OTHER"


class LongArchetypeV2(str, Enum):
    """Causal LONG setup families evaluated independently in Shadow."""

    TREND_PULLBACK = "TREND_PULLBACK"
    CONFIRMED_BREAKOUT = "CONFIRMED_BREAKOUT"
    CONFIRMED_REVERSAL = "CONFIRMED_REVERSAL"
    SPECULATIVE_REBOUND = "SPECULATIVE_REBOUND"
    OTHER = "OTHER"


LONG_SPECIALIST_FEATURE_NAMES = (
    "market_breadth_6",
    "market_direction_6",
    "market_concentration_6",
    "cross_dispersion_return_6",
    "cross_rank_return_6",
    "btc_trend_proxy",
    "btc_volatility_12",
    "btc_divergence_6",
    "eth_trend_proxy",
    "eth_divergence_6",
    "ret_1",
    "ret_3",
    "ret_6",
    "ret_12",
    "ret_24",
    "relative_return_6",
    "relative_return_12",
    "momentum_acceleration_3_12",
    "return_zscore_24",
    "upside_momentum_6",
    "downside_momentum_6",
    "persistence_6",
    "close_vs_ema_6",
    "close_vs_ema_12",
    "close_vs_ema_24",
    "close_vs_ema_48",
    "ema_gap_6_12",
    "ema_gap_12_24",
    "ema_slope_6",
    "ema_slope_12",
    "ema_slope_24",
    "trend_stack_long",
    "trend_stack_short",
    "trend_strength_12",
    "chop_12",
    "atr_12",
    "atr_24",
    "volatility_ratio_6_24",
    "volatility_compression_12_24",
    "range_expansion",
    "volume_ratio_6_24",
    "volume_zscore_24",
    "volume_return_1",
    "volume_trend_12",
    "close_position_in_range",
    "body_to_range",
    "lower_wick_fraction",
    "upper_wick_fraction",
    "distance_to_rolling_high_12",
    "distance_to_rolling_high_24",
    "distance_to_rolling_low_12",
    "distance_to_rolling_low_24",
    "consecutive_green_count",
    "consecutive_red_count",
    "exhaustion_down_proxy",
    "rebound_risk_proxy",
    "failed_breakdown_proxy",
    "fake_breakdown_risk_proxy",
    "immediate_reversal_risk_proxy",
    "high_vol_regime_proxy",
    "low_vol_regime_proxy",
)


def _number(values: Mapping[str, Any], name: str) -> float:
    try:
        value = float(values[name])
    except (KeyError, TypeError, ValueError) as exc:
        raise LongEntrySpecialistError(f"missing LONG feature: {name}") from exc
    if not math.isfinite(value):
        raise LongEntrySpecialistError(f"non-finite LONG feature: {name}")
    return value


def long_specialist_feature_vector(features: Mapping[str, Any]) -> tuple[float, ...]:
    """Return an ordered causal feature vector with no outcome fields."""

    return tuple(_number(features, name) for name in LONG_SPECIALIST_FEATURE_NAMES)


def classify_long_archetype(features: Mapping[str, Any]) -> Mapping[str, Any]:
    """Route a setup by observable structure, never by its future outcome."""

    continuation = {
        "market_breadth_positive": _number(features, "market_breadth_6") >= 0.55,
        "market_direction_positive": _number(features, "market_direction_6") > 0.0,
        "btc_trend_positive": _number(features, "btc_trend_proxy") > 0.0,
        "long_trend_stack": _number(features, "trend_stack_long") > 0.5,
        "above_ema12": _number(features, "close_vs_ema_12") > 0.0,
        "fast_slope_positive": _number(features, "ema_slope_6") > 0.0,
        "medium_slope_positive": _number(features, "ema_slope_12") > 0.0,
        "six_bar_return_positive": _number(features, "ret_6") > 0.0,
        "twelve_bar_return_positive": _number(features, "ret_12") > 0.0,
        "relative_strength_positive": _number(features, "relative_return_6") > 0.0,
        "volume_not_contracting": _number(features, "volume_ratio_6_24") >= 1.0,
    }
    reversal = {
        "prior_down_move": _number(features, "ret_12") < 0.0
        or _number(features, "ret_24") < 0.0,
        "one_bar_turn_positive": _number(features, "ret_1") > 0.0,
        "three_bar_turn_positive": _number(features, "ret_3") > 0.0,
        "momentum_accelerating_up": _number(features, "momentum_acceleration_3_12")
        > 0.0,
        "fast_slope_positive": _number(features, "ema_slope_6") > 0.0,
        "fast_ema_reclaimed": _number(features, "close_vs_ema_6") > 0.0,
        "close_near_bar_high": _number(features, "close_position_in_range") >= 0.60,
        "lower_wick_dominates": _number(features, "lower_wick_fraction")
        > _number(features, "upper_wick_fraction"),
        "explicit_exhaustion_or_rebound": _number(features, "exhaustion_down_proxy")
        > 0.5
        or _number(features, "rebound_risk_proxy") > 0.5,
    }
    continuation_count = sum(continuation.values())
    reversal_count = sum(reversal.values())
    if continuation_count >= 6 and continuation_count > reversal_count:
        archetype = LongArchetype.TREND_CONTINUATION
    elif reversal["prior_down_move"] and reversal_count >= 5:
        archetype = LongArchetype.CONFIRMED_REVERSAL
    else:
        archetype = LongArchetype.OTHER
    return {
        "schema_id": "aegis-long-entry-archetype-shadow-v1",
        "mode": "SHADOW",
        "archetype": archetype.value,
        "continuation_evidence": continuation,
        "continuation_evidence_count": continuation_count,
        "reversal_evidence": reversal,
        "reversal_evidence_count": reversal_count,
        "routing_semantics": "RESEARCH_ONLY_CAUSAL_ARCHETYPE_NOT_EXECUTION_GUARD",
        "selection_effect": "NONE",
        "exchange_authority": False,
        "exchange_mutations": 0,
    }


def classify_long_archetype_v2(features: Mapping[str, Any]) -> Mapping[str, Any]:
    """Classify a LONG setup without observing any future candle."""

    atr = max(_number(features, "atr_12"), 1e-12)
    pullback = {
        "long_trend_stack": _number(features, "trend_stack_long") > 0.5,
        "medium_slope_positive": _number(features, "ema_slope_12") > 0.0,
        "slow_slope_positive": _number(features, "ema_slope_24") > 0.0,
        "prior_direction_positive": _number(features, "ret_12") > 0.0,
        "near_fast_or_medium_ema": (
            -0.75 * atr <= _number(features, "close_vs_ema_12") <= 1.25 * atr
        ),
        "turning_up": _number(features, "ret_1") > 0.0,
        "not_broad_market_weakness": _number(features, "market_breadth_6") >= 0.40,
    }
    breakout = {
        "at_or_above_prior_high": _number(features, "distance_to_rolling_high_12")
        <= 0.0,
        "short_momentum_positive": _number(features, "ret_3") > 0.0,
        "medium_momentum_positive": _number(features, "ret_6") > 0.0,
        "close_near_bar_high": _number(features, "close_position_in_range") >= 0.65,
        "body_has_conviction": _number(features, "body_to_range") >= 0.45,
        "volume_confirmation": _number(features, "volume_ratio_6_24") >= 1.0,
        "market_not_broadly_negative": _number(features, "market_breadth_6") >= 0.40,
    }
    reversal = {
        "prior_down_move": _number(features, "ret_12") < 0.0
        or _number(features, "ret_24") < 0.0,
        "one_bar_turn_positive": _number(features, "ret_1") > 0.0,
        "three_bar_turn_positive": _number(features, "ret_3") > 0.0,
        "momentum_accelerating_up": _number(features, "momentum_acceleration_3_12")
        > 0.0,
        "fast_ema_reclaimed": _number(features, "close_vs_ema_6") > 0.0,
        "close_near_bar_high": _number(features, "close_position_in_range") >= 0.60,
        "lower_wick_confirmation": _number(features, "lower_wick_fraction")
        > _number(features, "upper_wick_fraction"),
    }
    rebound = {
        "prior_down_move": reversal["prior_down_move"],
        "downside_extension": _number(features, "extension_down_proxy") > 0.0,
        "exhaustion_or_rebound": _number(features, "exhaustion_down_proxy") > 0.0
        or _number(features, "rebound_risk_proxy") > 0.5,
        "wick_or_reclaim": _number(features, "lower_wick_fraction") >= 0.30
        or _number(features, "failed_breakdown_proxy") > 0.5,
        "confirmation_missing": not (
            reversal["three_bar_turn_positive"]
            and reversal["fast_ema_reclaimed"]
            and reversal["close_near_bar_high"]
        ),
    }

    counts = {
        LongArchetypeV2.TREND_PULLBACK: sum(pullback.values()),
        LongArchetypeV2.CONFIRMED_BREAKOUT: sum(breakout.values()),
        LongArchetypeV2.CONFIRMED_REVERSAL: sum(reversal.values()),
        LongArchetypeV2.SPECULATIVE_REBOUND: sum(rebound.values()),
    }
    if (
        breakout["at_or_above_prior_high"]
        and counts[LongArchetypeV2.CONFIRMED_BREAKOUT] >= 5
    ):
        archetype = LongArchetypeV2.CONFIRMED_BREAKOUT
    elif pullback["long_trend_stack"] and counts[LongArchetypeV2.TREND_PULLBACK] >= 5:
        archetype = LongArchetypeV2.TREND_PULLBACK
    elif (
        reversal["prior_down_move"] and counts[LongArchetypeV2.CONFIRMED_REVERSAL] >= 5
    ):
        archetype = LongArchetypeV2.CONFIRMED_REVERSAL
    elif (
        rebound["prior_down_move"] and counts[LongArchetypeV2.SPECULATIVE_REBOUND] >= 3
    ):
        archetype = LongArchetypeV2.SPECULATIVE_REBOUND
    else:
        archetype = LongArchetypeV2.OTHER
    return {
        "schema_id": "aegis-long-entry-archetype-shadow-v2",
        "mode": "SHADOW",
        "archetype": archetype.value,
        "evidence": {
            LongArchetypeV2.TREND_PULLBACK.value: pullback,
            LongArchetypeV2.CONFIRMED_BREAKOUT.value: breakout,
            LongArchetypeV2.CONFIRMED_REVERSAL.value: reversal,
            LongArchetypeV2.SPECULATIVE_REBOUND.value: rebound,
        },
        "evidence_counts": {key.value: value for key, value in counts.items()},
        "routing_semantics": "RESEARCH_ONLY_CAUSAL_ARCHETYPE_NOT_EXECUTION_GUARD",
        "selection_effect": "NONE",
        "exchange_authority": False,
        "exchange_mutations": 0,
    }


def exact_long_path_outcome(
    *,
    entry_price: float,
    future_candles: Sequence[Candle],
    round_trip_cost_fraction: float = 0.001,
    favorable_fraction: float = 0.003,
    adverse_fraction: float = 0.003,
    fast_bars: int = 6,
) -> Mapping[str, Any]:
    """Label the actual future LONG path, including barrier ordering."""

    if (
        not math.isfinite(entry_price)
        or entry_price <= 0.0
        or not future_candles
        or fast_bars <= 0
        or not all(
            math.isfinite(value) and value >= 0.0
            for value in (
                round_trip_cost_fraction,
                favorable_fraction,
                adverse_fraction,
            )
        )
    ):
        raise LongEntrySpecialistError("LONG exact outcome contract is invalid")

    favorable_path = [
        max(0.0, candle.high / entry_price - 1.0) for candle in future_candles
    ]
    adverse_path = [
        max(0.0, 1.0 - candle.low / entry_price) for candle in future_candles
    ]
    mfe = max(favorable_path)
    mae = max(adverse_path)
    time_to_mfe = favorable_path.index(mfe) + 1
    time_to_mae = adverse_path.index(mae) + 1
    favorable_bar = next(
        (
            index
            for index, value in enumerate(favorable_path, start=1)
            if value >= favorable_fraction
        ),
        None,
    )
    adverse_bar = next(
        (
            index
            for index, value in enumerate(adverse_path, start=1)
            if value >= adverse_fraction
        ),
        None,
    )
    same_bar_ambiguity = favorable_bar is not None and favorable_bar == adverse_bar
    if same_bar_ambiguity:
        barrier_order = "SAME_BAR_AMBIGUOUS"
    elif favorable_bar is not None and (
        adverse_bar is None or favorable_bar < adverse_bar
    ):
        barrier_order = "FAVORABLE_FIRST"
    elif adverse_bar is not None:
        barrier_order = "ADVERSE_FIRST"
    else:
        barrier_order = "NEITHER_REACHED"
    terminal_return = future_candles[-1].close / entry_price - 1.0
    clean_fast_success = (
        barrier_order == "FAVORABLE_FIRST" and favorable_bar <= fast_bars
    )
    dangerous_entry = (
        barrier_order in {"ADVERSE_FIRST", "SAME_BAR_AMBIGUOUS"}
        or mae >= adverse_fraction
    )
    return {
        "schema_id": "aegis-long-entry-path-label-shadow-v2",
        "entry_semantics": "SIGNAL_CANDLE_CLOSE",
        "horizon_bars": len(future_candles),
        "mfe_fraction": mfe,
        "mae_fraction": mae,
        "terminal_return_fraction": terminal_return,
        "net_return_after_costs": terminal_return - round_trip_cost_fraction,
        "time_to_mfe_peak": time_to_mfe,
        "time_to_mae_peak": time_to_mae,
        "time_underwater_bars": sum(
            candle.close < entry_price for candle in future_candles
        ),
        "first_favorable_bar": favorable_bar,
        "first_adverse_bar": adverse_bar,
        "barrier_order": barrier_order,
        "same_bar_ambiguity": same_bar_ambiguity,
        "clean_fast_success": clean_fast_success,
        "dangerous_entry": dangerous_entry,
        "source_semantics": "ACTUAL_FUTURE_OHLC_PATH_NOT_DIRECTIONAL_MIRROR",
        "outcome_available": True,
        "selection_effect": "NONE",
        "exchange_authority": False,
        "exchange_mutations": 0,
    }


def mirror_short_path_as_long_outcome(
    *,
    observed: Mapping[str, Any],
    label: Mapping[str, Any],
    round_trip_cost_fraction: float = 0.001,
    favorable_fraction: float = 0.003,
    adverse_fraction: float = 0.003,
    fast_bars: int = 6,
) -> Mapping[str, Any]:
    """Derive exact LONG extrema from the same price path summarized for SHORT."""

    if (
        not all(
            math.isfinite(value) and value >= 0.0
            for value in (
                round_trip_cost_fraction,
                favorable_fraction,
                adverse_fraction,
            )
        )
        or fast_bars <= 0
    ):
        raise LongEntrySpecialistError("LONG outcome contract is invalid")
    try:
        long_mfe = float(observed["mae_fraction"])
        long_mae = float(observed["mfe_fraction"])
        short_terminal = float(observed["path_metrics"]["12"]["terminal_short_return"])
        time_to_long_mfe = int(label["time_to_mae"])
        time_to_long_mae = int(label["time_to_mfe"])
    except (KeyError, TypeError, ValueError) as exc:
        raise LongEntrySpecialistError("LONG mirrored outcome is incomplete") from exc
    values: Sequence[float] = (long_mfe, long_mae, short_terminal)
    if (
        not all(math.isfinite(value) for value in values)
        or min(long_mfe, long_mae) < 0.0
    ):
        raise LongEntrySpecialistError("LONG mirrored outcome is invalid")
    terminal_long = -short_terminal
    net_long = terminal_long - round_trip_cost_fraction
    favorable_peak_first = time_to_long_mfe < time_to_long_mae
    clean_fast_success = (
        long_mfe >= favorable_fraction
        and 0 <= time_to_long_mfe <= fast_bars
        and favorable_peak_first
    )
    dangerous_entry = long_mae >= adverse_fraction or (
        time_to_long_mae <= time_to_long_mfe and long_mfe < favorable_fraction
    )
    return {
        "schema_id": "aegis-long-entry-path-label-shadow-v1",
        "mfe_fraction": long_mfe,
        "mae_fraction": long_mae,
        "terminal_return_fraction": terminal_long,
        "net_return_after_costs": net_long,
        "time_to_mfe_peak": time_to_long_mfe,
        "time_to_mae_peak": time_to_long_mae,
        "favorable_peak_first": favorable_peak_first,
        "clean_fast_success": clean_fast_success,
        "dangerous_entry": dangerous_entry,
        "source_semantics": "EXACT_DIRECTIONAL_MIRROR_OF_SAME_PRICE_PATH_EXTREMA",
        "outcome_available": True,
        "selection_effect": "NONE",
        "exchange_authority": False,
        "exchange_mutations": 0,
    }
