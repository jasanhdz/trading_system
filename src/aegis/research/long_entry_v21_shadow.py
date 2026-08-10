"""Preregistered LONG v2.1 labels and multitimeframe Shadow features."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from ..data import CanonicalBar
from ..domain import Candle
from ..features import DeterministicFeaturePipeline
from ..training.hybrid_directional import DirectionalSide
from .hybrid_ts_protection_replay import (
    IntrabarPath,
    TsProtectionConfig,
    replay_ts_price_protection,
)
from .long_entry_specialists_shadow import LONG_SPECIALIST_FEATURE_NAMES

MULTITIMEFRAME_LOCAL_FEATURES = (
    "ret_1",
    "ret_3",
    "ret_6",
    "ret_12",
    "ret_24",
    "close_to_open_return",
    "close_position_in_range",
    "body_to_range",
    "lower_wick_fraction",
    "upper_wick_fraction",
    "volume_zscore_24",
    "volume_ratio_6_24",
    "atr_12",
    "volatility_ratio_6_24",
    "close_vs_ema_6",
    "close_vs_ema_12",
    "close_vs_ema_24",
    "close_vs_ema_48",
    "ema_slope_6",
    "ema_slope_12",
    "ema_slope_24",
    "trend_stack_long",
    "trend_stack_short",
    "trend_strength_12",
    "chop_12",
    "range_expansion",
    "distance_to_rolling_high_12",
    "distance_to_rolling_low_12",
    "consecutive_green_count",
    "consecutive_red_count",
)
LONG_V21_FEATURE_NAMES = (
    *LONG_SPECIALIST_FEATURE_NAMES,
    *(f"15m_{name}" for name in MULTITIMEFRAME_LOCAL_FEATURES),
    *(f"1h_{name}" for name in MULTITIMEFRAME_LOCAL_FEATURES),
)


class LongV21ShadowError(ValueError):
    pass


@dataclass(frozen=True)
class AtrPathContract:
    favorable_atr_multiple: float
    adverse_atr_multiple: float
    favorable_floor_fraction: float
    adverse_floor_fraction: float
    favorable_ceiling_fraction: float
    adverse_ceiling_fraction: float
    fast_success_bars: int
    round_trip_cost_fraction: float

    def __post_init__(self) -> None:
        values = (
            self.favorable_atr_multiple,
            self.adverse_atr_multiple,
            self.favorable_floor_fraction,
            self.adverse_floor_fraction,
            self.favorable_ceiling_fraction,
            self.adverse_ceiling_fraction,
            self.round_trip_cost_fraction,
        )
        if (
            not all(math.isfinite(value) and value >= 0.0 for value in values)
            or self.fast_success_bars <= 0
            or self.favorable_floor_fraction > self.favorable_ceiling_fraction
            or self.adverse_floor_fraction > self.adverse_ceiling_fraction
        ):
            raise LongV21ShadowError("LONG v2.1 ATR contract is invalid")


def aggregate_causal_candles(
    candles: Sequence[Candle], factor: int, output_bars: int = 48
) -> tuple[Candle, ...]:
    """Build closed rolling higher-timeframe bars ending at the signal close."""

    required = factor * output_bars
    if factor <= 1 or output_bars < 48 or len(candles) < required:
        raise LongV21ShadowError("insufficient higher-timeframe history")
    selected = candles[-required:]
    result = []
    for offset in range(0, required, factor):
        group = selected[offset : offset + factor]
        for previous, current in zip(group, group[1:]):
            if previous.close_time != current.open_time:
                raise LongV21ShadowError("higher-timeframe source contains a gap")
        result.append(
            Candle(
                open_time=group[0].open_time,
                close_time=group[-1].close_time,
                open=group[0].open,
                high=max(candle.high for candle in group),
                low=min(candle.low for candle in group),
                close=group[-1].close,
                volume=math.fsum(candle.volume for candle in group),
                is_closed=all(candle.is_closed for candle in group),
                source="CAUSAL_ROLLING_AGGREGATION",
            )
        )
    return tuple(result)


def multitimeframe_long_features(
    base_features: Mapping[str, Any],
    history: Sequence[Candle],
    *,
    pipeline: DeterministicFeaturePipeline | None = None,
) -> tuple[tuple[float, ...], Mapping[str, float]]:
    """Combine current 5m features with causal rolling 15m and 1h context."""

    feature_pipeline = pipeline or DeterministicFeaturePipeline()
    context: dict[str, float] = {}
    for prefix, factor in (("15m", 3), ("1h", 12)):
        aggregated = aggregate_causal_candles(history, factor)
        local = feature_pipeline._local_features(aggregated)
        for name in MULTITIMEFRAME_LOCAL_FEATURES:
            value = float(local[name])
            if not math.isfinite(value):
                raise LongV21ShadowError("non-finite higher-timeframe feature")
            context[f"{prefix}_{name}"] = value
    try:
        base = tuple(
            float(base_features[name]) for name in LONG_SPECIALIST_FEATURE_NAMES
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise LongV21ShadowError("base LONG features are incomplete") from exc
    vector = (*base, *(context[name] for name in LONG_V21_FEATURE_NAMES[len(base) :]))
    if len(vector) != len(LONG_V21_FEATURE_NAMES) or not all(
        math.isfinite(value) for value in vector
    ):
        raise LongV21ShadowError("LONG v2.1 feature vector is invalid")
    return vector, context


def factorized_regime(
    base_features: Mapping[str, Any], context: Mapping[str, float]
) -> Mapping[str, str]:
    """Return separate causal direction, volatility, and structure axes."""

    hourly_return = float(context["1h_ret_3"])
    hourly_long = float(context["1h_trend_stack_long"]) > 0.5
    hourly_short = float(context["1h_trend_stack_short"]) > 0.5
    breadth = float(base_features["market_breadth_6"])
    if hourly_long and hourly_return > 0.0 and breadth >= 0.50:
        direction = "BULLISH"
    elif hourly_short and hourly_return < 0.0 and breadth <= 0.50:
        direction = "BEARISH"
    else:
        direction = "NEUTRAL"

    ratio = float(context["15m_volatility_ratio_6_24"])
    if float(base_features["high_vol_regime_proxy"]) > 0.5 or ratio >= 1.25:
        volatility = "HIGH"
    elif float(base_features["low_vol_regime_proxy"]) > 0.5 and ratio <= 0.80:
        volatility = "LOW"
    else:
        volatility = "NORMAL"

    hourly_chop = float(context["1h_chop_12"])
    hourly_strength = float(context["1h_trend_strength_12"])
    if (hourly_long or hourly_short) and hourly_strength >= 1.5 and hourly_chop <= 0.60:
        structure = "TREND"
    elif hourly_chop >= 0.70:
        structure = "RANGE"
    else:
        structure = "TRANSITION"
    return {
        "direction": direction,
        "volatility": volatility,
        "structure": structure,
        "identity": f"{direction}::{volatility}::{structure}",
    }


def atr_normalized_long_outcome(
    *,
    signal: Candle,
    future: Sequence[Candle],
    atr_fraction: float,
    contract: AtrPathContract,
) -> Mapping[str, Any]:
    """Label target/stop order with symbol-relative ATR barriers."""

    if (
        not future
        or not math.isfinite(atr_fraction)
        or atr_fraction <= 0.0
        or signal.close_time != future[0].open_time
    ):
        raise LongV21ShadowError("LONG v2.1 outcome path is invalid")
    entry = future[0].open
    favorable = min(
        contract.favorable_ceiling_fraction,
        max(
            contract.favorable_floor_fraction,
            atr_fraction * contract.favorable_atr_multiple,
        ),
    )
    adverse = min(
        contract.adverse_ceiling_fraction,
        max(
            contract.adverse_floor_fraction,
            atr_fraction * contract.adverse_atr_multiple,
        ),
    )
    favorable_path = [max(0.0, candle.high / entry - 1.0) for candle in future]
    adverse_path = [max(0.0, 1.0 - candle.low / entry) for candle in future]
    favorable_bar = next(
        (
            index
            for index, value in enumerate(favorable_path, start=1)
            if value >= favorable
        ),
        None,
    )
    adverse_bar = next(
        (
            index
            for index, value in enumerate(adverse_path, start=1)
            if value >= adverse
        ),
        None,
    )
    ambiguous = favorable_bar is not None and favorable_bar == adverse_bar
    if ambiguous:
        order = "SAME_BAR_AMBIGUOUS"
    elif favorable_bar is not None and (
        adverse_bar is None or favorable_bar < adverse_bar
    ):
        order = "FAVORABLE_FIRST"
    elif adverse_bar is not None:
        order = "ADVERSE_FIRST"
    else:
        order = "NEITHER_REACHED"
    mfe = max(favorable_path)
    mae = max(adverse_path)
    return {
        "entry_price": entry,
        "atr_fraction": atr_fraction,
        "favorable_barrier_fraction": favorable,
        "adverse_barrier_fraction": adverse,
        "first_favorable_bar": favorable_bar,
        "first_adverse_bar": adverse_bar,
        "barrier_order": order,
        "same_bar_ambiguity": ambiguous,
        "target_before_stop": order == "FAVORABLE_FIRST",
        "clean_fast_success": order == "FAVORABLE_FIRST"
        and favorable_bar <= contract.fast_success_bars,
        "mfe_fraction": mfe,
        "mae_fraction": mae,
        "time_underwater_bars": sum(candle.close < entry for candle in future),
        "terminal_return_after_costs": future[-1].close / entry
        - 1.0
        - contract.round_trip_cost_fraction,
    }


def _canonical(candles: Sequence[Candle]) -> tuple[CanonicalBar, ...]:
    return tuple(
        CanonicalBar(
            candle.open_time,
            candle.open,
            candle.high,
            candle.low,
            candle.close,
            candle.volume,
        )
        for candle in candles
    )


def protected_long_utility(
    *,
    history: Sequence[Candle],
    future: Sequence[Candle],
    outcome: Mapping[str, Any],
    protection: TsProtectionConfig,
    mae_penalty_weight: float,
    underwater_bar_penalty_fraction: float,
    catastrophic_mae_atr_multiple: float,
) -> Mapping[str, Any]:
    """Use the pessimistic intrabar TypeScript replay as the utility target."""

    if (
        min(
            mae_penalty_weight,
            underwater_bar_penalty_fraction,
            catastrophic_mae_atr_multiple,
        )
        < 0.0
    ):
        raise LongV21ShadowError("LONG v2.1 utility contract is invalid")
    results = {
        path.value: replay_ts_price_protection(
            side=DirectionalSide.LONG,
            history=_canonical(history),
            future=_canonical(future),
            path=path,
            config=protection,
        )
        for path in IntrabarPath
    }
    net_values = [value.net_return_after_costs for value in results.values()]
    worst_net = min(net_values)
    best_net = max(net_values)
    mae = float(outcome["mae_fraction"])
    underwater = int(outcome["time_underwater_bars"])
    utility = (
        worst_net
        - mae_penalty_weight * mae
        - underwater_bar_penalty_fraction * underwater
    )
    catastrophic = mae >= (
        float(outcome["atr_fraction"]) * catastrophic_mae_atr_multiple
    )
    return {
        "protected_worst_net_return": worst_net,
        "protected_best_net_return": best_net,
        "protected_path_spread": best_net - worst_net,
        "utility_target": utility,
        "catastrophic_path": catastrophic,
        "protection_results": {
            path: {
                "net_return_after_costs": result.net_return_after_costs,
                "exit_reason": result.exit_reason.value,
                "bars_held": result.bars_held,
                "break_even_armed": result.break_even_armed,
                "trailing_armed": result.trailing_armed,
            }
            for path, result in results.items()
        },
        "selection_effect": "NONE",
        "exchange_authority": False,
        "exchange_mutations": 0,
    }
