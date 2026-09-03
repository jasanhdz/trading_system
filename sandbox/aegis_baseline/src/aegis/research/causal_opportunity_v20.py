"""Causal opportunity families and economic feasibility for V20 research."""

from __future__ import annotations

import math
import random
from collections import defaultdict
from enum import Enum
from statistics import mean
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from .decomposed_entry_v9 import V9_FEATURE_NAMES
from .feature_information_v14 import TAKER_FLOW_FEATURE_NAMES


class OpportunityV20Error(ValueError):
    """Raised when V20 source evidence violates its frozen contract."""


class OpportunityFamily(str, Enum):
    TREND_CONTINUATION = "TREND_CONTINUATION"
    BREAKOUT_EXPANSION = "BREAKOUT_EXPANSION"
    PULLBACK_RECLAIM = "PULLBACK_RECLAIM"
    CONFIRMED_REVERSAL = "CONFIRMED_REVERSAL"
    VOLATILITY_EXPANSION = "VOLATILITY_EXPANSION"


def _finite(value: Any, name: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise OpportunityV20Error(f"non-finite V20 value: {name}")
    return result


def _named(values: Sequence[Any], names: Sequence[str], label: str) -> dict[str, float]:
    if len(values) != len(names):
        raise OpportunityV20Error(f"invalid V20 {label} feature count")
    result = {name: _finite(value, name) for name, value in zip(names, values, strict=True)}
    if len(result) != len(names):
        raise OpportunityV20Error(f"duplicate V20 {label} feature name")
    return result


def _named_v9(values: Sequence[Any]) -> dict[str, float]:
    if len(values) != len(V9_FEATURE_NAMES):
        raise OpportunityV20Error("invalid V20 V9 feature count")
    result: dict[str, float] = {}
    for name, raw in zip(V9_FEATURE_NAMES, values, strict=True):
        value = _finite(raw, name)
        if name in result and not math.isclose(result[name], value, abs_tol=1e-12):
            raise OpportunityV20Error(f"inconsistent duplicated V9 feature: {name}")
        result[name] = value
    return result


def side_adjusted_flow(source: Mapping[str, Any]) -> dict[str, float]:
    names = tuple(str(name) for name in source["v14_taker_flow_feature_names"])
    if names != tuple(TAKER_FLOW_FEATURE_NAMES):
        raise OpportunityV20Error("V20 taker-flow order mismatch")
    raw = _named(source["v14_taker_flow_features"], names, "taker-flow")
    side = str(source["side"])
    if side not in {"LONG", "SHORT"}:
        raise OpportunityV20Error("invalid V20 side")
    sign = 1.0 if side == "LONG" else -1.0
    breadth = raw["market_taker_breadth_6"]
    if not 0.0 <= breadth <= 1.0:
        raise OpportunityV20Error("invalid market taker breadth")
    return {
        "side_taker_imbalance_1": sign * raw["taker_imbalance_1"],
        "side_taker_imbalance_3": sign * raw["taker_imbalance_3"],
        "side_taker_imbalance_6": sign * raw["taker_imbalance_6"],
        "side_taker_imbalance_12": sign * raw["taker_imbalance_12"],
        "side_taker_imbalance_24": sign * raw["taker_imbalance_24"],
        "side_taker_acceleration_3_12": sign
        * raw["taker_imbalance_acceleration_3_12"],
        "side_market_taker_imbalance_6": sign * raw["market_taker_imbalance_6"],
        "side_market_taker_breadth_6": breadth if side == "LONG" else 1.0 - breadth,
        "side_btc_taker_imbalance_6": sign * raw["btc_taker_imbalance_6"],
        "side_relative_taker_imbalance_6": sign * raw["relative_taker_imbalance_6"],
    }


def classify_opportunities(
    source: Mapping[str, Any], config: Mapping[str, Any]
) -> tuple[OpportunityFamily, ...]:
    features = _named_v9(source["v9_features"])
    flow = side_adjusted_flow(source)
    families = config["opportunity_families"]
    detected: list[OpportunityFamily] = []

    trend = families[OpportunityFamily.TREND_CONTINUATION.value]
    if (
        features["side_ret_3"] > float(trend["side_ret_3_min_exclusive"])
        and features["side_ret_12"] > float(trend["side_ret_12_min_exclusive"])
        and features["side_trend_stack"] > float(trend["side_trend_stack_min_exclusive"])
        and features["timeframe_alignment_score"] >= float(trend["timeframe_alignment_min"])
        and features["volume_ratio_6_24"] >= float(trend["volume_ratio_6_24_min"])
        and flow["side_taker_imbalance_6"]
        > float(trend["side_taker_imbalance_6_min_exclusive"])
    ):
        detected.append(OpportunityFamily.TREND_CONTINUATION)

    breakout = families[OpportunityFamily.BREAKOUT_EXPANSION.value]
    if (
        features["soft_archetype_BREAKOUT"] >= float(breakout["soft_breakout_min"])
        and features["side_ret_3"] > float(breakout["side_ret_3_min_exclusive"])
        and features["favorable_close_location"]
        >= float(breakout["favorable_close_location_min"])
        and features["volume_ratio_6_24"] >= float(breakout["volume_ratio_6_24_min"])
        and features["range_expansion"] >= float(breakout["range_expansion_min"])
        and flow["side_taker_imbalance_6"]
        > float(breakout["side_taker_imbalance_6_min_exclusive"])
        and flow["side_taker_acceleration_3_12"]
        > float(breakout["side_taker_acceleration_min_exclusive"])
    ):
        detected.append(OpportunityFamily.BREAKOUT_EXPANSION)

    pullback = families[OpportunityFamily.PULLBACK_RECLAIM.value]
    if (
        features["regime_phase_PULLBACK"] > float(pullback["regime_pullback_min_exclusive"])
        and features["side_ret_12"] > float(pullback["side_ret_12_min_exclusive"])
        and features["side_ret_1"] > float(pullback["side_ret_1_min_exclusive"])
        and features["side_trend_stack"] > float(pullback["side_trend_stack_min_exclusive"])
        and features["favorable_close_location"]
        >= float(pullback["favorable_close_location_min"])
        and flow["side_taker_imbalance_3"]
        > float(pullback["side_taker_imbalance_3_min_exclusive"])
    ):
        detected.append(OpportunityFamily.PULLBACK_RECLAIM)

    reversal = families[OpportunityFamily.CONFIRMED_REVERSAL.value]
    wick_passes = (
        features["favorable_wick_fraction"] > features["adverse_wick_fraction"]
        if bool(reversal["favorable_wick_must_exceed_adverse"])
        else True
    )
    if (
        features["soft_archetype_REVERSAL"] >= float(reversal["soft_reversal_min"])
        and features["side_ret_1"] > float(reversal["side_ret_1_min_exclusive"])
        and features["side_acceleration"] > float(reversal["side_acceleration_min_exclusive"])
        and wick_passes
        and features["timeframe_conflict_score"] <= float(reversal["timeframe_conflict_max"])
        and flow["side_taker_imbalance_3"]
        > float(reversal["side_taker_imbalance_3_min_exclusive"])
        and flow["side_taker_acceleration_3_12"]
        > float(reversal["side_taker_acceleration_min_exclusive"])
    ):
        detected.append(OpportunityFamily.CONFIRMED_REVERSAL)

    expansion = families[OpportunityFamily.VOLATILITY_EXPANSION.value]
    if (
        features["volatility_ratio_6_24"] >= float(expansion["volatility_ratio_6_24_min"])
        and features["range_expansion"] >= float(expansion["range_expansion_min"])
        and features["volume_ratio_6_24"] >= float(expansion["volume_ratio_6_24_min"])
        and features["volume_direction_impulse"]
        > float(expansion["volume_direction_impulse_min_exclusive"])
        and features["side_ret_3"] > float(expansion["side_ret_3_min_exclusive"])
        and flow["side_taker_imbalance_3"]
        > float(expansion["side_taker_imbalance_3_min_exclusive"])
    ):
        detected.append(OpportunityFamily.VOLATILITY_EXPANSION)
    return tuple(detected)


def opportunity_record(source: Mapping[str, Any], family: OpportunityFamily) -> dict[str, Any]:
    protection = source["protection_profiles"]["CURRENT_TS"]
    contract = source["v10_contract_outcomes"]["ROE_10_H12"]
    flow = side_adjusted_flow(source)
    return {
        "schema_id": "aegis-causal-opportunity-v20-row-v1",
        "timestamp": str(source["timestamp"]),
        "symbol": str(source["symbol"]),
        "side": str(source["side"]),
        "family": family.value,
        "regime": str(source["v11_causal_regime"]),
        "entry_price": _finite(source["entry_price"], "entry_price"),
        "features": [*map(float, source["v9_features"]), *flow.values()],
        "feature_schema": "V9_176_PLUS_SIDE_ADJUSTED_V14_TAKER_FLOW_10",
        "protected_net_return": _finite(protection["worst_net_return"], "protected_net"),
        "protected_exit_reason": str(protection["worst_exit_reason"]),
        "protected_bars_held": int(protection["worst_bars_held"]),
        "break_even_armed": bool(protection["break_even_armed"]),
        "trailing_armed": bool(protection["trailing_armed"]),
        "intrabar_path_spread": _finite(protection["path_spread"], "path_spread"),
        "contract_utility": _finite(contract["realized_utility"], "contract_utility"),
        "contract_outcome": str(contract["outcome"]),
        "mae_fraction": _finite(source["mae_fraction"], "mae"),
        "mfe_fraction": _finite(source["mfe_fraction"], "mfe"),
        "time_underwater_bars": _finite(source["time_underwater_bars"], "underwater"),
        "clean_entry": bool(source["v11_clean_entry_label"]),
        "selection_effect": "NONE",
        "exchange_authority": False,
        "exchange_calls": 0,
        "exchange_mutations": 0,
    }


def economic_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {"events": 0}
    returns = np.asarray([_finite(row["protected_net_return"], "return") for row in rows])
    utilities = np.asarray([_finite(row["contract_utility"], "utility") for row in rows])
    mae = np.asarray([_finite(row["mae_fraction"], "mae") for row in rows])
    mfe = np.asarray([_finite(row["mfe_fraction"], "mfe") for row in rows])
    underwater = np.asarray(
        [_finite(row["time_underwater_bars"], "underwater") for row in rows]
    )
    cumulative = np.cumsum(returns)
    peak = np.maximum.accumulate(np.concatenate(([0.0], cumulative)))
    drawdown = np.concatenate(([0.0], cumulative)) - peak
    positive = float(np.sum(returns[returns > 0.0]))
    negative = float(-np.sum(returns[returns < 0.0]))
    tail_count = max(1, math.ceil(len(returns) * 0.05))
    return {
        "events": len(rows),
        "mean_protected_net": float(np.mean(returns)),
        "median_protected_net": float(np.median(returns)),
        "protected_win_rate": float(np.mean(returns > 0.0)),
        "profit_factor": positive / negative if negative > 0.0 else None,
        "cumulative_protected_net": float(np.sum(returns)),
        "maximum_additive_drawdown": float(np.min(drawdown)),
        "cvar_05": float(np.mean(np.sort(returns)[:tail_count])),
        "mean_contract_utility": float(np.mean(utilities)),
        "mean_mae": float(np.mean(mae)),
        "p90_mae": float(np.quantile(mae, 0.90)),
        "mean_mfe": float(np.mean(mfe)),
        "mean_time_underwater_bars": float(np.mean(underwater)),
        "break_even_armed_rate": mean(bool(row["break_even_armed"]) for row in rows),
        "trailing_armed_rate": mean(bool(row["trailing_armed"]) for row in rows),
    }


def temporal_thirds(rows: Sequence[Mapping[str, Any]]) -> tuple[dict[str, Any], ...]:
    if not rows:
        return ()
    ordered = sorted(rows, key=lambda row: (str(row["timestamp"]), str(row["symbol"])))
    timestamps = sorted({str(row["timestamp"]) for row in ordered})
    boundaries = (timestamps[len(timestamps) // 3], timestamps[(2 * len(timestamps)) // 3])
    blocks = (
        [row for row in ordered if str(row["timestamp"]) < boundaries[0]],
        [row for row in ordered if boundaries[0] <= str(row["timestamp"]) < boundaries[1]],
        [row for row in ordered if str(row["timestamp"]) >= boundaries[1]],
    )
    return tuple(economic_summary(block) for block in blocks)


def monthly_bootstrap_mean_interval(
    rows: Sequence[Mapping[str, Any]], *, seed: int, resamples: int = 1000
) -> dict[str, float] | None:
    grouped: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        grouped[str(row["timestamp"])[:7]].append(float(row["protected_net_return"]))
    months = sorted(grouped)
    if len(months) < 3:
        return None
    rng = random.Random(seed)
    estimates = []
    for _ in range(resamples):
        sampled = [rng.choice(months) for _ in months]
        values = [value for month in sampled for value in grouped[month]]
        estimates.append(mean(values))
    return {
        "lower_95": float(np.quantile(estimates, 0.025)),
        "upper_95": float(np.quantile(estimates, 0.975)),
    }


def viability(
    rows: Sequence[Mapping[str, Any]], gate: Mapping[str, Any]
) -> dict[str, Any]:
    metrics = economic_summary(rows)
    blocks = temporal_thirds(rows)
    positive_blocks = sum(
        float(block.get("mean_protected_net", 0.0)) > 0.0 for block in blocks
    )
    checks = {
        "minimum_events": len(rows) >= int(gate["minimum_events_per_side_family"]),
        "positive_mean_protected_net": float(metrics.get("mean_protected_net", -math.inf))
        > 0.0,
        "positive_temporal_blocks": positive_blocks
        >= int(gate["require_positive_temporal_blocks"]),
        "positive_utility_mean": float(metrics.get("mean_contract_utility", -math.inf))
        > 0.0,
        "mean_mae": float(metrics.get("mean_mae", math.inf))
        <= float(gate["maximum_mean_mae"]),
        "protected_win_rate": float(metrics.get("protected_win_rate", 0.0))
        >= float(gate["minimum_protected_win_rate"]),
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "positive_temporal_blocks": positive_blocks,
        "metrics": metrics,
        "temporal_thirds": blocks,
    }


def matched_random_control(
    population: Sequence[Mapping[str, Any]], count: int, *, seed: int
) -> list[Mapping[str, Any]]:
    if count > len(population):
        raise OpportunityV20Error("random control exceeds population")
    return random.Random(seed).sample(list(population), count)


def group_rows(
    rows: Iterable[Mapping[str, Any]], *fields: str
) -> dict[tuple[str, ...], list[Mapping[str, Any]]]:
    result: dict[tuple[str, ...], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        result[tuple(str(row[field]) for field in fields)].append(row)
    return result
