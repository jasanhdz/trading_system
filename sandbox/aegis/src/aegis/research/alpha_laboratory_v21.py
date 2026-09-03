"""Preregistered simple-rule alpha laboratory for V21 research."""

from __future__ import annotations

import math
import random
from collections import defaultdict
from datetime import datetime
from enum import Enum
from typing import Any, Iterable, Mapping, Sequence

from .causal_opportunity_v20 import economic_summary, side_adjusted_flow, temporal_thirds
from .decomposed_entry_v9 import V9_FEATURE_NAMES


class AlphaLaboratoryV21Error(ValueError):
    """Raised when V21 input evidence violates the preregistered contract."""


class AlphaStrategy(str, Enum):
    CROSS_SECTIONAL_MOMENTUM = "CROSS_SECTIONAL_MOMENTUM"
    EXTREME_REVERSAL = "EXTREME_REVERSAL"
    BREAKOUT_FLOW_FUNDING = "BREAKOUT_FLOW_FUNDING"
    FUNDING_BASIS_CARRY = "FUNDING_BASIS_CARRY"


def timestamp_ms(value: str) -> int:
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as exc:
        raise AlphaLaboratoryV21Error("invalid V21 timestamp") from exc
    if parsed.tzinfo is None:
        raise AlphaLaboratoryV21Error("V21 timestamp must be timezone-aware")
    return int(parsed.timestamp() * 1000)


def _finite(value: Any, name: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise AlphaLaboratoryV21Error(f"non-finite V21 value: {name}")
    return result


def named_v9(values: Sequence[Any]) -> dict[str, float]:
    if len(values) != len(V9_FEATURE_NAMES):
        raise AlphaLaboratoryV21Error("invalid V21 V9 feature count")
    result: dict[str, float] = {}
    for name, raw in zip(V9_FEATURE_NAMES, values, strict=True):
        value = _finite(raw, name)
        if name in result and not math.isclose(result[name], value, abs_tol=1e-12):
            raise AlphaLaboratoryV21Error(f"inconsistent duplicated V9 feature: {name}")
        result[name] = value
    return result


def prepare_row(
    source: Mapping[str, Any],
    *,
    funding_rate: float | None,
    funding_age_ms: int | None,
) -> dict[str, Any]:
    side = str(source["side"])
    if side not in {"LONG", "SHORT"}:
        raise AlphaLaboratoryV21Error("invalid V21 side")
    features = named_v9(source["v9_features"])
    flow = side_adjusted_flow(source)
    sign = 1.0 if side == "LONG" else -1.0
    return {
        "source": source,
        "timestamp": str(source["timestamp"]),
        "timestamp_ms": timestamp_ms(str(source["timestamp"])),
        "symbol": str(source["symbol"]),
        "side": side,
        "features": features,
        "flow": flow,
        "funding_rate": funding_rate,
        "funding_age_ms": funding_age_ms,
        "side_adjusted_funding_rate": (
            sign * _finite(funding_rate, "funding_rate")
            if funding_rate is not None
            else None
        ),
    }


def classify_timestamp(
    rows: Sequence[Mapping[str, Any]], config: Mapping[str, Any]
) -> dict[int, tuple[AlphaStrategy, ...]]:
    """Classify one timestamp using only causal row values and cross-sectional ranks."""

    if not rows:
        return {}
    if len({str(row["timestamp"]) for row in rows}) != 1:
        raise AlphaLaboratoryV21Error("V21 timestamp group is inconsistent")
    strategies = config["strategies"]
    result: dict[int, list[AlphaStrategy]] = defaultdict(list)
    for side in ("LONG", "SHORT"):
        side_rows = [(index, row) for index, row in enumerate(rows) if row["side"] == side]
        if not side_rows:
            continue
        if len({str(row["symbol"]) for _, row in side_rows}) != len(side_rows):
            raise AlphaLaboratoryV21Error("duplicate V21 symbol/side at timestamp")

        momentum_config = strategies[AlphaStrategy.CROSS_SECTIONAL_MOMENTUM.value]
        momentum_count = min(int(momentum_config["selected_rank_per_timestamp"]), len(side_rows))
        momentum_ranks = {
            index
            for index, _ in sorted(
                side_rows,
                key=lambda item: (
                    -float(item[1]["features"]["side_rolling_4h_return"]),
                    str(item[1]["symbol"]),
                ),
            )[:momentum_count]
        }
        for index, row in side_rows:
            features, flow = row["features"], row["flow"]
            if (
                index in momentum_ranks
                and features["side_rolling_4h_return"]
                > float(momentum_config["side_rolling_4h_return_min_exclusive"])
                and features["side_ret_3"]
                > float(momentum_config["side_ret_3_min_exclusive"])
                and features["volume_ratio_6_24"]
                >= float(momentum_config["volume_ratio_6_24_min"])
                and flow["side_taker_imbalance_6"]
                > float(momentum_config["side_taker_imbalance_6_min_exclusive"])
            ):
                result[index].append(AlphaStrategy.CROSS_SECTIONAL_MOMENTUM)

        reversal_config = strategies[AlphaStrategy.EXTREME_REVERSAL.value]
        reversal_count = min(int(reversal_config["adverse_rank_per_timestamp"]), len(side_rows))
        reversal_ranks = {
            index
            for index, _ in sorted(
                side_rows,
                key=lambda item: (
                    float(item[1]["features"]["side_rolling_4h_return"]),
                    str(item[1]["symbol"]),
                ),
            )[:reversal_count]
        }
        for index, row in side_rows:
            features, flow = row["features"], row["flow"]
            wick_passes = (
                features["favorable_wick_fraction"] > features["adverse_wick_fraction"]
                if bool(reversal_config["favorable_wick_must_exceed_adverse"])
                else True
            )
            if (
                index in reversal_ranks
                and features["side_rolling_4h_return"]
                < float(reversal_config["side_rolling_4h_return_max_exclusive"])
                and features["side_ret_1"]
                > float(reversal_config["side_ret_1_min_exclusive"])
                and features["side_acceleration"]
                > float(reversal_config["side_acceleration_min_exclusive"])
                and wick_passes
                and flow["side_taker_imbalance_3"]
                > float(reversal_config["side_taker_imbalance_3_min_exclusive"])
                and flow["side_taker_acceleration_3_12"]
                > float(reversal_config["side_taker_acceleration_min_exclusive"])
            ):
                result[index].append(AlphaStrategy.EXTREME_REVERSAL)

        breakout_config = strategies[AlphaStrategy.BREAKOUT_FLOW_FUNDING.value]
        maximum_age_ms = int(float(breakout_config["maximum_funding_age_hours"]) * 3_600_000)
        for index, row in side_rows:
            features, flow = row["features"], row["flow"]
            funding = row["side_adjusted_funding_rate"]
            funding_age = row["funding_age_ms"]
            if (
                funding is not None
                and funding_age is not None
                and 0 <= int(funding_age) <= maximum_age_ms
                and float(funding)
                <= float(breakout_config["maximum_side_adjusted_funding_rate"])
                and features["soft_archetype_BREAKOUT"]
                >= float(breakout_config["soft_breakout_min"])
                and features["side_ret_3"]
                > float(breakout_config["side_ret_3_min_exclusive"])
                and features["favorable_close_location"]
                >= float(breakout_config["favorable_close_location_min"])
                and features["volume_ratio_6_24"]
                >= float(breakout_config["volume_ratio_6_24_min"])
                and features["range_expansion"]
                >= float(breakout_config["range_expansion_min"])
                and flow["side_taker_imbalance_6"]
                > float(breakout_config["side_taker_imbalance_6_min_exclusive"])
                and flow["side_taker_acceleration_3_12"]
                > float(breakout_config["side_taker_acceleration_min_exclusive"])
            ):
                result[index].append(AlphaStrategy.BREAKOUT_FLOW_FUNDING)
    return {index: tuple(values) for index, values in result.items()}


def partition_name(timestamp: str, protocol: Mapping[str, Any]) -> str:
    if protocol["discovery"]["start_inclusive"] <= timestamp < protocol["discovery"]["end_exclusive"]:
        return "DISCOVERY"
    if protocol["validation"]["start_inclusive"] <= timestamp < protocol["validation"]["end_exclusive"]:
        return "VALIDATION"
    if protocol["final_holdout"]["start_inclusive"] <= timestamp <= protocol["final_holdout"]["end_inclusive"]:
        return "FINAL_HOLDOUT"
    raise AlphaLaboratoryV21Error("V21 row falls outside frozen temporal protocol")


def candidate_record(row: Mapping[str, Any], strategy: AlphaStrategy, config: Mapping[str, Any]) -> dict[str, Any]:
    source = row["source"]
    profile_name = str(config["strategies"][strategy.value]["exit_profile"])
    profile = source["protection_profiles"][profile_name]
    contract = source["v10_contract_outcomes"]["ROE_10_H12"]
    return {
        "schema_id": "aegis-alpha-laboratory-v21-row-v1",
        "timestamp": row["timestamp"],
        "timestamp_ms": row["timestamp_ms"],
        "partition": partition_name(row["timestamp"], config["temporal_protocol"]),
        "symbol": row["symbol"],
        "side": row["side"],
        "strategy": strategy.value,
        "exit_profile": profile_name,
        "entry_price": _finite(source["entry_price"], "entry_price"),
        "funding_rate": row["funding_rate"],
        "funding_age_ms": row["funding_age_ms"],
        "protected_net_return": _finite(profile["worst_net_return"], "protected_net_return"),
        "protected_exit_reason": str(profile["worst_exit_reason"]),
        "protected_bars_held": int(profile["worst_bars_held"]),
        "break_even_armed": bool(profile["break_even_armed"]),
        "trailing_armed": bool(profile["trailing_armed"]),
        "intrabar_path_spread": _finite(profile["path_spread"], "path_spread"),
        "current_ts_net_return": _finite(
            source["protection_profiles"]["CURRENT_TS"]["worst_net_return"],
            "current_ts_net_return",
        ),
        "contract_utility": _finite(contract["realized_utility"], "contract_utility"),
        "mae_fraction": _finite(source["mae_fraction"], "mae_fraction"),
        "mfe_fraction": _finite(source["mfe_fraction"], "mfe_fraction"),
        "time_underwater_bars": _finite(source["time_underwater_bars"], "time_underwater_bars"),
        "selection_effect": "NONE",
        "exchange_authority": False,
        "exchange_calls": 0,
        "exchange_mutations": 0,
    }


def apply_event_spacing(
    rows: Iterable[Mapping[str, Any]], minimum_minutes: int
) -> list[Mapping[str, Any]]:
    minimum_ms = minimum_minutes * 60_000
    latest: dict[tuple[str, str, str], int] = {}
    accepted: list[Mapping[str, Any]] = []
    for row in sorted(rows, key=lambda value: (int(value["timestamp_ms"]), str(value["symbol"]), str(value["side"]), str(value["strategy"]))):
        key = (str(row["symbol"]), str(row["side"]), str(row["strategy"]))
        current = int(row["timestamp_ms"])
        if key not in latest or current - latest[key] >= minimum_ms:
            accepted.append(row)
            latest[key] = current
    return accepted


def matched_control(
    population: Sequence[Mapping[str, Any]], count: int, *, seed: int
) -> list[Mapping[str, Any]]:
    if count > len(population):
        raise AlphaLaboratoryV21Error("V21 control exceeds population")
    return random.Random(seed).sample(list(population), count)


def gate_assessment(
    periods: Mapping[str, Sequence[Mapping[str, Any]]],
    random_holdout: Sequence[Mapping[str, Any]],
    gate: Mapping[str, Any],
) -> dict[str, Any]:
    discovery = economic_summary(periods["DISCOVERY"])
    validation = economic_summary(periods["VALIDATION"])
    holdout = economic_summary(periods["FINAL_HOLDOUT"])
    random_metrics = economic_summary(random_holdout)
    holdout_profit_factor = holdout.get("profit_factor")
    no_holdout_losses = bool(periods["FINAL_HOLDOUT"]) and all(
        float(row["protected_net_return"]) >= 0.0
        for row in periods["FINAL_HOLDOUT"]
    )
    positive_thirds = sum(
        float(block.get("mean_protected_net", 0.0)) > 0.0
        for block in temporal_thirds(periods["FINAL_HOLDOUT"])
    )
    checks = {
        "minimum_discovery_events": len(periods["DISCOVERY"]) >= int(gate["minimum_discovery_events"]),
        "minimum_validation_events": len(periods["VALIDATION"]) >= int(gate["minimum_validation_events"]),
        "minimum_holdout_events": len(periods["FINAL_HOLDOUT"]) >= int(gate["minimum_holdout_events"]),
        "positive_validation_mean_net": float(validation.get("mean_protected_net", -math.inf)) > 0.0,
        "positive_holdout_mean_net": float(holdout.get("mean_protected_net", -math.inf)) > 0.0,
        "positive_holdout_utility": float(holdout.get("mean_contract_utility", -math.inf)) > 0.0,
        "holdout_profit_factor": no_holdout_losses
        or float(holdout_profit_factor or 0.0)
        >= float(gate["minimum_holdout_profit_factor"]),
        "holdout_mean_mae": float(holdout.get("mean_mae", math.inf)) <= float(gate["maximum_holdout_mean_mae"]),
        "positive_holdout_temporal_thirds": positive_thirds >= int(gate["minimum_positive_holdout_temporal_thirds"]),
        "holdout_outperforms_no_trade": float(holdout.get("mean_protected_net", -math.inf)) > 0.0,
        "holdout_outperforms_random": float(holdout.get("mean_protected_net", -math.inf)) > float(random_metrics.get("mean_protected_net", math.inf)),
        "validation_and_holdout_same_sign": float(validation.get("mean_protected_net", 0.0)) * float(holdout.get("mean_protected_net", 0.0)) > 0.0,
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "positive_holdout_temporal_thirds": positive_thirds,
        "periods": {
            "DISCOVERY": discovery,
            "VALIDATION": validation,
            "FINAL_HOLDOUT": holdout,
        },
        "final_holdout_temporal_thirds": temporal_thirds(periods["FINAL_HOLDOUT"]),
        "random_holdout_control": random_metrics,
    }
