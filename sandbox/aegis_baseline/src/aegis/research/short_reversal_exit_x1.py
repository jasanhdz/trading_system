"""Economic compatibility checks for the preregistered SHORT reversal X1."""

from __future__ import annotations

import math
import random
from collections import defaultdict
from typing import Any, Mapping, Sequence

import numpy as np

from .causal_opportunity_v20 import economic_summary, temporal_thirds


class ShortReversalExitX1Error(ValueError):
    """Raised when X1 evidence violates its frozen contract."""


def _finite(value: Any, name: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ShortReversalExitX1Error(f"non-finite X1 value: {name}")
    return result


def profile_record(prepared: Mapping[str, Any], profile_name: str) -> dict[str, Any]:
    source = prepared["source"]
    if str(source["side"]) != "SHORT":
        raise ShortReversalExitX1Error("X1 accepts SHORT rows only")
    try:
        profile = source["protection_profiles"][profile_name]
        contract = source["v10_contract_outcomes"]["ROE_10_H12"]
    except (KeyError, TypeError) as exc:
        raise ShortReversalExitX1Error("X1 protection outcome is incomplete") from exc
    return {
        "schema_id": "aegis-short-reversal-exit-x1-row-v1",
        "timestamp": str(prepared["timestamp"]),
        "timestamp_ms": int(prepared["timestamp_ms"]),
        "symbol": str(prepared["symbol"]),
        "side": "SHORT",
        "strategy": "EXTREME_REVERSAL",
        "profile": profile_name,
        "protected_net_return": _finite(profile["worst_net_return"], "protected_net"),
        "protected_exit_reason": str(profile["worst_exit_reason"]),
        "protected_bars_held": int(profile["worst_bars_held"]),
        "break_even_armed": bool(profile["break_even_armed"]),
        "trailing_armed": bool(profile["trailing_armed"]),
        "intrabar_path_spread": _finite(profile["path_spread"], "path_spread"),
        "contract_utility": _finite(contract["realized_utility"], "contract_utility"),
        "mae_fraction": _finite(source["mae_fraction"], "mae"),
        "mfe_fraction": _finite(source["mfe_fraction"], "mfe"),
        "time_underwater_bars": _finite(source["time_underwater_bars"], "underwater"),
        "selection_effect": "NONE",
        "exchange_authority": False,
        "exchange_calls": 0,
        "exchange_mutations": 0,
    }


def daily_block_bootstrap_interval(
    rows: Sequence[Mapping[str, Any]], *, seed: int, resamples: int
) -> Mapping[str, float] | None:
    grouped: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        grouped[str(row["timestamp"])[:10]].append(
            _finite(row["protected_net_return"], "protected_net_return")
        )
    days = sorted(grouped)
    if len(days) < 3 or resamples <= 0:
        return None
    generator = random.Random(seed)
    estimates = []
    for _ in range(resamples):
        sampled_days = [generator.choice(days) for _ in days]
        values = [value for day in sampled_days for value in grouped[day]]
        estimates.append(float(np.mean(values)))
    return {
        "lower_95": float(np.quantile(estimates, 0.025)),
        "median": float(np.quantile(estimates, 0.5)),
        "upper_95": float(np.quantile(estimates, 0.975)),
        "utc_day_blocks": len(days),
        "resamples": resamples,
    }


def cost_stress(
    rows: Sequence[Mapping[str, Any]], additional_cost: float
) -> list[Mapping[str, Any]]:
    if not math.isfinite(additional_cost) or additional_cost < 0.0:
        raise ShortReversalExitX1Error("invalid X1 cost stress")
    return [
        {
            **row,
            "protected_net_return": _finite(
                row["protected_net_return"], "protected_net_return"
            )
            - additional_cost,
        }
        for row in rows
    ]


def assessment(
    *,
    candidate: Sequence[Mapping[str, Any]],
    v21_exit: Sequence[Mapping[str, Any]],
    random_control: Sequence[Mapping[str, Any]],
    diagnostic_profiles: Mapping[str, Sequence[Mapping[str, Any]]],
    config: Mapping[str, Any],
) -> Mapping[str, Any]:
    if not (len(candidate) == len(v21_exit)):
        raise ShortReversalExitX1Error("X1 same-event controls are misaligned")
    candidate_metrics = economic_summary(candidate)
    v21_metrics = economic_summary(v21_exit)
    random_metrics = economic_summary(random_control)
    stress_levels = [float(value) for value in config["cost_contract"]["stress_additional_round_trip_cost_fraction"]]
    stress_metrics = {
        str(level): economic_summary(cost_stress(candidate, level))
        for level in stress_levels
    }
    primary_stress = str(float(config["cost_contract"]["primary_stress_level"]))
    thirds = temporal_thirds(candidate)
    positive_thirds = sum(
        float(block.get("mean_protected_net", 0.0)) > 0.0 for block in thirds
    )
    uncertainty = daily_block_bootstrap_interval(
        candidate,
        seed=int(config["uncertainty"]["seed"]),
        resamples=int(config["uncertainty"]["bootstrap_resamples"]),
    )
    profit_factor = candidate_metrics.get("profit_factor")
    no_losses = bool(candidate) and all(
        float(row["protected_net_return"]) >= 0.0 for row in candidate
    )
    gate = config["gate"]
    checks = {
        "minimum_candidate_events": len(candidate)
        >= int(gate["minimum_candidate_events"]),
        "positive_mean_current_ts_net": float(
            candidate_metrics.get("mean_protected_net", -math.inf)
        )
        > 0.0,
        "positive_mean_at_primary_cost_stress": float(
            stress_metrics[primary_stress].get("mean_protected_net", -math.inf)
        )
        > 0.0,
        "minimum_profit_factor": no_losses
        or float(profit_factor or 0.0) >= float(gate["minimum_profit_factor"]),
        "maximum_mean_mae": float(candidate_metrics.get("mean_mae", math.inf))
        <= float(gate["maximum_mean_mae"]),
        "positive_temporal_thirds": positive_thirds
        >= int(gate["minimum_positive_temporal_thirds"]),
        "outperformance_of_v21_exit": float(
            candidate_metrics.get("mean_protected_net", -math.inf)
        )
        > float(v21_metrics.get("mean_protected_net", math.inf)),
        "outperformance_of_random_control": float(
            candidate_metrics.get("mean_protected_net", -math.inf)
        )
        > float(random_metrics.get("mean_protected_net", math.inf)),
        "lower_daily_block_bootstrap_bound_non_negative": uncertainty is not None
        and float(uncertainty["lower_95"]) >= 0.0,
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "candidate_current_ts": candidate_metrics,
        "v21_lock_at_5_same_events": v21_metrics,
        "random_short_matched_count": random_metrics,
        "cost_stress": stress_metrics,
        "daily_block_bootstrap": uncertainty,
        "temporal_thirds": thirds,
        "positive_temporal_thirds": positive_thirds,
        "diagnostic_profiles_not_eligible_for_selection": {
            name: economic_summary(rows)
            for name, rows in sorted(diagnostic_profiles.items())
        },
    }

