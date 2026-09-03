"""Causal multi-objective LONG entry contracts for the v5 Shadow experiment."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping, Sequence


class LongV5ShadowError(ValueError):
    pass


@dataclass(frozen=True)
class MultiObjectiveEstimate:
    success_probability: float
    expected_protected_net: float
    mae_q90: float
    time_to_profit_fraction: float
    success_uncertainty: float
    net_uncertainty: float
    mae_uncertainty: float
    time_uncertainty: float

    def __post_init__(self) -> None:
        values = tuple(self.__dict__.values())
        if not all(math.isfinite(value) for value in values):
            raise LongV5ShadowError("LONG v5 estimate contains non-finite values")
        if not 0.0 <= self.success_probability <= 1.0:
            raise LongV5ShadowError("LONG v5 success probability is invalid")
        if min(
            self.mae_q90,
            self.time_to_profit_fraction,
            self.success_uncertainty,
            self.net_uncertainty,
            self.mae_uncertainty,
            self.time_uncertainty,
        ) < 0.0:
            raise LongV5ShadowError("LONG v5 risk or uncertainty is negative")


def time_to_profit_fraction(row: Mapping[str, Any], *, horizon_bars: int) -> float:
    """Encode prompt target achievement without treating failures as fast wins."""

    if horizon_bars <= 0:
        raise LongV5ShadowError("LONG v5 horizon must be positive")
    first = row.get("first_favorable_bar")
    if not bool(row.get("target_before_stop")) or first is None:
        return 1.0
    value = int(first)
    if value <= 0:
        raise LongV5ShadowError("LONG v5 favorable bar is invalid")
    return min(1.0, value / horizon_bars)


def multiobjective_score(
    estimate: MultiObjectiveEstimate,
    *,
    atr_fraction: float,
    adverse_barrier_fraction: float,
    confidence_standard_deviations: float,
) -> Mapping[str, float]:
    """Combine conservative output bounds without manufacturing model outputs."""

    if (
        not math.isfinite(atr_fraction)
        or not math.isfinite(adverse_barrier_fraction)
        or not math.isfinite(confidence_standard_deviations)
        or atr_fraction <= 0.0
        or adverse_barrier_fraction <= 0.0
        or confidence_standard_deviations < 0.0
    ):
        raise LongV5ShadowError("LONG v5 score scale is invalid")
    z = confidence_standard_deviations
    success_lower = max(
        0.0, estimate.success_probability - z * estimate.success_uncertainty
    )
    net_lower = estimate.expected_protected_net - z * estimate.net_uncertainty
    mae_upper = max(0.0, estimate.mae_q90 + z * estimate.mae_uncertainty)
    time_upper = max(
        0.0, estimate.time_to_profit_fraction + z * estimate.time_uncertainty
    )
    net_quality = 1.0 / (1.0 + math.exp(-max(-30.0, min(30.0, net_lower / atr_fraction))))
    mae_quality = math.exp(-mae_upper / adverse_barrier_fraction)
    speed_quality = max(0.0, 1.0 - min(1.0, time_upper))
    normalized_uncertainty = max(
        min(1.0, estimate.success_uncertainty / 0.5),
        min(1.0, estimate.net_uncertainty / atr_fraction),
        min(1.0, estimate.mae_uncertainty / adverse_barrier_fraction),
        min(1.0, estimate.time_uncertainty),
    )
    certainty = 1.0 - normalized_uncertainty
    score = success_lower * net_quality * mae_quality * speed_quality * certainty
    return {
        "success_lower_bound": success_lower,
        "expected_net_lower_bound": net_lower,
        "mae_upper_bound": mae_upper,
        "time_to_profit_upper_bound": time_upper,
        "normalized_uncertainty": normalized_uncertainty,
        "certainty": certainty,
        "committee_score": score,
    }


def classify_regime_evidence(
    rows: Sequence[Mapping[str, Any]],
    *,
    unconditional_target_rate: float,
    unconditional_mae: float,
    minimum_rows: int,
) -> Mapping[str, Any]:
    """Classify a regime by realized commercial evidence, not by its name."""

    if minimum_rows <= 0 or not 0.0 <= unconditional_target_rate <= 1.0:
        raise LongV5ShadowError("LONG v5 regime contract is invalid")
    if not math.isfinite(unconditional_mae) or unconditional_mae < 0.0:
        raise LongV5ShadowError("LONG v5 unconditional MAE is invalid")
    if not rows:
        return {"status": "INSUFFICIENT", "rows": 0, "supported": False}
    net = sum(float(row["protected_worst_net_return"]) for row in rows) / len(rows)
    target_rate = sum(bool(row["target_before_stop"]) for row in rows) / len(rows)
    mae = sum(float(row["mae_fraction"]) for row in rows) / len(rows)
    enough = len(rows) >= minimum_rows
    supported = bool(
        enough
        and net > 0.0
        and target_rate > unconditional_target_rate
        and mae < unconditional_mae
    )
    return {
        "status": (
            "COMMERCIALLY_SUPPORTED"
            if supported
            else "UNSUPPORTED" if enough else "INSUFFICIENT"
        ),
        "rows": len(rows),
        "protected_net": net,
        "target_before_stop_rate": target_rate,
        "mae": mae,
        "supported": supported,
    }
