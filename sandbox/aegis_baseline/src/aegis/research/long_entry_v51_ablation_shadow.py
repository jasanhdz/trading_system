"""Combinatorial ablation contracts for the four LONG v5 output heads."""

from __future__ import annotations

import itertools
import math
from enum import Enum
from typing import Any, Mapping, Sequence


class LongV51AblationError(ValueError):
    pass


class LongV5Head(str, Enum):
    SUCCESS = "SUCCESS"
    NET = "NET"
    MAE = "MAE"
    SPEED = "SPEED"


def all_head_combinations() -> tuple[tuple[LongV5Head, ...], ...]:
    heads = tuple(LongV5Head)
    return tuple(
        combination
        for size in range(1, len(heads) + 1)
        for combination in itertools.combinations(heads, size)
    )


def combination_identity(heads: Sequence[LongV5Head]) -> str:
    if not heads or len(set(heads)) != len(heads):
        raise LongV51AblationError("LONG v5.1 head combination is invalid")
    return "+".join(head.value for head in heads)


def ablation_factors(
    row: Mapping[str, Any],
    heads: Sequence[LongV5Head],
    *,
    conservative: bool = True,
) -> Mapping[str, Any]:
    if not heads:
        raise LongV51AblationError("LONG v5.1 requires at least one head")
    atr = float(row["atr_fraction"])
    adverse = float(row["adverse_barrier_fraction"])
    if not all(math.isfinite(value) and value > 0.0 for value in (atr, adverse)):
        raise LongV51AblationError("LONG v5.1 score scale is invalid")
    success_value = (
        float(row["success_lower_bound"])
        if conservative
        else float(row["success_probability"])
    )
    net_value = (
        float(row["expected_net_lower_bound"])
        if conservative
        else float(row["expected_protected_net"])
    )
    mae_value = (
        float(row["mae_upper_bound"])
        if conservative
        else float(row["mae_q90"])
    )
    time_value = (
        float(row["time_to_profit_upper_bound"])
        if conservative
        else float(row["time_to_profit_fraction"])
    )
    values: dict[LongV5Head, float] = {
        LongV5Head.SUCCESS: max(0.0, success_value),
        LongV5Head.NET: 1.0
        / (
            1.0
            + math.exp(
                -max(-30.0, min(30.0, net_value / atr))
            )
        ),
        LongV5Head.MAE: math.exp(-mae_value / adverse),
        LongV5Head.SPEED: max(
            0.0, 1.0 - min(1.0, time_value)
        ),
    }
    uncertainties: dict[LongV5Head, float] = {
        LongV5Head.SUCCESS: min(1.0, float(row["success_uncertainty"]) / 0.5),
        LongV5Head.NET: min(1.0, float(row["net_uncertainty"]) / atr),
        LongV5Head.MAE: min(1.0, float(row["mae_uncertainty"]) / adverse),
        LongV5Head.SPEED: min(1.0, float(row["time_uncertainty"])),
    }
    factor_values = [values[head] for head in heads]
    uncertainty = max(uncertainties[head] for head in heads) if conservative else 0.0
    certainty = 1.0 - uncertainty
    score = math.prod(factor_values) ** (1.0 / len(factor_values)) * certainty
    return {
        "combination": combination_identity(heads),
        "estimate_mode": "CONSERVATIVE_BOUNDS" if conservative else "POINT_ESTIMATES",
        "factors": {head.value: values[head] for head in heads},
        "normalized_uncertainty": uncertainty,
        "certainty": certainty,
        "committee_score": score,
    }
