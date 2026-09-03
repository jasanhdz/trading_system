"""Joint direction/path labels and conservative utility for V12 research."""

from __future__ import annotations

import math
from enum import Enum
from typing import Any, Mapping, Sequence

from .competing_barrier_v10 import BarrierContract, BarrierOutcome, BarrierResearchError


class JointPathState(str, Enum):
    COHERENT_CLEAN_FAVORABLE = "COHERENT_CLEAN_FAVORABLE"
    COHERENT_DIRTY_FAVORABLE = "COHERENT_DIRTY_FAVORABLE"
    ADVERSE_FIRST = "ADVERSE_FIRST"
    SAME_BAR_AMBIGUOUS = "SAME_BAR_AMBIGUOUS"
    UNRESOLVED_OR_DIRECTION_MISMATCH = "UNRESOLVED_OR_DIRECTION_MISMATCH"


def joint_path_state(
    *, side: str, direction_label: str, clean_entry: bool, outcome: str
) -> str:
    if side not in {"LONG", "SHORT"}:
        raise BarrierResearchError("invalid V12 side")
    if outcome == BarrierOutcome.ADVERSE_FIRST.value:
        return JointPathState.ADVERSE_FIRST.value
    if outcome == BarrierOutcome.SAME_BAR_AMBIGUOUS.value:
        return JointPathState.SAME_BAR_AMBIGUOUS.value
    if outcome == BarrierOutcome.FAVORABLE_FIRST.value and direction_label == side:
        return (
            JointPathState.COHERENT_CLEAN_FAVORABLE.value
            if clean_entry
            else JointPathState.COHERENT_DIRTY_FAVORABLE.value
        )
    if outcome not in {value.value for value in BarrierOutcome}:
        raise BarrierResearchError("invalid V12 barrier outcome")
    return JointPathState.UNRESOLVED_OR_DIRECTION_MISMATCH.value


def joint_state_utility(
    probabilities: Mapping[str, float],
    contract: BarrierContract,
    config: Mapping[str, Any],
) -> Mapping[str, float]:
    expected = {state.value for state in JointPathState}
    if set(probabilities) != expected:
        raise BarrierResearchError("incomplete V12 joint probability vector")
    values = {name: float(value) for name, value in probabilities.items()}
    if any(
        not math.isfinite(value) or value < 0.0 or value > 1.0
        for value in values.values()
    ) or not math.isclose(sum(values.values()), 1.0, abs_tol=1e-6):
        raise BarrierResearchError("invalid V12 joint probability vector")
    clean = (
        values[JointPathState.COHERENT_CLEAN_FAVORABLE.value]
        * contract.favorable_fraction
        * float(config["clean_favorable_discount"])
    )
    dirty = (
        values[JointPathState.COHERENT_DIRTY_FAVORABLE.value]
        * contract.favorable_fraction
        * float(config["dirty_favorable_discount"])
    )
    adverse = -values[JointPathState.ADVERSE_FIRST.value] * contract.adverse_fraction
    ambiguous = (
        -values[JointPathState.SAME_BAR_AMBIGUOUS.value]
        * contract.adverse_fraction
        * float(config["ambiguous_penalty_fraction_of_adverse"])
    )
    unresolved = (
        -values[JointPathState.UNRESOLVED_OR_DIRECTION_MISMATCH.value]
        * contract.adverse_fraction
        * float(config["unresolved_penalty_fraction_of_adverse"])
    )
    cost = -contract.severe_cost_fraction
    return {
        "clean_favorable_value": clean,
        "dirty_favorable_value": dirty,
        "adverse_value": adverse,
        "ambiguous_value": ambiguous,
        "unresolved_value": unresolved,
        "cost": cost,
        "total_utility": clean + dirty + adverse + ambiguous + unresolved + cost,
    }


def select_joint_cross_section(
    rows: Sequence[Mapping[str, Any]], policy: Mapping[str, Any]
) -> tuple[bool, ...]:
    required = (
        "minimum_utility",
        "minimum_coherent_probability",
        "maximum_adverse_probability",
        "maximum_unknown_probability",
    )
    try:
        values = {name: float(policy[name]) for name in required}
        maximum = int(policy["maximum_selected_per_timestamp"])
    except (KeyError, TypeError, ValueError) as exc:
        raise BarrierResearchError("incomplete V12 selection policy") from exc
    if (
        maximum <= 0
        or values["minimum_utility"] < 0.0
        or any(not math.isfinite(value) for value in values.values())
        or any(not 0.0 <= values[name] <= 1.0 for name in required[1:])
    ):
        raise BarrierResearchError("invalid V12 selection policy")
    by_timestamp: dict[str, list[tuple[int, Mapping[str, Any]]]] = {}
    for index, row in enumerate(rows):
        if (
            float(row["predicted_utility"]) >= values["minimum_utility"]
            and float(row["coherent_probability"])
            >= values["minimum_coherent_probability"]
            and float(row["adverse_probability"])
            <= values["maximum_adverse_probability"]
            and float(row["unknown_probability"])
            <= values["maximum_unknown_probability"]
        ):
            by_timestamp.setdefault(str(row["timestamp"]), []).append((index, row))
    selected = [False] * len(rows)
    for candidates in by_timestamp.values():
        candidates.sort(
            key=lambda item: (
                -float(item[1]["predicted_utility"]),
                -float(item[1]["coherent_probability"]),
                float(item[1]["adverse_probability"]),
                str(item[1]["symbol"]),
                str(item[1]["side"]),
            )
        )
        for index, _ in candidates[:maximum]:
            selected[index] = True
    return tuple(selected)


def path_quality_metrics(rows: Sequence[Mapping[str, Any]]) -> Mapping[str, Any]:
    if not rows:
        return {
            "count": 0,
            "clean_rate": None,
            "adverse_first_rate": None,
            "mean_mae_fraction": None,
            "median_first_positive_bar": None,
            "mean_maximum_favorable_excursion_fraction": None,
        }
    first_positive = [
        int(value)
        for row in rows
        if (
            value := row["v11_path_diagnostics"].get(
                "first_positive_after_severe_cost_bar"
            )
        )
        is not None
    ]
    return {
        "count": len(rows),
        "clean_rate": sum(bool(row["v11_clean_entry_label"]) for row in rows)
        / len(rows),
        "adverse_first_rate": sum(
            row["actual_outcome"] == BarrierOutcome.ADVERSE_FIRST.value for row in rows
        )
        / len(rows),
        "mean_mae_fraction": sum(float(row["mae_fraction"]) for row in rows)
        / len(rows),
        "median_first_positive_bar": (
            float(sorted(first_positive)[len(first_positive) // 2])
            if first_positive
            else None
        ),
        "mean_maximum_favorable_excursion_fraction": sum(
            float(row["v11_path_diagnostics"]["maximum_favorable_excursion_fraction"])
            for row in rows
        )
        / len(rows),
    }
