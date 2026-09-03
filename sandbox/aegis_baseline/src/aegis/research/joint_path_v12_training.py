"""Contract assignment and validation helpers for V12 research."""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Mapping, Sequence

import numpy as np

from .competing_barrier_v10 import BarrierContract, BarrierResearchError


def assign_contracts_by_regime(
    rows: Sequence[Mapping[str, Any]],
    contracts: Sequence[BarrierContract],
    *,
    minimum_group_rows: int,
    minimum_contract_observations: int,
) -> Mapping[str, Any]:
    if (
        not rows
        or not contracts
        or min(minimum_group_rows, minimum_contract_observations) <= 0
    ):
        raise BarrierResearchError("invalid V12 contract assignment inputs")
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["regime"])].append(row)

    def choose(source: Sequence[Mapping[str, Any]]) -> Mapping[str, Any]:
        choices = []
        for contract in contracts:
            values = [
                float(row["outcomes"][contract.name]["realized_utility"])
                for row in source
                if contract.name in row["outcomes"]
            ]
            if len(values) >= minimum_contract_observations:
                choices.append((float(np.mean(values)), contract.name, len(values)))
        if not choices:
            raise BarrierResearchError("no supported V12 contract assignment")
        utility, name, count = max(choices, key=lambda item: (item[0], item[1]))
        return {"contract": name, "mean_realized_utility": utility, "rows": count}

    global_assignment = choose(rows)
    regimes = {
        regime: choose(group_rows)
        for regime, group_rows in sorted(grouped.items())
        if len(group_rows) >= minimum_group_rows
    }
    return {
        "source": "ASSIGNMENT_WINDOW_ONLY",
        "global": global_assignment,
        "regimes": regimes,
    }


def assigned_contract_name(
    row: Mapping[str, Any], assignment: Mapping[str, Any]
) -> str:
    regime = str(row["regime"])
    selected = assignment["regimes"].get(regime, assignment["global"])
    return str(selected["contract"])
