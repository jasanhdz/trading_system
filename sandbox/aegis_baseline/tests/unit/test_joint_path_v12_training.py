from __future__ import annotations

from aegis.research.competing_barrier_v10 import BarrierContract
from aegis.research.joint_path_v12_training import (
    assign_contracts_by_regime,
    assigned_contract_name,
)


def test_contract_assignment_uses_policy_outcomes_and_global_fallback() -> None:
    contracts = (
        BarrierContract("FAST", 0.01, 0.01, 6, 0.002),
        BarrierContract("SLOW", 0.02, 0.02, 12, 0.002),
    )
    rows = [
        {
            "regime": "TREND",
            "outcomes": {
                "FAST": {"realized_utility": 0.01},
                "SLOW": {"realized_utility": -0.01},
            },
        }
        for _ in range(3)
    ]
    assignment = assign_contracts_by_regime(
        rows,
        contracts,
        minimum_group_rows=2,
        minimum_contract_observations=2,
    )
    assert assignment["regimes"]["TREND"]["contract"] == "FAST"
    assert assigned_contract_name({"regime": "UNKNOWN"}, assignment) == "FAST"
