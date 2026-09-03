from __future__ import annotations

from pathlib import Path

import yaml

from aegis.research.competing_barrier_v10 import contracts_from_config


def test_v11_frozen_horizon_contracts_match_v10_authority() -> None:
    root = Path(__file__).resolve().parents[2]
    v10 = yaml.safe_load(
        (root / "config/experiments/aegis_competing_barrier_v10_research.yaml").read_text()
    )
    v11 = yaml.safe_load(
        (root / "config/experiments/aegis_calibrated_horizon_v11_research.yaml").read_text()
    )
    contracts = contracts_from_config(v10)
    assert {contract.horizon_bars for contract in contracts} == set(
        v11["models"]["horizon_specialists"]["horizons_bars"]
    )
    assert v11["labels"]["reuse_v10_barrier_outcomes_unchanged"] is True
    assert v11["deployment"]["live_runtime"] == "PROHIBITED"
