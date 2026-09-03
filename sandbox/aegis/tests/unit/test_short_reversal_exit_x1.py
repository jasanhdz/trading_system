from __future__ import annotations

from copy import deepcopy

import pytest
import yaml

from aegis.research.short_reversal_exit_x1 import (
    ShortReversalExitX1Error,
    assessment,
    cost_stress,
    daily_block_bootstrap_interval,
    profile_record,
)


def config():
    with open(
        "config/experiments/aegis_short_reversal_exit_compatibility_x1.yaml",
        encoding="utf-8",
    ) as source:
        return yaml.safe_load(source)


def prepared(side="SHORT"):
    profile = {
        "worst_net_return": 0.004,
        "worst_exit_reason": "TRAILING_STOP",
        "worst_bars_held": 5,
        "break_even_armed": True,
        "trailing_armed": True,
        "path_spread": 0.0,
    }
    return {
        "timestamp": "2026-07-18T00:00:00+00:00",
        "timestamp_ms": 1,
        "symbol": "BTCUSDT",
        "source": {
            "side": side,
            "protection_profiles": {
                "CURRENT_TS": profile,
                "LOCK_AT_5_ROE": {**profile, "worst_net_return": 0.001},
            },
            "v10_contract_outcomes": {"ROE_10_H12": {"realized_utility": 0.003}},
            "mae_fraction": 0.001,
            "mfe_fraction": 0.005,
            "time_underwater_bars": 1,
        },
    }


def test_profile_record_preserves_frozen_outcome_and_rejects_long():
    result = profile_record(prepared(), "CURRENT_TS")
    assert result["protected_net_return"] == 0.004
    assert result["profile"] == "CURRENT_TS"
    assert result["exchange_mutations"] == 0
    with pytest.raises(ShortReversalExitX1Error, match="SHORT rows only"):
        profile_record(prepared("LONG"), "CURRENT_TS")


def test_cost_stress_is_additional_and_does_not_modify_source():
    row = profile_record(prepared(), "CURRENT_TS")
    stressed = cost_stress([row], 0.0005)
    assert stressed[0]["protected_net_return"] == pytest.approx(0.0035)
    assert row["protected_net_return"] == 0.004


def test_daily_block_bootstrap_is_deterministic():
    base = profile_record(prepared(), "CURRENT_TS")
    rows = []
    for day in range(1, 8):
        row = deepcopy(base)
        row["timestamp"] = f"2026-07-{day + 17:02d}T00:00:00+00:00"
        rows.append(row)
    first = daily_block_bootstrap_interval(rows, seed=7, resamples=100)
    second = daily_block_bootstrap_interval(rows, seed=7, resamples=100)
    assert first == second
    assert first["lower_95"] == pytest.approx(0.004)


def test_assessment_requires_every_economic_and_uncertainty_gate():
    candidate = []
    v21 = []
    random_control = []
    for index in range(21):
        current = profile_record(prepared(), "CURRENT_TS")
        current["timestamp"] = f"2026-07-{(index % 7) + 18:02d}T{index % 24:02d}:00:00+00:00"
        candidate.append(current)
        v21.append({**current, "protected_net_return": 0.001})
        random_control.append({**current, "protected_net_return": -0.001})
    result = assessment(
        candidate=candidate,
        v21_exit=v21,
        random_control=random_control,
        diagnostic_profiles={},
        config=config(),
    )
    assert result["passed"] is True
    candidate[0]["mae_fraction"] = 1.0
    assert assessment(
        candidate=candidate,
        v21_exit=v21,
        random_control=random_control,
        diagnostic_profiles={},
        config=config(),
    )["passed"] is False

