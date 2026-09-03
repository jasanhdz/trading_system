from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from aegis.research.long_entry_v5_multiobjective_shadow import (
    MultiObjectiveEstimate,
    classify_regime_evidence,
    multiobjective_score,
    time_to_profit_fraction,
)


def test_time_to_profit_uses_target_bar_and_failure_is_not_fast() -> None:
    assert (
        time_to_profit_fraction(
            {"target_before_stop": True, "first_favorable_bar": 3}, horizon_bars=12
        )
        == 0.25
    )
    assert (
        time_to_profit_fraction(
            {"target_before_stop": False, "first_favorable_bar": 3}, horizon_bars=12
        )
        == 1.0
    )
    assert (
        time_to_profit_fraction(
            {"target_before_stop": False, "first_favorable_bar": None},
            horizon_bars=12,
        )
        == 1.0
    )


def test_multiobjective_score_uses_conservative_bounds() -> None:
    strong = MultiObjectiveEstimate(
        success_probability=0.8,
        expected_protected_net=0.004,
        mae_q90=0.002,
        time_to_profit_fraction=0.2,
        success_uncertainty=0.01,
        net_uncertainty=0.0001,
        mae_uncertainty=0.0001,
        time_uncertainty=0.01,
    )
    weak = MultiObjectiveEstimate(
        **{
            **strong.__dict__,
            "success_uncertainty": 0.25,
            "net_uncertainty": 0.003,
        }
    )
    strong_score = multiobjective_score(
        strong,
        atr_fraction=0.01,
        adverse_barrier_fraction=0.005,
        confidence_standard_deviations=1.645,
    )
    weak_score = multiobjective_score(
        weak,
        atr_fraction=0.01,
        adverse_barrier_fraction=0.005,
        confidence_standard_deviations=1.645,
    )
    assert strong_score["expected_net_lower_bound"] > 0.0
    assert weak_score["committee_score"] < strong_score["committee_score"]


def test_invalid_or_fabricated_probabilities_fail_closed() -> None:
    with pytest.raises(ValueError):
        MultiObjectiveEstimate(
            success_probability=1.1,
            expected_protected_net=0.0,
            mae_q90=0.0,
            time_to_profit_fraction=0.0,
            success_uncertainty=0.0,
            net_uncertainty=0.0,
            mae_uncertainty=0.0,
            time_uncertainty=0.0,
        )


def test_regime_name_does_not_imply_commercial_support() -> None:
    weak = [
        {
            "protected_worst_net_return": -0.001,
            "target_before_stop": False,
            "mae_fraction": 0.02,
        }
        for _ in range(40)
    ]
    evidence = classify_regime_evidence(
        weak,
        unconditional_target_rate=0.4,
        unconditional_mae=0.01,
        minimum_rows=30,
    )
    assert evidence["status"] == "UNSUPPORTED"
    assert evidence["supported"] is False


def test_regime_requires_sample_net_direction_and_path_quality() -> None:
    strong = [
        {
            "protected_worst_net_return": 0.002,
            "target_before_stop": True if index < 25 else False,
            "mae_fraction": 0.004,
        }
        for index in range(40)
    ]
    evidence = classify_regime_evidence(
        strong,
        unconditional_target_rate=0.5,
        unconditional_mae=0.01,
        minimum_rows=30,
    )
    assert evidence["status"] == "COMMERCIALLY_SUPPORTED"
    assert evidence["supported"] is True


def test_v5_is_shadow_only_and_freezes_v4() -> None:
    root = Path(__file__).parents[2]
    payload = yaml.safe_load(
        (root / "config/experiments/aegis_long_entry_v5_shadow.yaml").read_text()
    )
    assert payload["mode"] == "SHADOW"
    assert payload["selection_effect"] == "NONE"
    assert payload["automatic_live_promotion"] is False
    assert payload["uncertainty_and_abstention"]["no_forced_trade_quota"] is True
    assert payload["regime_truth_audit"]["regime_is_not_a_hard_gate_in_v5"] is True
    assert payload["deployment"]["live_runtime"] == "PROHIBITED"
    assert payload["deployment"]["exchange_mutations"] == 0
