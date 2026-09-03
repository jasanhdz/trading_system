from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from aegis.research.long_entry_v51_ablation_shadow import (
    LongV5Head,
    ablation_factors,
    all_head_combinations,
    combination_identity,
)


def _row() -> dict[str, float]:
    return {
        "atr_fraction": 0.01,
        "adverse_barrier_fraction": 0.005,
        "success_lower_bound": 0.70,
        "success_probability": 0.75,
        "expected_net_lower_bound": 0.002,
        "expected_protected_net": 0.003,
        "mae_upper_bound": 0.003,
        "mae_q90": 0.0025,
        "time_to_profit_upper_bound": 0.25,
        "time_to_profit_fraction": 0.20,
        "success_uncertainty": 0.02,
        "net_uncertainty": 0.0002,
        "mae_uncertainty": 0.0001,
        "time_uncertainty": 0.03,
    }


def test_generates_exactly_all_fifteen_non_empty_combinations() -> None:
    combinations = all_head_combinations()
    assert len(combinations) == 15
    assert len({combination_identity(value) for value in combinations}) == 15
    assert (LongV5Head.MAE,) in combinations
    assert tuple(LongV5Head) in combinations


@pytest.mark.parametrize("head", tuple(LongV5Head))
def test_each_single_head_uses_only_its_factor(head: LongV5Head) -> None:
    result = ablation_factors(_row(), (head,))
    assert result["combination"] == head.value
    assert set(result["factors"]) == {head.value}
    assert 0.0 <= result["committee_score"] <= 1.0


def test_combination_score_penalizes_uncertain_included_head() -> None:
    stable = ablation_factors(_row(), (LongV5Head.SUCCESS, LongV5Head.MAE))
    uncertain_row = {**_row(), "mae_uncertainty": 0.006}
    uncertain = ablation_factors(
        uncertain_row, (LongV5Head.SUCCESS, LongV5Head.MAE)
    )
    assert uncertain["committee_score"] < stable["committee_score"]


def test_point_estimate_sensitivity_measures_ranking_without_uncertainty_block() -> None:
    result = ablation_factors(
        _row(), (LongV5Head.MAE,), conservative=False
    )
    assert result["estimate_mode"] == "POINT_ESTIMATES"
    assert result["normalized_uncertainty"] == 0.0
    assert result["committee_score"] > 0.0


def test_duplicate_or_empty_combination_fails_closed() -> None:
    with pytest.raises(ValueError):
        combination_identity(())
    with pytest.raises(ValueError):
        combination_identity((LongV5Head.MAE, LongV5Head.MAE))


def test_v51_is_exploratory_shadow_and_cannot_promote() -> None:
    root = Path(__file__).parents[2]
    payload = yaml.safe_load(
        (
            root
            / "config/experiments/aegis_long_entry_v51_ablation_shadow.yaml"
        ).read_text()
    )
    assert payload["mode"] == "SHADOW"
    assert payload["analysis_class"] == "EXPLORATORY_NOT_PROMOTION_ELIGIBLE"
    assert payload["ablation"]["expected_combination_count"] == 15
    assert payload["deployment"]["live_runtime"] == "PROHIBITED"
    assert payload["deployment"]["model_export"] == "PROHIBITED"
    assert payload["deployment"]["exchange_mutations"] == 0
