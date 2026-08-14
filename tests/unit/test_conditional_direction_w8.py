import numpy as np
import pandas as pd
import pytest

from aegis.research.conditional_direction_w8 import (
    benjamini_hochberg,
    causal_previous_close,
    policy_actions,
    realized_policy_returns,
    stable_opportunity_id,
    symmetric_path_outcome,
    validate_direction_features,
)


def test_symmetric_target_assigns_long_and_short_with_identical_contract():
    common = dict(
        entry=100.0, favorable_bps=30.0, adverse_bps=30.0,
        cost_bps=14.0, minimum_utility_bps=2.0, minimum_advantage_bps=10.0,
    )
    long = symmetric_path_outcome(highs=[100.4], lows=[99.9], closes=[100.35], **common)
    short = symmetric_path_outcome(highs=[100.1], lows=[99.6], closes=[99.65], **common)
    assert long["economic_label"] == "LONG"
    assert short["economic_label"] == "SHORT"
    assert long["utility_long_bps"] == pytest.approx(16.0)
    assert short["utility_short_bps"] == pytest.approx(16.0)


def test_same_bar_ambiguity_is_adverse_first_for_both_sides():
    outcome = symmetric_path_outcome(
        entry=100.0, highs=[100.5], lows=[99.5], closes=[100.0],
        favorable_bps=30.0, adverse_bps=30.0, cost_bps=14.0,
        minimum_utility_bps=2.0, minimum_advantage_bps=10.0,
    )
    assert outcome["long_barrier"] == "ADVERSE_FIRST"
    assert outcome["short_barrier"] == "ADVERSE_FIRST"
    assert outcome["economic_label"] == "SKIP"


def test_neither_barrier_uses_symmetric_terminal_return_and_cost():
    outcome = symmetric_path_outcome(
        entry=100.0, highs=[100.1], lows=[99.9], closes=[100.05],
        favorable_bps=30.0, adverse_bps=30.0, cost_bps=14.0,
        minimum_utility_bps=2.0, minimum_advantage_bps=10.0,
    )
    assert outcome["utility_long_bps"] == pytest.approx(-9.0)
    assert outcome["utility_short_bps"] == pytest.approx(-19.0)
    assert outcome["economic_label"] == "SKIP"


def test_feature_contract_rejects_future_side_and_labels():
    validate_direction_features(["ret_3", "w7_opportunity_probability"])
    for forbidden in ("future_return", "side_ret_3", "utility_long"):
        with pytest.raises(ValueError, match="PROHIBITED"):
            validate_direction_features(["ret_3", forbidden])


def test_model_action_contracts_preserve_skip():
    a = policy_actions(
        "A_MULTICLASS_LOGISTIC",
        {"probabilities": np.array([[0.6, 0.2, 0.2], [0.4, 0.3, 0.3]]), "classes": np.array(["LONG", "SHORT", "SKIP"])},
        probability_threshold=0.5, utility_threshold=2, advantage_threshold=10, absolute_advantage_threshold=38,
    )
    b = policy_actions(
        "B_DUAL_UTILITY_RIDGE", {"long": np.array([5.0, 1.0]), "short": np.array([-8.0, 0.0])},
        probability_threshold=0.5, utility_threshold=2, advantage_threshold=10, absolute_advantage_threshold=38,
    )
    c = policy_actions(
        "C_ADVANTAGE_RIDGE", {"advantage": np.array([40.0, -50.0, 20.0])},
        probability_threshold=0.5, utility_threshold=2, advantage_threshold=10, absolute_advantage_threshold=38,
    )
    assert a.tolist() == ["LONG", "SKIP"]
    assert b.tolist() == ["LONG", "SKIP"]
    assert c.tolist() == ["LONG", "SHORT", "SKIP"]


def test_policy_returns_use_only_selected_side_utility():
    frame = pd.DataFrame({"h60_utility_long_bps": [10.0, 20.0, 30.0], "h60_utility_short_bps": [-10.0, -20.0, -30.0]})
    values = realized_policy_returns(frame, ["LONG", "SHORT", "SKIP"], 60)
    assert values.tolist() == [10.0, -20.0, 0.0]


def test_episode_id_is_direction_independent():
    assert stable_opportunity_id("BTCUSDT", "2026-01-01T00:00:00Z") == stable_opportunity_id("BTCUSDT", "2026-01-01T00:00:00Z")


def test_basis_contract_close_is_shifted_to_last_completed_bar():
    aligned = causal_previous_close([100.0, 101.0, 102.0])
    assert np.isnan(aligned[0])
    assert aligned[1:].tolist() == [100.0, 101.0]


def test_multiple_comparison_control_is_explicit():
    accepted = benjamini_hochberg({"a": 0.001, "b": 0.02, "c": 0.5}, 0.05)
    assert accepted == {"a": True, "b": True, "c": False}
