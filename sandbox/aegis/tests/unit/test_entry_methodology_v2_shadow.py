from __future__ import annotations

from aegis.research.entry_methodology_v2_shadow import (
    ENTRY_PATH_MODEL_FEATURE_NAMES,
    EntryMethodologyV2Policy,
    assess_entry_methodology_v2_shadow,
    entry_path_model_features,
    label_clean_entry_path,
)


def assessment(**overrides):
    values = {
        "market_timestamp": "2026-08-09T00:00:00Z",
        "side": "LONG",
        "prediction": {
            "opportunity_probability": 0.65,
            "danger_probability": 0.25,
            "mae_q90": 0.002,
            "mfe_q50": 0.008,
            "net_return_mean": 0.002,
        },
        "confirmation": {"state": "CONFIRMED", "components_passed": 4},
        "confirmation_features": {
            "signed_ret_1_atr": 0.2,
            "signed_ret_3_atr": 0.4,
            "signed_ret_6_atr": 0.6,
            "signed_ret_12_atr": 0.8,
        },
        "current_layer": {"rv2_tail_risk": 0.20},
        "entry_intelligence": {
            "regime_v3_shadow": {"volatility": "NORMAL", "structure": "TREND"},
            "entry_timing_shadow": {"state": "NO_PENDING_SETUP"},
        },
        "confirmed_same_side": 2,
        "confirmed_total": 3,
        "previous": None,
    }
    values.update(overrides)
    return assess_entry_methodology_v2_shadow(**values)


def bars(*values: tuple[float, float, float]):
    return [{"high": high, "low": low, "close": close} for high, low, close in values]


def test_grade_a_is_immediate_and_observational_only() -> None:
    result = assessment()
    assert result["tier"] == "A"
    assert result["counterfactual_action"] == "COUNTERFACTUAL_ENTER_NOW"
    assert result["selection_effect"] == "NONE"
    assert result["exchange_authority"] is False
    assert result["exchange_mutations"] == 0
    assert result["typescript_sizing_unchanged"] is True


def test_grade_b_waits_for_the_next_closed_bar() -> None:
    result = assessment(confirmation={"state": "CONFIRMED", "components_passed": 3})
    assert result["tier"] == "B"
    assert result["counterfactual_action"] == "COUNTERFACTUAL_WAIT_NEXT_CLOSED_BAR"
    assert result["sequential_confirmation"]["state"] == "WAITING"


def test_waiting_candidate_can_confirm_without_fabricating_evidence() -> None:
    waiting = assessment(confirmation={"state": "CONFIRMED", "components_passed": 3})
    confirmed = assessment(
        market_timestamp="2026-08-09T00:05:00Z",
        previous=waiting,
    )
    assert confirmed["tier"] == "A"
    assert (
        confirmed["counterfactual_action"] == "COUNTERFACTUAL_ENTER_AFTER_CONFIRMATION"
    )
    assert confirmed["sequential_confirmation"]["confirmed_after_wait"] is True


def test_wait_expires_fail_closed() -> None:
    waiting = assessment(confirmation={"state": "CONFIRMED", "components_passed": 3})
    waiting = {
        **waiting,
        "market_timestamp": "2026-08-09T00:10:00Z",
        "sequential_confirmation": {
            **waiting["sequential_confirmation"],
            "state": "WAITING",
            "age_bars": 2,
        },
    }
    result = assessment(
        market_timestamp="2026-08-09T00:15:00Z",
        confirmation={"state": "CONFIRMED", "components_passed": 3},
        previous=waiting,
    )
    assert result["tier"] == "C"
    assert result["counterfactual_action"] == "COUNTERFACTUAL_ABSTAIN_EXPIRED"


def test_weak_absolute_quality_is_grade_c() -> None:
    result = assessment(
        confirmation={"state": "ABSTAIN_WEAK_QUALITY", "components_passed": 5}
    )
    assert result["tier"] == "C"
    assert result["counterfactual_action"] == "COUNTERFACTUAL_ABSTAIN"


def test_long_clean_path_records_fast_favorable_first() -> None:
    policy = EntryMethodologyV2Policy(horizon_bars=3, fast_edge_bars=2)
    result = label_clean_entry_path(
        side="LONG",
        entry_price=100.0,
        future_bars=bars(
            (100.2, 99.9, 100.1),
            (100.5, 100.0, 100.4),
            (100.6, 100.2, 100.5),
        ),
        policy=policy,
    )
    assert result["classification"] == "CLEAN_FAST_SUCCESS"
    assert result["clean_path_success"] is True
    assert result["favorable_barrier_bar"] == 2
    assert result["adverse_barrier_bar"] is None


def test_short_clean_path_orients_prices_correctly() -> None:
    policy = EntryMethodologyV2Policy(horizon_bars=3, fast_edge_bars=2)
    result = label_clean_entry_path(
        side="SHORT",
        entry_price=100.0,
        future_bars=bars(
            (100.1, 99.9, 99.95),
            (100.0, 99.5, 99.6),
            (99.8, 99.4, 99.5),
        ),
        policy=policy,
    )
    assert result["classification"] == "CLEAN_FAST_SUCCESS"
    assert result["net_return_after_costs"] > 0.0


def test_adverse_first_and_same_bar_ambiguity_are_not_successes() -> None:
    policy = EntryMethodologyV2Policy(
        horizon_bars=2, fast_edge_bars=1, maximum_wait_bars=2
    )
    adverse = label_clean_entry_path(
        side="LONG",
        entry_price=100.0,
        future_bars=bars((100.1, 99.5, 99.7), (100.5, 99.8, 100.4)),
        policy=policy,
    )
    ambiguous = label_clean_entry_path(
        side="LONG",
        entry_price=100.0,
        future_bars=bars((100.5, 99.5, 100.0), (100.6, 100.0, 100.4)),
        policy=policy,
    )
    assert adverse["classification"] == "ADVERSE_FIRST"
    assert ambiguous["classification"] == "AMBIGUOUS_SAME_BAR"
    assert adverse["clean_path_success"] is False
    assert ambiguous["clean_path_success"] is False


def test_meta_model_features_are_ordered_finite_and_directional() -> None:
    prediction = {
        "opportunity_probability": 0.6,
        "danger_probability": 0.2,
        "mae_q50": 0.001,
        "mae_q90": 0.003,
        "mfe_q50": 0.006,
        "net_return_mean": 0.002,
        "shadow_rank_score": 0.7,
    }
    confirmation = {
        "components_passed": 4,
        "relative_quality": {
            "opportunity_percentile": 0.8,
            "danger_quality_percentile": 0.7,
            "net_return_percentile": 0.6,
            "path_efficiency_percentile": 0.9,
            "path_efficiency": 2.0,
        },
    }
    confirmation_features = {
        name: float(index + 1) / 10.0
        for index, name in enumerate(ENTRY_PATH_MODEL_FEATURE_NAMES[13:32])
    }
    intelligence = {
        "regime_v3_shadow": {
            "global_direction": "BULLISH",
            "symbol_direction": "BULLISH",
            "volatility": "NORMAL",
            "structure": "TREND",
            "alignment": "ALIGNED_BULLISH",
            "extension": "NORMAL",
            "evidence_ready": True,
            "global_stability_bars": 4,
            "symbol_stability_bars": 5,
            "volatility_stability_bars": 6,
            "structure_stability_bars": 7,
        },
        "entry_timing_shadow": {
            "reversal_flags": {"rebound": False, "fake_break": True}
        },
        "directional_acceleration_shadow": {
            "component_count": 10,
            "upward_component_count": 8,
            "downward_component_count": 2,
        },
    }
    long_values = entry_path_model_features(
        side="LONG",
        prediction=prediction,
        confirmation=confirmation,
        confirmation_features=confirmation_features,
        entry_intelligence=intelligence,
    )
    short_values = entry_path_model_features(
        side="SHORT",
        prediction=prediction,
        confirmation=confirmation,
        confirmation_features=confirmation_features,
        entry_intelligence=intelligence,
    )
    assert len(long_values) == len(ENTRY_PATH_MODEL_FEATURE_NAMES)
    assert len(short_values) == len(ENTRY_PATH_MODEL_FEATURE_NAMES)
    global_alignment = ENTRY_PATH_MODEL_FEATURE_NAMES.index(
        "global_direction_alignment"
    )
    directional_acceleration = ENTRY_PATH_MODEL_FEATURE_NAMES.index(
        "directional_acceleration_fraction"
    )
    assert long_values[global_alignment] == 1.0
    assert short_values[global_alignment] == -1.0
    assert long_values[directional_acceleration] == 0.8
    assert short_values[directional_acceleration] == 0.2
