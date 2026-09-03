from __future__ import annotations

import pytest

from aegis.research.regime_entry_exit_v7 import V7_FEATURE_NAMES
from aegis.research.tail_aware_entry_v8 import (
    SoftArchetype,
    TailLabelContract,
    classify_forward_regime,
    tail_labels,
    v8_feature_vector,
)


def row(**overrides: object) -> dict[str, object]:
    features = {name: 0.0 for name in V7_FEATURE_NAMES}
    features.update(
        {
            "atr_12": 0.004,
            "side_ret_1": 0.001,
            "side_ret_3": 0.003,
            "side_ret_12": 0.006,
            "volume_ratio_6_24": 1.2,
            "side_close_position_in_range": 0.7,
            "favorable_wick_fraction": 0.3,
            "adverse_wick_fraction": 0.1,
            "trend_agreement_score": 0.8,
            "side_acceleration": 0.001,
            "range_expansion": 0.2,
        }
    )
    value: dict[str, object] = {
        "v7_features": tuple(features[name] for name in V7_FEATURE_NAMES),
        "regime": {"structure": "TREND", "phase": "CONTINUATION"},
        "directional_role": "PRIMARY_TREND",
        "mae_fraction": 0.003,
        "first_positive_after_cost_bar": 2,
        "first_adverse_bar": 8,
        "first_favorable_bar": 3,
        "early_reversal": False,
        "same_bar_ambiguity": False,
        "target_before_stop": True,
    }
    value.update(overrides)
    return value


def test_soft_archetypes_cover_every_candidate_without_hard_transition() -> None:
    vector, memberships = v8_feature_vector(row())
    assert len(vector) == len(V7_FEATURE_NAMES) + len(SoftArchetype)
    assert set(memberships) == {value.value for value in SoftArchetype}
    assert sum(memberships.values()) == pytest.approx(1.0)
    assert all(value > 0.0 for value in memberships.values())


def test_forward_regime_requires_multihorizon_consensus() -> None:
    returns = {
        horizon: {"BTCUSDT": 0.004, "ETHUSDT": 0.003, "SOLUSDT": 0.002}
        for horizon in (6, 12, 24)
    }
    result = classify_forward_regime(
        returns,
        btc_threshold_at_24_fraction=0.0025,
        breadth_threshold=0.55,
        range_breadth_band=(0.45, 0.55),
        consensus_horizons=2,
    )
    assert result["label"] == "BULLISH"


def test_tail_labels_separate_entry_quality_from_stress_loss() -> None:
    contract = TailLabelContract(0.006, 6, 12, -0.01)
    labels = tail_labels(row(), {"CURRENT": -0.012, "TIGHTER": 0.002}, contract)
    assert labels["clean_entry"] is True
    assert labels["late_entry"] is False
    assert labels["positive_stress_net"] is True
    assert labels["catastrophic_stress_loss"] is False
    assert labels["hindsight_best_profile"] == "TIGHTER"


def test_slow_or_early_adverse_path_is_late() -> None:
    contract = TailLabelContract(0.006, 6, 12, -0.01)
    labels = tail_labels(
        row(first_positive_after_cost_bar=None, first_adverse_bar=1),
        {"CURRENT": -0.02},
        contract,
    )
    assert labels["late_entry"] is True
    assert labels["catastrophic_stress_loss"] is True
