from __future__ import annotations

import numpy as np
import pytest

from aegis.research.temporal_stability_v13 import (
    consensus_probabilities,
    conservative_mae,
    distribution_scores,
    fit_robust_distribution,
    jensen_shannon_divergence,
    select_temporal_cross_section,
)


def test_consensus_requires_matching_states_and_small_divergence() -> None:
    historical = {"GOOD": 0.8, "BAD": 0.2}
    recent = {"GOOD": 0.75, "BAD": 0.25}
    result = consensus_probabilities(
        historical, recent, regime_expert=None, maximum_divergence=0.08
    )
    assert result["eligible"] is True
    assert result["probabilities"]["GOOD"] == pytest.approx(0.775)
    assert jensen_shannon_divergence(historical, recent) >= 0.0


def test_consensus_abstains_when_recent_model_disagrees() -> None:
    result = consensus_probabilities(
        {"GOOD": 0.8, "BAD": 0.2},
        {"GOOD": 0.2, "BAD": 0.8},
        regime_expert=None,
        maximum_divergence=1.0,
    )
    assert result["eligible"] is False


def test_distribution_gate_detects_large_feature_shift() -> None:
    reference = fit_robust_distribution(
        [[0.0, 0.0], [1.0, 1.0], [2.0, 2.0], [3.0, 3.0]],
        minimum_scale=1e-6,
    )
    scores = distribution_scores([[1.5, 1.5], [100.0, 100.0]], reference)
    assert scores[0] < scores[1]


def test_conservative_mae_uses_larger_temporal_estimate() -> None:
    result = conservative_mae([0.01, 0.03], [0.02, 0.01])
    assert np.allclose(result, [0.02, 0.03])


def test_temporal_selection_requires_consensus_distribution_and_mae() -> None:
    policy = {
        "minimum_utility": 0.001,
        "minimum_coherent_probability": 0.6,
        "maximum_adverse_probability": 0.3,
        "maximum_unknown_probability": 0.3,
        "maximum_predicted_mae": 0.01,
        "maximum_selected_per_timestamp": 1,
    }
    base = {
        "timestamp": "2026-01-01T00:00:00+00:00",
        "side": "LONG",
        "predicted_utility": 0.002,
        "coherent_probability": 0.7,
        "adverse_probability": 0.2,
        "unknown_probability": 0.1,
        "temporal_consensus": True,
        "in_distribution": True,
        "consensus_divergence": 0.01,
    }
    rows = [
        {**base, "symbol": "BTCUSDT", "predicted_mae_q90": 0.005},
        {**base, "symbol": "ETHUSDT", "predicted_mae_q90": 0.02},
    ]
    assert select_temporal_cross_section(rows, policy) == (True, False)
