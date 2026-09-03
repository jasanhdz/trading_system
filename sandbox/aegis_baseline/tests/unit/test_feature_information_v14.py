from __future__ import annotations

import math

import pytest

from aegis.research.decomposed_entry_v9 import V9_FEATURE_NAMES
from aegis.research.feature_information_v14 import (
    binary_probability_metrics,
    feature_families,
    local_taker_flow,
    market_taker_flow,
    positional_feature_names,
    quality_profile,
    quantile_pinball_loss,
    robust_shift,
    taker_imbalance,
)


def test_feature_families_partition_all_v9_features_once() -> None:
    families = feature_families()
    flattened = [name for values in families.values() for name in values]
    assert len(flattened) == len(V9_FEATURE_NAMES) == 176
    assert set(flattened) == set(V9_FEATURE_NAMES)


def test_positional_names_make_existing_v9_collision_explicit() -> None:
    names = positional_feature_names(("a", "b", "a"))
    assert names == ("a__index_0", "b", "a__index_2")
    assert len(set(names)) == len(names)


def test_taker_flow_uses_only_closed_history() -> None:
    assert taker_imbalance(100.0, 60.0) == pytest.approx(0.2)
    result = local_taker_flow([0.1] * 12 + [-0.1] * 12)
    assert result["taker_imbalance_24"] == pytest.approx(0.0)
    assert result["taker_imbalance_3"] == pytest.approx(-0.1)


def test_market_flow_requires_complete_universe() -> None:
    local = {
        symbol: {"taker_imbalance_6": value}
        for symbol, value in [("BTCUSDT", 0.2)]
        + [(f"S{index}", -0.1) for index in range(10)]
    }
    result = market_taker_flow(local, symbol="S0")
    assert result["btc_taker_imbalance_6"] == 0.2
    assert result["market_taker_breadth_6"] == pytest.approx(1 / 11)


def test_quality_and_drift_are_finite_and_named() -> None:
    quality = quality_profile(
        [[1.0, 0.0], [2.0, 0.0]], ["a", "b"], near_constant_std=1e-12
    )
    assert quality["single_value"] == ["b"]
    shift = robust_shift([[0.0, 0.0], [1.0, 1.0]], [[1.0, 1.0], [2.0, 2.0]], ["a", "b"])
    assert shift["a"] == pytest.approx(2.0)


def test_probability_and_pinball_metrics_are_exact() -> None:
    metrics = binary_probability_metrics([0, 1], [0.1, 0.9])
    assert metrics["average_precision"] == 1.0
    assert metrics["log_loss"] == pytest.approx(-math.log(0.9))
    assert quantile_pinball_loss([0.0, 2.0], [1.0, 1.0], quantile=0.9) == pytest.approx(
        0.5
    )
