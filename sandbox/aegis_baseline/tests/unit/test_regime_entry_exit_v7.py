from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from aegis.data import CanonicalBar
from aegis.research.hybrid_ts_protection_replay import TsProtectionConfig
from aegis.research.regime_aware_directional_v6 import (
    REGIME_AWARE_V6_FEATURE_NAMES,
)
from aegis.research.regime_entry_exit_v7 import (
    EntryArchetype,
    TrajectoryAuditContract,
    causal_entry_context,
    replay_protection_profiles,
    trajectory_attribution,
    v7_feature_vector,
)
from aegis.training.hybrid_directional import DirectionalSide


def source_features(**overrides: float) -> tuple[float, ...]:
    values = {name: 0.0 for name in REGIME_AWARE_V6_FEATURE_NAMES}
    values.update(
        {
            "atr_12": 0.004,
            "market_breadth_6": 0.6,
            "side_market_breadth_6": 0.6,
            "side_ret_1": 0.001,
            "side_ret_3": 0.003,
            "side_ret_12": 0.006,
            "volume_ratio_6_24": 1.2,
            "side_close_position_in_range": 0.7,
            "side_trend_stack": 1.0,
            "15m_trend_stack_long": 1.0,
            "1h_trend_stack_long": 1.0,
            "15m_ret_3": 0.002,
            "1h_ret_3": 0.004,
        }
    )
    values.update(overrides)
    return tuple(values[name] for name in REGIME_AWARE_V6_FEATURE_NAMES)


def candidate(**overrides: object) -> dict[str, object]:
    row: dict[str, object] = {
        "side": "LONG",
        "features": source_features(),
        "regime": {"structure": "TREND", "phase": "CONTINUATION"},
        "directional_role": "PRIMARY_TREND",
        "mfe_fraction": 0.012,
        "mae_fraction": 0.002,
        "full_lifecycle_worst_net_return": 0.003,
        "first_positive_after_cost_bar": 2,
        "protectable_advantage": True,
        "early_reversal": False,
        "same_bar_ambiguity": False,
    }
    row.update(overrides)
    return row


def contract() -> TrajectoryAuditContract:
    return TrajectoryAuditContract(0.001, 0.006, 6, 2.0, 12, 0.001)


def test_v7_context_and_archetype_are_causal() -> None:
    row = candidate()
    vector, archetype, context = v7_feature_vector(row)
    assert len(vector) == 162
    assert archetype is EntryArchetype.TREND_CONTINUATION
    assert context["side_extension_atr"] == pytest.approx(1.5)
    assert context["trend_agreement_score"] == 1.0


def test_trajectory_separates_clean_entry_from_poor_capture() -> None:
    _, _, context = v7_feature_vector(candidate())
    clean = trajectory_attribution(candidate(), context, contract())
    poor = trajectory_attribution(
        candidate(full_lifecycle_worst_net_return=-0.002), context, contract()
    )
    assert clean["responsibility"] == "CLEAN_REALIZED_WIN"
    assert clean["capture_efficiency"] > 0.0
    assert poor["responsibility"] == "GOOD_ENTRY_POOR_CAPTURE"
    assert poor["capture_efficiency"] == 0.0


def test_late_extended_entry_is_not_mislabeled_as_exit_failure() -> None:
    row = candidate(
        features=source_features(side_ret_12=0.012),
        protectable_advantage=False,
        full_lifecycle_worst_net_return=-0.002,
        first_positive_after_cost_bar=None,
    )
    _, _, context = v7_feature_vector(row)
    audit = trajectory_attribution(row, context, contract())
    assert audit["late_entry"] is True
    assert audit["responsibility"] == "LATE_OR_ADVERSE_ENTRY"


def test_protection_profiles_replay_both_paths_fail_closed() -> None:
    start = datetime(2026, 1, 1, tzinfo=timezone.utc)
    history = tuple(
        CanonicalBar(
            start + timedelta(minutes=5 * index),
            100.0,
            100.2,
            99.8,
            100.0,
            1.0,
        )
        for index in range(20)
    )
    future = (
        CanonicalBar(start + timedelta(minutes=100), 100.0, 101.0, 99.9, 100.8, 1.0),
        CanonicalBar(start + timedelta(minutes=105), 100.8, 100.9, 100.0, 100.1, 1.0),
    )
    result = replay_protection_profiles(
        side=DirectionalSide.LONG,
        history=history,
        future=future,
        profiles={"TEST": TsProtectionConfig(use_atr_trailing=False)},
    )
    assert set(result) == {"TEST"}
    assert result["TEST"]["path_spread"] >= 0.0


def test_non_finite_context_fails_closed() -> None:
    values = dict(zip(REGIME_AWARE_V6_FEATURE_NAMES, source_features()))
    values["side_ret_3"] = float("nan")
    with pytest.raises(ValueError):
        causal_entry_context(values, DirectionalSide.LONG)
