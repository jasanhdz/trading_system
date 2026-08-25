from dataclasses import FrozenInstanceError
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import pytest

from aegis_ephemeral_regime_w11.modeling import (
    cosine_similarity,
    diagonal_covariance_similarity,
    ExpirationGuardian,
    FrozenExpert,
    ResolvedOutcome,
    ValidationCandidate,
    fit_regime_similarity,
    make_instance_id,
    select_validation_candidate,
    standardized_euclidean_similarity,
    temporal_block_bootstrap,
    train_candidate,
)


UTC = timezone.utc


def _expert(*, window_hours=24, threshold=0.0):
    created = datetime(2023, 11, 1, 12, tzinfo=UTC)
    ttl = {6: 6, 12: 6, 24: 12, 48: 24, 72: 24}[window_hours]
    return FrozenExpert(
        instance_id=make_instance_id(created, window_hours, 15, 1),
        model_version="w11-frozen-v1",
        window_hours=window_hours,
        horizon_minutes=15,
        sequence=1,
        training_start=created - timedelta(hours=window_hours + 7),
        training_end=created - timedelta(hours=7),
        validation_start=created - timedelta(hours=7),
        validation_end=created - timedelta(hours=1),
        created_at=created,
        expires_at=created + timedelta(hours=ttl),
        similarity_method="standardized_euclidean",
        similarity_threshold=threshold,
    )


def _training_data():
    rng = np.random.default_rng(77)
    features = pd.DataFrame(rng.normal(size=(240, 4)), columns=list("abcd"))
    features.loc[::23, "b"] = np.nan
    opportunity = np.zeros(240, dtype=int)
    opportunity[::2] = 1
    direction = (features["a"].fillna(0).to_numpy() > 0).astype(int)
    return features, opportunity, direction


def test_class_count_failure_is_fail_closed():
    features, opportunity, direction = _training_data()
    opportunity[:] = 0
    opportunity[:9] = 1
    candidate = train_candidate(features, opportunity, direction)

    assert not candidate.active
    assert candidate.failure_reason == "OPPORTUNITY_CLASS_COUNT"
    assert {prediction.decision for prediction in candidate.predict(features.iloc[:3])} == {"SKIP"}


def test_predictions_are_deterministic_and_include_expected_edge():
    features, opportunity, direction = _training_data()
    first = train_candidate(features, opportunity, direction, expected_edge_bps=3.25)
    second = train_candidate(features, opportunity, direction, expected_edge_bps=3.25)

    assert first.active and second.active
    assert first.predict(features.iloc[:30]) == second.predict(features.iloc[:30])
    assert all(item.decision in {"LONG", "SHORT", "SKIP"} for item in first.predict(features))
    assert all(item.expected_edge_bps == 3.25 for item in first.predict(features.iloc[:4]))
    assert first.opportunity_model.named_steps["imputer"].strategy == "median"
    assert first.opportunity_model.named_steps["logistic"].C == 0.25


def test_three_similarities_and_standardized_euclidean_tie_preference():
    training = np.column_stack((np.arange(40.0), np.arange(40.0) ** 2))
    validation = training[-8:] + 0.1
    # A constant edge gives every method the same undefined rank relationship.
    selected = fit_regime_similarity(training, validation, np.ones(8))

    assert selected.method == "standardized_euclidean"
    assert np.isfinite(selected.threshold)
    assert selected.score(validation).shape == (8,)
    for diagnostic in (
        standardized_euclidean_similarity,
        cosine_similarity,
        diagonal_covariance_similarity,
    ):
        assert np.all(np.isfinite(diagnostic(training, validation)))
    alternatives = []
    for edge in (np.arange(8.0), -np.arange(8.0)):
        alternatives.append(fit_regime_similarity(training, validation, edge).method)
    assert set(alternatives).issubset(
        {"standardized_euclidean", "cosine", "diagonal_covariance"}
    )


def test_validation_selection_is_fixed_and_deterministic():
    def candidate(window, horizon, stress):
        return ValidationCandidate(
            window,
            horizon,
            stress,
            1.0,
            12,
            {"BTC": 4, "ETH": 4, "SOL": 4},
            0.8,
        )

    selected = select_validation_candidate(
        [candidate(24, 30, 2.0), candidate(12, 60, 2.0), candidate(12, 15, 2.0)]
    )
    assert (selected.window_hours, selected.horizon_minutes) == (12, 15)


def test_ttl_expires_at_the_frozen_timestamp():
    expert = _expert(window_hours=6)
    guardian = ExpirationGuardian()
    guardian.add(expert)

    assert guardian.evaluate(expert, expert.expires_at - timedelta(seconds=1), similarity=1.0) is None
    expired = guardian.evaluate(expert, expert.expires_at, similarity=1.0)
    assert expired.reason == "TTL"
    assert not guardian.is_active(expert, expert.expires_at)


def test_regime_drift_requires_three_distinct_consecutive_snapshots():
    expert = _expert(threshold=0.0)
    guardian = ExpirationGuardian()
    guardian.add(expert)

    first = expert.created_at + timedelta(minutes=15)
    assert guardian.evaluate(expert, first, similarity=-0.1) is None
    assert guardian.evaluate(expert, first, similarity=-0.1) is None
    assert guardian.evaluate(expert, first + timedelta(minutes=15), similarity=-0.1) is None
    expired = guardian.evaluate(expert, first + timedelta(minutes=30), similarity=-0.1)
    assert expired.reason == "REGIME_DRIFT"


def test_edge_decay_uses_only_post_creation_resolved_outcomes_and_has_priority():
    expert = _expert(threshold=0.0)
    guardian = ExpirationGuardian()
    guardian.add(expert)
    now = expert.created_at + timedelta(hours=2)
    pre_creation = [
        ResolvedOutcome(
            expert.created_at - timedelta(minutes=30),
            expert.created_at + timedelta(minutes=i + 1),
            -20.0,
        )
        for i in range(12)
    ]
    assert guardian.evaluate(expert, now, similarity=1.0, resolved_outcomes=pre_creation) is None

    post_creation = [
        ResolvedOutcome(
            expert.created_at + timedelta(minutes=i),
            expert.created_at + timedelta(minutes=i + 16),
            -3.0,
        )
        for i in range(12)
    ]
    # This is also the third low-similarity snapshot; EDGE_DECAY wins frozen priority.
    guardian.evaluate(expert, now + timedelta(minutes=15), similarity=-1.0)
    guardian.evaluate(expert, now + timedelta(minutes=30), similarity=-1.0)
    expired = guardian.evaluate(
        expert,
        now + timedelta(minutes=45),
        similarity=-1.0,
        resolved_outcomes=post_creation,
    )
    assert expired.reason == "EDGE_DECAY"


def test_expiration_is_permanent_and_metadata_is_immutable_and_unique():
    expert = _expert(window_hours=6)
    guardian = ExpirationGuardian()
    guardian.add(expert)
    expired = guardian.evaluate(expert, expert.expires_at, similarity=1.0)

    assert guardian.evaluate(expert, expert.expires_at + timedelta(days=1), similarity=1.0) is expired
    assert not guardian.is_active(expert, expert.expires_at + timedelta(days=1))
    with pytest.raises(FrozenInstanceError):
        expert.window_hours = 72
    with pytest.raises(ValueError, match="duplicate instance_id"):
        guardian.registry.register(expert)


def test_attribution_freezes_required_instance_metadata():
    expert = _expert()
    record = expert.attribution(
        decision_id="decision-1",
        decision_at=expert.created_at + timedelta(minutes=15),
        similarity=0.75,
        expected_edge_bps=4.5,
        decision="LONG",
        reason="MODEL_THRESHOLD",
        symbol="BTCUSDT",
        confidence=0.8,
    )
    required = {
        "decision_id",
        "decision_at",
        "model_family",
        "model_version",
        "model_instance_id",
        "training_window",
        "validation_window",
        "created_at",
        "expires_at",
        "regime_similarity_at_decision",
        "expected_edge_bps",
        "expected_direction",
        "horizon_minutes",
        "decision",
        "reason",
    }
    assert required.issubset(record)
    assert record["model_instance_id"] == expert.instance_id
    assert record["expected_edge_bps"] == 4.5


def test_temporal_block_bootstrap_is_deterministic_and_synchronized():
    timestamps = []
    values = []
    start = datetime(2023, 11, 1, tzinfo=UTC)
    for hour in range(8):
        for symbol_value in (1.0, 2.0, 3.0):
            timestamps.append(start + timedelta(hours=hour))
            values.append(symbol_value + hour / 10)

    first = temporal_block_bootstrap(timestamps, values)
    second = temporal_block_bootstrap(timestamps, values)
    np.testing.assert_array_equal(first.draws, second.draws)
    assert first.probability_positive == 1.0
    assert first.ci_lower <= first.observed_mean <= first.ci_upper
