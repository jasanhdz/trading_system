from dataclasses import FrozenInstanceError
from datetime import datetime, timezone

import pytest

from aegis_strategy_router.domain.serialization import content_hash
from aegis_strategy_router.domain.types import (
    DataStatus,
    FeatureObservation,
    FeatureSet,
    FutureLeakageError,
    Timeframe,
    TimeframeSnapshot,
    UndeclaredFeatureAccess,
)


NOW = datetime(2026, 8, 17, 12, 0, tzinfo=timezone.utc)


def test_feature_contract_is_immutable_and_allowlisted() -> None:
    observation = FeatureObservation(
        "tf1m__return_1_bps", 2.0, NOW, NOW, "owner", "v1", DataStatus.AVAILABLE
    )
    features = FeatureSet((observation,), content_hash({"schema": 1}))
    with pytest.raises(FrozenInstanceError):
        observation.value = 5.0  # type: ignore[misc]
    assert features.view({observation.name}).get(observation.name).value == 2.0
    with pytest.raises(UndeclaredFeatureAccess):
        features.view(set()).get(observation.name)


def test_future_feature_fails_closed() -> None:
    future = datetime(2026, 8, 17, 12, 1, tzinfo=timezone.utc)
    observation = FeatureObservation(
        "tf1m__return_1_bps", 2.0, NOW, future, "owner", "v1", DataStatus.AVAILABLE
    )
    state = TimeframeSnapshot(
        Timeframe.M1, DataStatus.AVAILABLE, 100, 99, NOW, None,
        FeatureSet((observation,), content_hash({"schema": 1})), None,
    )
    with pytest.raises(FutureLeakageError):
        state.assert_causal(NOW)


def test_unknown_feature_cannot_expose_value() -> None:
    with pytest.raises(ValueError, match="cannot expose"):
        FeatureObservation(
            "tf1d__rsi12", 50.0, NOW, None, "owner", "v1",
            DataStatus.UNKNOWN, "WARMUP",
        )
