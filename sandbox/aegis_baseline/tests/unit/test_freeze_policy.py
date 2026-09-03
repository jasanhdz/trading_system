from dataclasses import replace

import pytest

from aegis.decision import GlobalSelectionPolicy
from aegis.freeze import (
    REQUIRED_FREEZE_COMPONENTS, BundleLifecycleState, FreezeValidationError,
    FrozenSelectionPolicy, SystemFreeze, validate_lifecycle_transition,
)


def _policy() -> FrozenSelectionPolicy:
    return FrozenSelectionPolicy.derive(
        policy_id="policy-1", bundle_id="bundle-1", bundle_hash="a" * 64,
        dataset_hash="b" * 64, econ_report_hash="c" * 64,
        scores=(0.1, 0.2, 0.3, 0.4, 0.5), budget_fraction=0.2,
    )


def test_selection_policy_is_hashed_absolute_and_runtime_compatible() -> None:
    policy = _policy()
    expected = {"bundle_hash": "a" * 64, "dataset_hash": "b" * 64, "econ_report_hash": "c" * 64}
    policy.validate(scores=(0.1, 0.2, 0.3, 0.4, 0.5), expected_hashes=expected)
    assert GlobalSelectionPolicy.from_frozen(policy).selection_threshold == policy.threshold
    with pytest.raises(FreezeValidationError, match="drift"):
        policy.validate(scores=(0.1, 0.2, 0.3, 0.4, 0.6), expected_hashes=expected)
    with pytest.raises(FreezeValidationError, match="dataset_hash"):
        policy.validate(scores=(0.1, 0.2, 0.3, 0.4, 0.5), expected_hashes={**expected, "dataset_hash": "x"})


def test_system_freeze_binds_every_required_component() -> None:
    components = {name: f"{index:064x}" for index, name in enumerate(REQUIRED_FREEZE_COMPONENTS, start=1)}
    freeze = SystemFreeze.create(
        freeze_id="freeze-1", component_hashes=components, universe_id="universe-1",
        symbol_set_hash="d" * 64, timeframe="5m", code_commit="e" * 40,
        environment={"python": "3.12", "numpy": "2"},
    )
    freeze.validate(components)
    with pytest.raises(FreezeValidationError):
        SystemFreeze.create(
            freeze_id="bad", component_hashes={"dataset": "a"}, universe_id="universe-1",
            symbol_set_hash="d" * 64, timeframe="5m", code_commit="e" * 40, environment={},
        )
    with pytest.raises(FreezeValidationError):
        replace(freeze, content_hash="0" * 64).validate(components)


def test_bundle_lifecycle_is_forward_only() -> None:
    validate_lifecycle_transition(BundleLifecycleState.EXPERIMENTAL, BundleLifecycleState.CANDIDATE)
    validate_lifecycle_transition(BundleLifecycleState.CANDIDATE, BundleLifecycleState.SHADOW_APPROVED)
    with pytest.raises(FreezeValidationError):
        validate_lifecycle_transition(BundleLifecycleState.EXPERIMENTAL, BundleLifecycleState.SHADOW_APPROVED)
