"""Hashed lifecycle, selection-policy, and system-freeze contracts."""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import Any, Mapping, Sequence

from .domain import TradeSide
from .utils import Sha256HashProvider


class FreezeValidationError(ValueError):
    pass


class BundleLifecycleState(str, Enum):
    EXPERIMENTAL = "EXPERIMENTAL"
    CANDIDATE = "CANDIDATE"
    SHADOW_APPROVED = "SHADOW_APPROVED"
    LIVE_APPROVED = "LIVE_APPROVED"


_TRANSITIONS = {
    BundleLifecycleState.EXPERIMENTAL: {BundleLifecycleState.CANDIDATE},
    BundleLifecycleState.CANDIDATE: {BundleLifecycleState.SHADOW_APPROVED},
    BundleLifecycleState.SHADOW_APPROVED: {BundleLifecycleState.LIVE_APPROVED},
    BundleLifecycleState.LIVE_APPROVED: set(),
}


def validate_lifecycle_transition(current: BundleLifecycleState, target: BundleLifecycleState) -> None:
    if target not in _TRANSITIONS[current]:
        raise FreezeValidationError(f"invalid bundle lifecycle transition: {current.value}->{target.value}")


@dataclass(frozen=True)
class FrozenSelectionPolicy:
    schema_version: str
    policy_id: str
    bundle_id: str
    bundle_hash: str
    dataset_hash: str
    econ_report_hash: str
    side: TradeSide
    score_unit: str
    budget_fraction: float
    threshold: float
    content_hash: str

    @classmethod
    def derive(
        cls, *, policy_id: str, bundle_id: str, bundle_hash: str, dataset_hash: str,
        econ_report_hash: str, scores: Sequence[float], budget_fraction: float,
    ) -> "FrozenSelectionPolicy":
        if not scores or not 0.0 < budget_fraction < 1.0 or not all(math.isfinite(value) for value in scores):
            raise FreezeValidationError("selection policy derivation inputs are invalid")
        ordered = sorted(float(value) for value in scores)
        position = (len(ordered) - 1) * (1.0 - budget_fraction)
        lower = int(math.floor(position)); upper = int(math.ceil(position))
        threshold = ordered[lower] if lower == upper else ordered[lower] + (position - lower) * (ordered[upper] - ordered[lower])
        unsigned = {
            "schema_version": "aegis-selection-policy-v1", "policy_id": policy_id,
            "bundle_id": bundle_id, "bundle_hash": bundle_hash, "dataset_hash": dataset_hash,
            "econ_report_hash": econ_report_hash, "side": TradeSide.SHORT,
            "score_unit": "EXPECTED_CLEAN_RETURN_FRACTION", "budget_fraction": budget_fraction,
            "threshold": threshold,
        }
        digest = Sha256HashProvider().digest_value(unsigned)
        return cls(**unsigned, content_hash=digest)

    def validate(self, *, scores: Sequence[float], expected_hashes: Mapping[str, str]) -> None:
        unsigned = {key: value for key, value in self.__dict__.items() if key != "content_hash"}
        if Sha256HashProvider().digest_value(unsigned) != self.content_hash:
            raise FreezeValidationError("selection policy content hash mismatch")
        for key in ("bundle_hash", "dataset_hash", "econ_report_hash"):
            if expected_hashes.get(key) != getattr(self, key):
                raise FreezeValidationError(f"selection policy {key} mismatch")
        recomputed = self.derive(
            policy_id=self.policy_id, bundle_id=self.bundle_id, bundle_hash=self.bundle_hash,
            dataset_hash=self.dataset_hash, econ_report_hash=self.econ_report_hash,
            scores=scores, budget_fraction=self.budget_fraction,
        )
        if abs(recomputed.threshold - self.threshold) > 1e-9:
            raise FreezeValidationError("selection policy threshold drift")


REQUIRED_FREEZE_COMPONENTS = (
    "dataset", "snapshots", "features", "labels", "folds", "normalizer", "models",
    "calibrators", "selection_policy", "econ_report", "bundle", "promotion_criteria",
)


@dataclass(frozen=True)
class SystemFreeze:
    schema_version: str
    freeze_id: str
    component_hashes: Mapping[str, str]
    universe_id: str
    symbol_set_hash: str
    timeframe: str
    code_commit: str
    environment: Mapping[str, str]
    content_hash: str

    @classmethod
    def create(
        cls, *, freeze_id: str, component_hashes: Mapping[str, str], universe_id: str,
        symbol_set_hash: str, timeframe: str, code_commit: str, environment: Mapping[str, str],
    ) -> "SystemFreeze":
        if set(component_hashes) != set(REQUIRED_FREEZE_COMPONENTS):
            raise FreezeValidationError("system freeze component set is incomplete")
        unsigned: dict[str, Any] = {
            "schema_version": "aegis-system-freeze-v1", "freeze_id": freeze_id,
            "component_hashes": dict(sorted(component_hashes.items())), "universe_id": universe_id,
            "symbol_set_hash": symbol_set_hash, "timeframe": timeframe, "code_commit": code_commit,
            "environment": dict(sorted(environment.items())),
        }
        return cls(**unsigned, content_hash=Sha256HashProvider().digest_value(unsigned))

    def validate(self, expected_components: Mapping[str, str]) -> None:
        recreated = self.create(
            freeze_id=self.freeze_id, component_hashes=self.component_hashes,
            universe_id=self.universe_id, symbol_set_hash=self.symbol_set_hash,
            timeframe=self.timeframe, code_commit=self.code_commit, environment=self.environment,
        )
        if recreated.content_hash != self.content_hash or dict(expected_components) != dict(self.component_hashes):
            raise FreezeValidationError("system freeze mismatch")
