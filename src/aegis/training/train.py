"""Deterministic offline linear reference training."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Protocol

import numpy as np

from .dataset import TrainingDataset
from ..utils import HashProvider


@dataclass(frozen=True)
class ModelArtifact:
    artifact_id: str
    model_family: str
    dataset_id: str
    feature_schema_version: str
    feature_hash: str
    coefficients: Mapping[str, tuple[float, ...]]
    intercepts: Mapping[str, float]
    artifact_hash: str


class Trainer(Protocol):
    def train(self, dataset: TrainingDataset) -> ModelArtifact: ...


@dataclass(frozen=True)
class DeterministicLinearTrainer:
    hashing: HashProvider
    ridge: float = 1e-6

    def train(self, dataset: TrainingDataset) -> ModelArtifact:
        if self.ridge < 0:
            raise ValueError("ridge cannot be negative")
        x = np.asarray([row.features for row in dataset.rows], dtype=np.float64)
        if x.ndim != 2 or not np.isfinite(x).all():
            raise ValueError("training features must be a finite matrix")
        targets = {
            "direction": np.asarray([row.target.direction for row in dataset.rows]),
            "expected_return": np.asarray([row.target.expected_return for row in dataset.rows]),
            "tail_event": np.asarray([row.target.tail_event for row in dataset.rows]),
            "qmae": np.asarray([row.target.qmae for row in dataset.rows]),
            "clean_quality": np.asarray([row.target.clean_quality for row in dataset.rows]),
        }
        augmented = np.column_stack([np.ones(len(x)), x])
        penalty = np.eye(augmented.shape[1]) * self.ridge
        penalty[0, 0] = 0.0
        gram = augmented.T @ augmented + penalty
        coefficients: dict[str, tuple[float, ...]] = {}
        intercepts: dict[str, float] = {}
        for name, y in targets.items():
            fitted = np.linalg.solve(gram, augmented.T @ y)
            intercepts[name] = float(fitted[0])
            coefficients[name] = tuple(float(value) for value in fitted[1:])
        payload = {"dataset": dataset.dataset_id, "features": dataset.feature_hash, "coefficients": coefficients, "intercepts": intercepts}
        artifact_hash = self.hashing.digest_value(payload)
        return ModelArtifact(f"linear-{artifact_hash[:20]}", "deterministic-linear-v1", dataset.dataset_id,
                             dataset.feature_schema_version, dataset.feature_hash, coefficients, intercepts, artifact_hash)
