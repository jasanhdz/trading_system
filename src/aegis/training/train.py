"""Deterministic offline linear reference training."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Protocol

import numpy as np

from .dataset import TrainingDataset
from ..models import CalibrationMethod, CalibratorSpec
from ..utils import HashProvider


def calibration_metrics(probabilities: np.ndarray, labels: np.ndarray, *, bins: int = 10) -> tuple[float, float]:
    """Return fixed-bin ECE and Brier for held-out binary probabilities."""
    values = np.asarray(probabilities, dtype=np.float64)
    actual = np.asarray(labels, dtype=np.float64)
    if values.ndim != 1 or values.shape != actual.shape or not len(values):
        raise ValueError("calibration arrays must be non-empty one-dimensional peers")
    if not np.isfinite(values).all() or not np.isfinite(actual).all() or np.any((values < 0) | (values > 1)):
        raise ValueError("calibration inputs must be finite probabilities")
    brier = float(np.mean(np.square(values - actual)))
    ece = 0.0
    boundaries = np.linspace(0.0, 1.0, bins + 1)
    for index in range(bins):
        include = (values >= boundaries[index]) & (values <= boundaries[index + 1] if index == bins - 1 else values < boundaries[index + 1])
        if np.any(include):
            ece += float(np.mean(include)) * abs(float(np.mean(values[include])) - float(np.mean(actual[include])))
    return ece, brier


def fit_platt_calibrator(probabilities: np.ndarray, labels: np.ndarray) -> CalibratorSpec:
    """Fit deterministic Platt scaling on a held-out or OOF probability vector."""
    values = np.asarray(probabilities, dtype=np.float64)
    actual = np.asarray(labels, dtype=np.float64)
    if values.shape != actual.shape or values.ndim != 1 or len(values) < 8:
        raise ValueError("Platt calibration requires at least eight held-out rows")
    if set(np.unique(actual)) - {0.0, 1.0} or len(np.unique(actual)) < 2:
        raise ValueError("Platt calibration requires both binary classes")
    clipped = np.clip(values, 1e-12, 1.0 - 1e-12)
    logits = np.log(clipped / (1.0 - clipped))
    design = np.column_stack([logits, np.ones(len(logits))])
    parameters = np.asarray([1.0, 0.0], dtype=np.float64)
    for _ in range(100):
        linear = np.clip(design @ parameters, -40.0, 40.0)
        fitted = 1.0 / (1.0 + np.exp(-linear))
        weights = np.maximum(fitted * (1.0 - fitted), 1e-9)
        gradient = design.T @ (fitted - actual)
        hessian = design.T @ (design * weights[:, None]) + np.eye(2) * 1e-8
        update = np.linalg.solve(hessian, gradient)
        parameters -= update
        if float(np.max(np.abs(update))) <= 1e-12:
            break
    calibrated = 1.0 / (1.0 + np.exp(-np.clip(design @ parameters, -40.0, 40.0)))
    ece, brier = calibration_metrics(calibrated, actual)
    return CalibratorSpec(
        CalibrationMethod.PLATT, ece, brier, len(values),
        parameters=(float(parameters[0]), float(parameters[1])),
    )


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
