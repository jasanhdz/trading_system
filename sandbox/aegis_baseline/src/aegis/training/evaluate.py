"""Offline metrics that never auto-promote artifacts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Protocol

import numpy as np

from .dataset import TrainingDataset
from .train import ModelArtifact
from ..utils import HashProvider


@dataclass(frozen=True)
class EvaluationReport:
    report_id: str
    artifact_id: str
    dataset_id: str
    metrics: Mapping[str, float]
    accepted: bool
    report_hash: str


class ModelEvaluator(Protocol):
    def evaluate(self, artifact: ModelArtifact, dataset: TrainingDataset) -> EvaluationReport: ...


@dataclass(frozen=True)
class OfflineModelEvaluator:
    hashing: HashProvider
    maximum_direction_brier: float = 0.30

    def evaluate(self, artifact: ModelArtifact, dataset: TrainingDataset) -> EvaluationReport:
        if artifact.feature_hash != dataset.feature_hash:
            raise ValueError("artifact/dataset feature mismatch")
        x = np.asarray([row.features for row in dataset.rows], dtype=np.float64)
        metrics: dict[str, float] = {}
        target_accessors = {
            "direction": lambda row: row.target.direction,
            "expected_return": lambda row: row.target.expected_return,
            "tail_event": lambda row: row.target.tail_event,
            "qmae": lambda row: row.target.qmae,
            "clean_quality": lambda row: row.target.clean_quality,
        }
        for name, accessor in target_accessors.items():
            predicted = x @ np.asarray(artifact.coefficients[name]) + artifact.intercepts[name]
            actual = np.asarray([accessor(row) for row in dataset.rows])
            metrics[f"{name}_mse"] = float(np.mean(np.square(predicted - actual)))
        accepted = metrics["direction_mse"] <= self.maximum_direction_brier
        report_hash = self.hashing.digest_value({"artifact": artifact.artifact_hash, "dataset": dataset.artifact_hash, "metrics": metrics, "accepted": accepted})
        return EvaluationReport(f"evaluation-{report_hash[:20]}", artifact.artifact_id, dataset.dataset_id, metrics, accepted, report_hash)
