"""Offline evaluation boundary; metrics and acceptance rules remain TODO."""

from dataclasses import dataclass
from typing import Mapping, Protocol

from .dataset import TrainingDataset
from .train import ModelArtifact


@dataclass(frozen=True)
class EvaluationReport:
    report_id: str
    artifact_id: str
    dataset_id: str
    metrics: Mapping[str, float]
    accepted: bool
    report_hash: str


class ModelEvaluator(Protocol):
    def evaluate(
        self,
        artifact: ModelArtifact,
        dataset: TrainingDataset,
    ) -> EvaluationReport:
        """TODO: evaluate without selecting on protected evidence."""
        ...
