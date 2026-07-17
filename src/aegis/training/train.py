"""Offline model-training boundary; no estimator is implemented here."""

from dataclasses import dataclass
from typing import Protocol

from .dataset import TrainingDataset


@dataclass(frozen=True)
class ModelArtifact:
    artifact_id: str
    model_family: str
    dataset_id: str
    feature_schema_version: str
    artifact_hash: str


class Trainer(Protocol):
    def train(self, dataset: TrainingDataset) -> ModelArtifact:
        """TODO: train candidates under an approved temporal protocol."""
        ...
