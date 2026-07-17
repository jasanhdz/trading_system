"""Versioned training-dataset boundary; dataset construction remains TODO."""

from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class TrainingDataset:
    dataset_id: str
    schema_version: str
    feature_schema_version: str
    symbols: tuple[str, ...]
    timeframe: str
    row_count: int
    artifact_hash: str


class DatasetBuilder(Protocol):
    def build(self, dataset_id: str) -> TrainingDataset:
        """TODO: construct a causal, versioned dataset in offline mode only."""
        ...
