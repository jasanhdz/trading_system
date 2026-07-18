"""Causal offline datasets built with the production feature pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Mapping, Protocol, Sequence

from ..domain import MarketSnapshot, Regime
from ..features import FeaturePipeline
from ..utils import HashProvider


@dataclass(frozen=True)
class TrainingTarget:
    direction: float
    expected_return: float
    tail_event: float
    qmae: float
    clean_quality: float
    net_quality_after_costs: float = 0.0
    bad_entry: float = 0.0
    label_valid: bool = True


@dataclass(frozen=True)
class TrainingRow:
    timestamp: datetime
    symbol: str
    features: tuple[float, ...]
    target: TrainingTarget
    regime: Regime = Regime.UNKNOWN


@dataclass(frozen=True)
class TrainingDataset:
    dataset_id: str
    schema_version: str
    feature_schema_version: str
    feature_hash: str
    symbols: tuple[str, ...]
    timeframe: str
    rows: tuple[TrainingRow, ...]
    artifact_hash: str

    @property
    def row_count(self) -> int:
        return len(self.rows)


class DatasetBuilder(Protocol):
    def build(self, dataset_id: str, snapshots: Sequence[MarketSnapshot], targets: Mapping[tuple[datetime, str], TrainingTarget]) -> TrainingDataset: ...


@dataclass(frozen=True)
class CausalDatasetBuilder:
    feature_pipeline: FeaturePipeline
    hashing: HashProvider
    symbols: tuple[str, ...]
    timeframe: str

    def build(self, dataset_id: str, snapshots: Sequence[MarketSnapshot], targets: Mapping[tuple[datetime, str], TrainingTarget]) -> TrainingDataset:
        rows: list[TrainingRow] = []
        for snapshot in sorted(snapshots, key=lambda item: item.closed_at):
            batch = self.feature_pipeline.transform(snapshot)
            for feature_row in batch.rows:
                key = (snapshot.closed_at, feature_row.symbol)
                if key not in targets:
                    continue
                rows.append(TrainingRow(snapshot.closed_at, feature_row.symbol, feature_row.normalized_values, targets[key]))
        if not rows:
            raise ValueError("training dataset has no aligned causal targets")
        payload = {"dataset_id": dataset_id, "feature_hash": self.feature_pipeline.feature_hash, "rows": rows}
        return TrainingDataset(dataset_id, "aegis-training-dataset-v1", self.feature_pipeline.schema_version,
                               self.feature_pipeline.feature_hash, self.symbols, self.timeframe,
                               tuple(rows), self.hashing.digest_value(payload))


def walk_forward_splits(dataset: TrainingDataset, fold_count: int = 4, embargo: timedelta = timedelta(minutes=120)) -> tuple[tuple[tuple[int, ...], tuple[int, ...]], ...]:
    """Expanding temporal folds; no row after validation enters training."""
    if fold_count < 1:
        raise ValueError("fold_count must be positive")
    timestamps = sorted({row.timestamp for row in dataset.rows})
    if len(timestamps) < fold_count + 2:
        raise ValueError("insufficient timestamps for walk-forward evaluation")
    folds = []
    for fold in range(fold_count):
        train_fraction = 0.50 + fold * (0.30 / max(1, fold_count - 1))
        validation_fraction = min(0.90, train_fraction + 0.10)
        train_end = timestamps[min(len(timestamps) - 2, int(len(timestamps) * train_fraction))]
        validation_end = timestamps[min(len(timestamps) - 1, int(len(timestamps) * validation_fraction))]
        train = tuple(index for index, row in enumerate(dataset.rows) if row.timestamp <= train_end - embargo)
        validation = tuple(index for index, row in enumerate(dataset.rows) if train_end < row.timestamp <= validation_end)
        if not train or not validation:
            raise ValueError("empty temporal fold")
        folds.append((train, validation))
    return tuple(folds)
