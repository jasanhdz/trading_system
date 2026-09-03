"""Purged temporal training contract for a genuine LONG opportunity model."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Mapping, Sequence

from ..features import FEATURE_HASH, FEATURE_NAMES, FEATURE_SCHEMA_VERSION, FrozenNormalizer
from .dataset import TrainingDataset
from .short_opportunity import (
    ShortOpportunityArtifact,
    ShortOpportunityTrainingContract,
    ShortOpportunityTrainingRow,
    fit_short_opportunity_model,
    short_opportunity_artifact_from_mapping,
    short_opportunity_artifact_mapping,
)


@dataclass(frozen=True)
class LongOpportunityTrainingRow:
    timestamp: datetime
    symbol: str
    features: tuple[float, ...]
    clean_opportunity: bool
    mae_fraction: float

    def __post_init__(self) -> None:
        if self.timestamp.tzinfo is None or not self.symbol:
            raise ValueError("LONG opportunity row identity is invalid")
        if len(self.features) != len(FEATURE_NAMES):
            raise ValueError("LONG opportunity row must preserve the 83-feature contract")
        if not all(math.isfinite(value) for value in self.features):
            raise ValueError("LONG opportunity row contains non-finite features")
        if not math.isfinite(self.mae_fraction) or self.mae_fraction < 0.0:
            raise ValueError("LONG opportunity row MAE is invalid")


@dataclass(frozen=True)
class LongOpportunityTrainingContract:
    schema_version: str
    feature_schema_version: str
    feature_hash: str
    maximum_acceptable_mae: float
    model_candidate_id: str
    model_parameters: Mapping[str, object]
    seed: int
    minimum_embargo_minutes: int
    minimum_symbol_calibration_rows: int
    symbol_calibration_shrinkage_rows: int

    def __post_init__(self) -> None:
        if self.schema_version != "aegis-long-opportunity-training-contract-v1":
            raise ValueError("unsupported LONG opportunity training schema")
        if (
            self.feature_schema_version != FEATURE_SCHEMA_VERSION
            or self.feature_hash != FEATURE_HASH
        ):
            raise ValueError("LONG opportunity training must preserve aegis-features-v2")
        if not 0.0 < self.maximum_acceptable_mae < 1.0:
            raise ValueError("LONG maximum acceptable MAE is invalid")
        if self.model_candidate_id not in {
            "eqm_logistic_clean_baseline",
            "eqm_random_forest_clean",
            "eqm_hgb_clean",
        } or not self.model_parameters:
            raise ValueError("LONG opportunity model contract is invalid")


@dataclass(frozen=True)
class LongOpportunityArtifact:
    schema_version: str
    direction: str
    learner: ShortOpportunityArtifact

    def __post_init__(self) -> None:
        if (
            self.schema_version != "aegis-long-opportunity-artifact-v1"
            or self.direction != "LONG"
        ):
            raise ValueError("unsupported LONG opportunity artifact")

    @property
    def feature_names(self) -> tuple[str, ...]:
        return self.learner.feature_names

    @property
    def model_candidate_id(self) -> str:
        return self.learner.model_candidate_id

    @property
    def scoring_metrics(self):
        return self.learner.scoring_metrics

    def raw_probability(self, features: Sequence[float]) -> float:
        return self.learner.raw_probability(features)

    def probability(self, symbol: str, features: Sequence[float]) -> float:
        return self.learner.probability(symbol, features)


def _as_short_rows(
    rows: Sequence[LongOpportunityTrainingRow],
) -> tuple[ShortOpportunityTrainingRow, ...]:
    return tuple(
        ShortOpportunityTrainingRow(
            timestamp=row.timestamp,
            symbol=row.symbol,
            features=row.features,
            clean_opportunity=row.clean_opportunity,
            mae_fraction=row.mae_fraction,
        )
        for row in rows
    )


def fit_long_opportunity_model(
    train: Sequence[LongOpportunityTrainingRow],
    calibration: Sequence[LongOpportunityTrainingRow],
    scoring: Sequence[LongOpportunityTrainingRow],
    contract: LongOpportunityTrainingContract,
    *,
    normalizer: FrozenNormalizer = FrozenNormalizer(),
) -> LongOpportunityArtifact:
    learner_contract = ShortOpportunityTrainingContract(
        schema_version="aegis-short-opportunity-training-contract-v1",
        feature_schema_version=contract.feature_schema_version,
        feature_hash=contract.feature_hash,
        maximum_acceptable_mae=contract.maximum_acceptable_mae,
        model_candidate_id=contract.model_candidate_id,
        model_parameters=contract.model_parameters,
        seed=contract.seed,
        minimum_embargo_minutes=contract.minimum_embargo_minutes,
        minimum_symbol_calibration_rows=contract.minimum_symbol_calibration_rows,
        symbol_calibration_shrinkage_rows=contract.symbol_calibration_shrinkage_rows,
    )
    learner = fit_short_opportunity_model(
        _as_short_rows(train),
        _as_short_rows(calibration),
        _as_short_rows(scoring),
        learner_contract,
        normalizer=normalizer,
    )
    return LongOpportunityArtifact(
        schema_version="aegis-long-opportunity-artifact-v1",
        direction="LONG",
        learner=learner,
    )


def long_opportunity_artifact_mapping(
    artifact: LongOpportunityArtifact,
) -> Mapping[str, object]:
    learner = dict(short_opportunity_artifact_mapping(artifact.learner))
    learner["schema_version"] = "aegis-short-opportunity-artifact-v1"
    return {
        "schema_version": artifact.schema_version,
        "direction": artifact.direction,
        "learner": learner,
    }


def write_long_opportunity_artifact(
    path: Path,
    artifact: LongOpportunityArtifact,
) -> None:
    from .run_state import atomic_write_json

    atomic_write_json(path, long_opportunity_artifact_mapping(artifact))


def load_long_opportunity_artifact(path: Path) -> LongOpportunityArtifact:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        learner = value["learner"]
        if not isinstance(learner, Mapping):
            raise ValueError("LONG opportunity learner is invalid")
        artifact = LongOpportunityArtifact(
            schema_version=str(value["schema_version"]),
            direction=str(value["direction"]),
            learner=short_opportunity_artifact_from_mapping(learner),
        )
        return artifact
    except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
        raise ValueError("LONG opportunity artifact is invalid") from exc


def rows_from_long_training_dataset(
    dataset: TrainingDataset,
) -> tuple[LongOpportunityTrainingRow, ...]:
    if (
        dataset.feature_schema_version != FEATURE_SCHEMA_VERSION
        or dataset.feature_hash != FEATURE_HASH
    ):
        raise ValueError("LONG training dataset feature authority mismatch")
    return tuple(
        LongOpportunityTrainingRow(
            timestamp=row.timestamp,
            symbol=row.symbol,
            features=row.features,
            clean_opportunity=row.target.clean_quality >= 0.5,
            mae_fraction=row.target.qmae,
        )
        for row in dataset.rows
        if row.target.label_valid
    )
