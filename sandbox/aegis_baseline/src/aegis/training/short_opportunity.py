"""Purged temporal training contract for a real SHORT opportunity probability."""

from __future__ import annotations

import math
import json
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score

from ..features import (
    FEATURE_HASH,
    FEATURE_NAMES,
    FEATURE_SCHEMA_VERSION,
    FrozenNormalizer,
)
from ..models import CalibrationMethod, CalibratorSpec
from ..tree_models import TreeEnsemble
from ..utils import canonical_json
from ..research.entry_quality import HierarchicalProbabilityCalibrator
from .dataset import TrainingDataset
from .train import calibration_metrics, fit_platt_calibrator


@dataclass(frozen=True)
class ShortOpportunityTrainingRow:
    timestamp: datetime
    symbol: str
    features: tuple[float, ...]
    clean_opportunity: bool
    mae_fraction: float

    def __post_init__(self) -> None:
        if self.timestamp.tzinfo is None or not self.symbol:
            raise ValueError("opportunity row identity is invalid")
        if len(self.features) != len(FEATURE_NAMES):
            raise ValueError("opportunity row must preserve the 83-feature contract")
        if not all(math.isfinite(value) for value in self.features):
            raise ValueError("opportunity row contains non-finite features")
        if not math.isfinite(self.mae_fraction) or self.mae_fraction < 0.0:
            raise ValueError("opportunity row MAE is invalid")


@dataclass(frozen=True)
class ShortOpportunityTrainingContract:
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
    probability_semantics: str = "CLEAN_ENTRY_LOW_MAE_H12"

    def __post_init__(self) -> None:
        if self.schema_version != "aegis-short-opportunity-training-contract-v1":
            raise ValueError("unsupported SHORT opportunity training schema")
        if (
            self.feature_schema_version != FEATURE_SCHEMA_VERSION
            or self.feature_hash != FEATURE_HASH
        ):
            raise ValueError(
                "SHORT opportunity training must preserve aegis-features-v2"
            )
        if not 0.0 < self.maximum_acceptable_mae < 1.0:
            raise ValueError("maximum acceptable MAE is invalid")
        if (
            self.model_candidate_id
            not in {
                "eqm_logistic_clean_baseline",
                "eqm_random_forest_clean",
                "eqm_hgb_clean",
            }
            or not self.model_parameters
        ):
            raise ValueError("SHORT opportunity model contract is invalid")
        if self.minimum_embargo_minutes <= 0:
            raise ValueError("temporal embargo must be positive")
        if (
            self.minimum_symbol_calibration_rows < 8
            or self.symbol_calibration_shrinkage_rows <= 0
        ):
            raise ValueError("symbol calibration sample contract is invalid")
        if self.probability_semantics not in {
            "CLEAN_ENTRY_LOW_MAE_H12",
            "TERMINAL_NET_POSITIVE_H12_AFTER_COSTS",
        }:
            raise ValueError("SHORT opportunity probability semantics are invalid")


@dataclass(frozen=True)
class ShortOpportunityMetrics:
    row_count: int
    positive_rate: float
    average_precision: float
    ece: float
    brier: float
    per_symbol: Mapping[str, Mapping[str, float | int]]


@dataclass(frozen=True)
class ShortOpportunityArtifact:
    schema_version: str
    feature_schema_version: str
    feature_hash: str
    feature_names: tuple[str, ...]
    model_candidate_id: str
    coefficients: tuple[float, ...]
    intercept: float
    tree_ensemble: TreeEnsemble | None
    normalizer: FrozenNormalizer
    calibrator: HierarchicalProbabilityCalibrator
    scoring_metrics: ShortOpportunityMetrics
    probability_semantics: str = "CLEAN_ENTRY_LOW_MAE_H12"

    def __post_init__(self) -> None:
        if self.schema_version != "aegis-short-opportunity-artifact-v1":
            raise ValueError("unsupported SHORT opportunity artifact")
        if (
            self.feature_schema_version != FEATURE_SCHEMA_VERSION
            or self.feature_hash != FEATURE_HASH
        ):
            raise ValueError("SHORT opportunity artifact feature authority mismatch")
        if self.feature_names != FEATURE_NAMES:
            raise ValueError("SHORT opportunity artifact feature ordering mismatch")
        if self.tree_ensemble is None:
            if self.model_candidate_id != "eqm_logistic_clean_baseline" or len(
                self.coefficients
            ) != len(FEATURE_NAMES):
                raise ValueError("SHORT opportunity linear artifact is invalid")
        elif (
            self.model_candidate_id not in {"eqm_random_forest_clean", "eqm_hgb_clean"}
            or self.coefficients
            or self.tree_ensemble.feature_names != FEATURE_NAMES
        ):
            raise ValueError("SHORT opportunity tree artifact is invalid")
        if self.probability_semantics not in {
            "CLEAN_ENTRY_LOW_MAE_H12",
            "TERMINAL_NET_POSITIVE_H12_AFTER_COSTS",
        }:
            raise ValueError("SHORT opportunity artifact semantics are invalid")

    def raw_probability(self, features: Sequence[float]) -> float:
        if len(features) != len(self.feature_names) or not all(
            math.isfinite(value) for value in features
        ):
            raise ValueError("SHORT opportunity inference features are invalid")
        normalized = tuple(
            self.normalizer.normalize(name, float(value))[0]
            for name, value in zip(self.feature_names, features)
        )
        if self.tree_ensemble is not None:
            return self.tree_ensemble.evaluate(normalized)
        logit = self.intercept + math.fsum(
            coefficient * value
            for coefficient, value in zip(self.coefficients, normalized)
        )
        return 1.0 / (1.0 + math.exp(-max(-40.0, min(40.0, logit))))

    def probability(self, symbol: str, features: Sequence[float]) -> float:
        return self.calibrator.apply(symbol, self.raw_probability(features))


def _calibrator_to_mapping(value: CalibratorSpec) -> Mapping[str, object]:
    return {
        "method": value.method.value,
        "ece": value.ece,
        "brier": value.brier,
        "sample_count": value.sample_count,
        "parameters": list(value.parameters),
        "x": list(value.x),
        "y": list(value.y),
    }


def _calibrator_from_mapping(value: Mapping[str, object]) -> CalibratorSpec:
    return CalibratorSpec(
        method=CalibrationMethod(str(value["method"])),
        ece=float(value["ece"]),
        brier=float(value["brier"]),
        sample_count=int(value["sample_count"]),
        parameters=tuple(float(item) for item in value.get("parameters", ())),
        x=tuple(float(item) for item in value.get("x", ())),
        y=tuple(float(item) for item in value.get("y", ())),
    )


def short_opportunity_artifact_mapping(
    artifact: ShortOpportunityArtifact,
) -> Mapping[str, object]:
    return {
        "schema_version": artifact.schema_version,
        "feature_schema_version": artifact.feature_schema_version,
        "feature_hash": artifact.feature_hash,
        "feature_names": list(artifact.feature_names),
        "model_candidate_id": artifact.model_candidate_id,
        "probability_semantics": artifact.probability_semantics,
        "coefficients": list(artifact.coefficients),
        "intercept": artifact.intercept,
        "tree_ensemble": (
            artifact.tree_ensemble.to_payload()
            if artifact.tree_ensemble is not None
            else None
        ),
        "normalizer": {
            "means": {
                name: float(artifact.normalizer.means.get(name, 0.0))
                for name in artifact.feature_names
            },
            "scales": {
                name: float(artifact.normalizer.scales.get(name, 1.0))
                for name in artifact.feature_names
            },
            "clip_absolute": artifact.normalizer.clip_absolute,
        },
        "calibrator": {
            "schema_version": artifact.calibrator.schema_version,
            "global": _calibrator_to_mapping(artifact.calibrator.global_calibrator),
            "symbols": {
                symbol: _calibrator_to_mapping(value)
                for symbol, value in sorted(
                    artifact.calibrator.symbol_calibrators.items()
                )
            },
            "symbol_sample_counts": dict(
                sorted(artifact.calibrator.symbol_sample_counts.items())
            ),
            "shrinkage_sample_count": artifact.calibrator.shrinkage_sample_count,
        },
        "scoring_metrics": {
            "row_count": artifact.scoring_metrics.row_count,
            "positive_rate": artifact.scoring_metrics.positive_rate,
            "average_precision": artifact.scoring_metrics.average_precision,
            "ece": artifact.scoring_metrics.ece,
            "brier": artifact.scoring_metrics.brier,
            "per_symbol": artifact.scoring_metrics.per_symbol,
        },
    }


def write_short_opportunity_artifact(
    path: Path,
    artifact: ShortOpportunityArtifact,
) -> None:
    """Persist a deterministic research artifact; promotion is a separate step."""
    from .run_state import atomic_write_json

    atomic_write_json(path, short_opportunity_artifact_mapping(artifact))


def load_short_opportunity_artifact(path: Path) -> ShortOpportunityArtifact:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        return short_opportunity_artifact_from_mapping(value)
    except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
        raise ValueError("SHORT opportunity artifact is invalid") from exc


def short_opportunity_artifact_from_mapping(
    value: Mapping[str, object],
) -> ShortOpportunityArtifact:
    try:
        calibrator = value["calibrator"]
        metrics = value["scoring_metrics"]
        if not isinstance(calibrator, Mapping) or not isinstance(metrics, Mapping):
            raise ValueError("SHORT opportunity artifact sections are invalid")
        normalizer = value["normalizer"]
        if not isinstance(normalizer, Mapping):
            raise ValueError("SHORT opportunity normalizer is invalid")
        means = normalizer["means"]
        scales = normalizer["scales"]
        if not isinstance(means, Mapping) or not isinstance(scales, Mapping):
            raise ValueError("SHORT opportunity normalizer values are invalid")
        symbol_calibrators = calibrator["symbols"]
        symbol_sample_counts = calibrator["symbol_sample_counts"]
        per_symbol = metrics["per_symbol"]
        if (
            not isinstance(symbol_calibrators, Mapping)
            or not isinstance(symbol_sample_counts, Mapping)
            or not isinstance(per_symbol, Mapping)
        ):
            raise ValueError("SHORT opportunity calibration metrics are invalid")
        return ShortOpportunityArtifact(
            schema_version=str(value["schema_version"]),
            feature_schema_version=str(value["feature_schema_version"]),
            feature_hash=str(value["feature_hash"]),
            feature_names=tuple(str(item) for item in value["feature_names"]),
            model_candidate_id=str(
                value.get(
                    "model_candidate_id",
                    "eqm_logistic_clean_baseline",
                )
            ),
            coefficients=tuple(float(item) for item in value["coefficients"]),
            intercept=float(value["intercept"]),
            tree_ensemble=(
                TreeEnsemble.from_payload(value["tree_ensemble"])  # type: ignore[arg-type]
                if value.get("tree_ensemble") is not None
                else None
            ),
            normalizer=FrozenNormalizer(
                means={str(name): float(metric) for name, metric in means.items()},
                scales={str(name): float(metric) for name, metric in scales.items()},
                clip_absolute=float(normalizer["clip_absolute"]),
            ),
            calibrator=HierarchicalProbabilityCalibrator(
                schema_version=str(calibrator["schema_version"]),
                global_calibrator=_calibrator_from_mapping(calibrator["global"]),  # type: ignore[arg-type]
                symbol_calibrators={
                    str(symbol): _calibrator_from_mapping(spec)  # type: ignore[arg-type]
                    for symbol, spec in symbol_calibrators.items()
                },
                symbol_sample_counts={
                    str(symbol): int(count)
                    for symbol, count in symbol_sample_counts.items()
                },
                shrinkage_sample_count=int(calibrator["shrinkage_sample_count"]),
            ),
            scoring_metrics=ShortOpportunityMetrics(
                row_count=int(metrics["row_count"]),
                positive_rate=float(metrics["positive_rate"]),
                average_precision=float(metrics["average_precision"]),
                ece=float(metrics["ece"]),
                brier=float(metrics["brier"]),
                per_symbol={
                    str(symbol): {
                        str(name): (
                            float(metric) if isinstance(metric, float) else int(metric)
                        )
                        for name, metric in values.items()
                    }
                    for symbol, values in per_symbol.items()  # type: ignore[union-attr]
                },
            ),
            probability_semantics=str(
                value.get(
                    "probability_semantics",
                    "CLEAN_ENTRY_LOW_MAE_H12",
                )
            ),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("SHORT opportunity artifact is invalid") from exc


def _label(
    row: ShortOpportunityTrainingRow,
    contract: ShortOpportunityTrainingContract,
) -> int:
    if contract.probability_semantics == "TERMINAL_NET_POSITIVE_H12_AFTER_COSTS":
        return int(row.clean_opportunity)
    return int(
        row.clean_opportunity and row.mae_fraction <= contract.maximum_acceptable_mae
    )


def _matrix(rows: Sequence[ShortOpportunityTrainingRow]) -> np.ndarray:
    return np.asarray([row.features for row in rows], dtype=np.float64)


def _labels(
    rows: Sequence[ShortOpportunityTrainingRow],
    contract: ShortOpportunityTrainingContract,
) -> np.ndarray:
    return np.asarray([_label(row, contract) for row in rows], dtype=np.int64)


def _validate_temporal_blocks(
    train: Sequence[ShortOpportunityTrainingRow],
    calibration: Sequence[ShortOpportunityTrainingRow],
    scoring: Sequence[ShortOpportunityTrainingRow],
    embargo: timedelta,
) -> None:
    if min(len(train), len(calibration), len(scoring)) < 8:
        raise ValueError(
            "training, calibration, and scoring blocks require at least eight rows"
        )
    if not max(row.timestamp for row in train) + embargo <= min(
        row.timestamp for row in calibration
    ):
        raise ValueError("training and calibration blocks overlap or are unordered")
    if not max(row.timestamp for row in calibration) + embargo <= min(
        row.timestamp for row in scoring
    ):
        raise ValueError("calibration and scoring blocks overlap or are unordered")
    identities = [
        {(row.timestamp, row.symbol) for row in block}
        for block in (train, calibration, scoring)
    ]
    if (
        identities[0] & identities[1]
        or identities[0] & identities[2]
        or identities[1] & identities[2]
    ):
        raise ValueError("temporal training blocks share row identities")


def _fit_symbol_calibrators(
    rows: Sequence[ShortOpportunityTrainingRow],
    raw_probabilities: np.ndarray,
    labels: np.ndarray,
    contract: ShortOpportunityTrainingContract,
) -> tuple[dict[str, CalibratorSpec], dict[str, int]]:
    calibrators: dict[str, CalibratorSpec] = {}
    counts: dict[str, int] = {}
    for symbol in sorted({row.symbol for row in rows}):
        indices = [index for index, row in enumerate(rows) if row.symbol == symbol]
        symbol_labels = labels[indices]
        if (
            len(indices) < contract.minimum_symbol_calibration_rows
            or len(np.unique(symbol_labels)) < 2
        ):
            continue
        calibrators[symbol] = fit_platt_calibrator(
            raw_probabilities[indices], symbol_labels
        )
        counts[symbol] = len(indices)
    return calibrators, counts


def _scoring_metrics(
    rows: Sequence[ShortOpportunityTrainingRow],
    probabilities: np.ndarray,
    labels: np.ndarray,
) -> ShortOpportunityMetrics:
    ece, brier = calibration_metrics(probabilities, labels)
    per_symbol: dict[str, Mapping[str, float | int]] = {}
    for symbol in sorted({row.symbol for row in rows}):
        indices = [index for index, row in enumerate(rows) if row.symbol == symbol]
        symbol_labels = labels[indices]
        symbol_probabilities = probabilities[indices]
        symbol_ece, symbol_brier = calibration_metrics(
            symbol_probabilities, symbol_labels
        )
        per_symbol[symbol] = {
            "row_count": len(indices),
            "positive_rate": float(np.mean(symbol_labels)),
            "average_precision": (
                float(average_precision_score(symbol_labels, symbol_probabilities))
                if len(np.unique(symbol_labels)) == 2
                else float(np.mean(symbol_labels))
            ),
            "ece": symbol_ece,
            "brier": symbol_brier,
        }
    return ShortOpportunityMetrics(
        row_count=len(rows),
        positive_rate=float(np.mean(labels)),
        average_precision=float(average_precision_score(labels, probabilities)),
        ece=ece,
        brier=brier,
        per_symbol=per_symbol,
    )


def fit_short_opportunity_model(
    train: Sequence[ShortOpportunityTrainingRow],
    calibration: Sequence[ShortOpportunityTrainingRow],
    scoring: Sequence[ShortOpportunityTrainingRow],
    contract: ShortOpportunityTrainingContract,
    *,
    normalizer: FrozenNormalizer = FrozenNormalizer(),
) -> ShortOpportunityArtifact:
    """Fit and score without publishing or wiring the resulting artifact."""
    _validate_temporal_blocks(
        train,
        calibration,
        scoring,
        timedelta(minutes=contract.minimum_embargo_minutes),
    )
    train_labels = _labels(train, contract)
    calibration_labels = _labels(calibration, contract)
    scoring_labels = _labels(scoring, contract)
    if any(
        len(np.unique(values)) < 2
        for values in (train_labels, calibration_labels, scoring_labels)
    ):
        raise ValueError("every temporal block must contain both opportunity classes")

    parameters = dict(contract.model_parameters)
    if contract.model_candidate_id == "eqm_logistic_clean_baseline":
        model = LogisticRegression(**parameters, random_state=contract.seed)
    elif contract.model_candidate_id == "eqm_random_forest_clean":
        from sklearn.ensemble import RandomForestClassifier

        model = RandomForestClassifier(
            **parameters,
            random_state=contract.seed,
            n_jobs=1,
        )
    else:
        from sklearn.ensemble import HistGradientBoostingClassifier

        model = HistGradientBoostingClassifier(
            **parameters,
            random_state=contract.seed,
        )
    model.fit(_matrix(train), train_labels)
    calibration_raw = np.asarray(
        model.predict_proba(_matrix(calibration))[:, 1], dtype=np.float64
    )
    global_calibrator = fit_platt_calibrator(calibration_raw, calibration_labels)
    symbol_calibrators, symbol_counts = _fit_symbol_calibrators(
        calibration, calibration_raw, calibration_labels, contract
    )
    calibrator = HierarchicalProbabilityCalibrator(
        schema_version="aegis-hierarchical-symbol-calibration-v1",
        global_calibrator=global_calibrator,
        symbol_calibrators=symbol_calibrators,
        symbol_sample_counts=symbol_counts,
        shrinkage_sample_count=contract.symbol_calibration_shrinkage_rows,
    )
    scoring_raw = np.asarray(
        model.predict_proba(_matrix(scoring))[:, 1], dtype=np.float64
    )
    scoring_probabilities = np.asarray(
        [
            calibrator.apply(row.symbol, float(probability))
            for row, probability in zip(scoring, scoring_raw)
        ],
        dtype=np.float64,
    )
    tree_ensemble: TreeEnsemble | None = None
    coefficients: tuple[float, ...] = ()
    intercept = 0.0
    if contract.model_candidate_id == "eqm_logistic_clean_baseline":
        coefficients = tuple(float(value) for value in model.coef_[0])
        intercept = float(model.intercept_[0])
    elif contract.model_candidate_id == "eqm_random_forest_clean":
        from .competition import export_random_forest

        tree_ensemble = export_random_forest(
            model,
            contract.model_candidate_id,
            FEATURE_NAMES,
            classifier=True,
        )
    else:
        from .competition import export_hist_gradient_boosting

        tree_ensemble = export_hist_gradient_boosting(
            model,
            contract.model_candidate_id,
            FEATURE_NAMES,
            classifier=True,
        )
    return ShortOpportunityArtifact(
        schema_version="aegis-short-opportunity-artifact-v1",
        feature_schema_version=FEATURE_SCHEMA_VERSION,
        feature_hash=FEATURE_HASH,
        feature_names=FEATURE_NAMES,
        model_candidate_id=contract.model_candidate_id,
        coefficients=coefficients,
        intercept=intercept,
        tree_ensemble=tree_ensemble,
        normalizer=normalizer,
        calibrator=calibrator,
        scoring_metrics=_scoring_metrics(
            scoring, scoring_probabilities, scoring_labels
        ),
        probability_semantics=contract.probability_semantics,
    )


def rows_from_training_dataset(
    dataset: TrainingDataset,
) -> tuple[ShortOpportunityTrainingRow, ...]:
    """Adapt the existing causal dataset without changing feature or label order."""
    if (
        dataset.feature_schema_version != FEATURE_SCHEMA_VERSION
        or dataset.feature_hash != FEATURE_HASH
    ):
        raise ValueError("training dataset does not match aegis-features-v2 authority")
    return tuple(
        ShortOpportunityTrainingRow(
            timestamp=row.timestamp,
            symbol=row.symbol,
            features=row.features,
            clean_opportunity=row.target.clean_quality >= 0.5,
            mae_fraction=row.target.qmae,
        )
        for row in dataset.rows
        if row.target.label_valid
    )
