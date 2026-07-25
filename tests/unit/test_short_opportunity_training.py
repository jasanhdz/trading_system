from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from aegis.features import FEATURE_HASH, FEATURE_NAMES, FEATURE_SCHEMA_VERSION
from aegis.training.short_opportunity import (
    ShortOpportunityTrainingContract,
    ShortOpportunityTrainingRow,
    fit_short_opportunity_model,
    load_short_opportunity_artifact,
    write_short_opportunity_artifact,
)


def _contract(
    model_candidate_id: str = "eqm_logistic_clean_baseline",
) -> ShortOpportunityTrainingContract:
    parameters = (
        {
            "C": 1.0,
            "max_iter": 300,
            "solver": "liblinear",
        }
        if model_candidate_id == "eqm_logistic_clean_baseline"
        else {
            "n_estimators": 9,
            "max_depth": 4,
            "min_samples_leaf": 2,
            "class_weight": "balanced_subsample",
        }
    )
    return ShortOpportunityTrainingContract(
        schema_version="aegis-short-opportunity-training-contract-v1",
        feature_schema_version=FEATURE_SCHEMA_VERSION,
        feature_hash=FEATURE_HASH,
        maximum_acceptable_mae=0.02,
        model_candidate_id=model_candidate_id,
        model_parameters=parameters,
        seed=20260724,
        minimum_embargo_minutes=120,
        minimum_symbol_calibration_rows=8,
        symbol_calibration_shrinkage_rows=40,
    )


def _block(start: datetime, count: int = 40) -> tuple[ShortOpportunityTrainingRow, ...]:
    rows = []
    for index in range(count):
        positive = index % 2 == 0
        features = [0.0] * len(FEATURE_NAMES)
        features[0] = 2.0 if positive else -2.0
        features[1] = index / count
        rows.append(
            ShortOpportunityTrainingRow(
                timestamp=start + timedelta(minutes=index),
                symbol=("XRPUSDT", "SUIUSDT")[(index // 2) % 2],
                features=tuple(features),
                clean_opportunity=positive,
                mae_fraction=0.005 if positive else 0.03,
            )
        )
    return tuple(rows)


def test_short_opportunity_training_is_temporal_calibrated_and_83_feature_exact() -> None:
    start = datetime(2026, 1, 1, tzinfo=timezone.utc)
    artifact = fit_short_opportunity_model(
        _block(start),
        _block(start + timedelta(days=1)),
        _block(start + timedelta(days=2)),
        _contract(),
    )

    assert artifact.feature_names == FEATURE_NAMES
    assert len(artifact.coefficients) == 83
    assert artifact.scoring_metrics.average_precision > 0.95
    positive = [0.0] * len(FEATURE_NAMES)
    negative = [0.0] * len(FEATURE_NAMES)
    positive[0] = 2.0
    negative[0] = -2.0
    assert artifact.probability("XRPUSDT", positive) > artifact.probability("XRPUSDT", negative)


def test_short_opportunity_training_rejects_temporal_overlap() -> None:
    start = datetime(2026, 1, 1, tzinfo=timezone.utc)
    block = _block(start)
    with pytest.raises(ValueError, match="overlap or are unordered"):
        fit_short_opportunity_model(block, block, _block(start + timedelta(days=1)), _contract())


def test_short_opportunity_artifact_round_trip_is_prediction_exact(
    tmp_path: Path,
) -> None:
    start = datetime(2026, 1, 1, tzinfo=timezone.utc)
    artifact = fit_short_opportunity_model(
        _block(start),
        _block(start + timedelta(days=1)),
        _block(start + timedelta(days=2)),
        _contract(),
    )
    path = tmp_path / "short_opportunity.json"
    write_short_opportunity_artifact(path, artifact)
    restored = load_short_opportunity_artifact(path)
    values = [0.0] * len(FEATURE_NAMES)
    values[0] = 2.0

    assert restored.feature_names == artifact.feature_names
    assert restored.coefficients == artifact.coefficients
    assert restored.probability("XRPUSDT", values) == artifact.probability(
        "XRPUSDT", values
    )


def test_short_opportunity_random_forest_artifact_round_trip_is_prediction_exact(
    tmp_path: Path,
) -> None:
    start = datetime(2026, 1, 1, tzinfo=timezone.utc)
    artifact = fit_short_opportunity_model(
        _block(start),
        _block(start + timedelta(days=1)),
        _block(start + timedelta(days=2)),
        _contract("eqm_random_forest_clean"),
    )
    path = tmp_path / "short_opportunity_forest.json"
    write_short_opportunity_artifact(path, artifact)
    restored = load_short_opportunity_artifact(path)
    values = [0.0] * len(FEATURE_NAMES)
    values[0] = 2.0

    assert artifact.tree_ensemble is not None
    assert restored.model_candidate_id == "eqm_random_forest_clean"
    assert restored.probability("XRPUSDT", values) == artifact.probability(
        "XRPUSDT", values
    )


def test_research_path_is_not_imported_by_live_composition_roots() -> None:
    for path in (
        Path("src/aegis/runtime.py"),
        Path("src/aegis/live_decision.py"),
        Path("src/aegis/layers.py"),
    ):
        source = path.read_text(encoding="utf-8")
        assert "aegis.research" not in source
        assert "short_opportunity" not in source
