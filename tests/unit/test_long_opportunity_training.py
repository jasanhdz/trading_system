from datetime import datetime, timedelta, timezone
from pathlib import Path

from aegis.features import FEATURE_HASH, FEATURE_NAMES, FEATURE_SCHEMA_VERSION
from aegis.training.long_opportunity import (
    LongOpportunityTrainingContract,
    LongOpportunityTrainingRow,
    fit_long_opportunity_model,
    load_long_opportunity_artifact,
    write_long_opportunity_artifact,
)


def _contract() -> LongOpportunityTrainingContract:
    return LongOpportunityTrainingContract(
        schema_version="aegis-long-opportunity-training-contract-v1",
        feature_schema_version=FEATURE_SCHEMA_VERSION,
        feature_hash=FEATURE_HASH,
        maximum_acceptable_mae=0.02,
        model_candidate_id="eqm_logistic_clean_baseline",
        model_parameters={"C": 1.0, "max_iter": 300, "solver": "liblinear"},
        seed=20260725,
        minimum_embargo_minutes=120,
        minimum_symbol_calibration_rows=8,
        symbol_calibration_shrinkage_rows=40,
    )


def _block(start: datetime) -> tuple[LongOpportunityTrainingRow, ...]:
    rows = []
    for index in range(40):
        positive = index % 2 == 0
        features = [0.0] * len(FEATURE_NAMES)
        features[0] = 2.0 if positive else -2.0
        rows.append(
            LongOpportunityTrainingRow(
                timestamp=start + timedelta(minutes=index),
                symbol=("BTCUSDT", "ETHUSDT")[(index // 2) % 2],
                features=tuple(features),
                clean_opportunity=positive,
                mae_fraction=0.005 if positive else 0.03,
            )
        )
    return tuple(rows)


def test_long_model_is_real_directional_artifact_and_round_trips(tmp_path: Path) -> None:
    start = datetime(2026, 1, 1, tzinfo=timezone.utc)
    artifact = fit_long_opportunity_model(
        _block(start),
        _block(start + timedelta(days=1)),
        _block(start + timedelta(days=2)),
        _contract(),
    )
    positive = [0.0] * len(FEATURE_NAMES)
    negative = [0.0] * len(FEATURE_NAMES)
    positive[0] = 2.0
    negative[0] = -2.0
    assert artifact.direction == "LONG"
    assert artifact.probability("BTCUSDT", positive) > artifact.probability(
        "BTCUSDT", negative
    )

    path = tmp_path / "long_opportunity.json"
    write_long_opportunity_artifact(path, artifact)
    restored = load_long_opportunity_artifact(path)
    assert restored.direction == "LONG"
    assert restored.probability("BTCUSDT", positive) == artifact.probability(
        "BTCUSDT", positive
    )
