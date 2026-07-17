from dataclasses import replace
from datetime import timedelta
from pathlib import Path

import pytest

from aegis.config import load_brain_config
from aegis.domain import ScientificContext, ValidationStatus
from aegis.features import DeterministicFeaturePipeline, FEATURE_HASH, FEATURE_NAMES, MarketSnapshotValidator, SnapshotValidationError
from aegis.layers import LayerSettings, OrderedScientificLayers
from aegis.models import DeterministicModelRuntime, ModelBundleError, load_model_bundle


def _components():
    config = load_brain_config(Path(__file__).parents[2] / "config")
    bundle = load_model_bundle(config.models.artifact_registry / f"{config.models.model_bundle_id}.json")
    features = DeterministicFeaturePipeline(bundle.normalizer)
    models = DeterministicModelRuntime(bundle, config.models.direction_threshold)
    layers = OrderedScientificLayers(LayerSettings(
        config.models.trrm_max_tail_probability, config.models.qmae_max_fraction,
        config.models.eqm_min_score, config.models.estimated_round_trip_cost_fraction,
        config.models.direction_threshold,
    ))
    return config, features, models, layers


def test_snapshot_validation_features_and_models_are_finite_and_deterministic(snapshot_factory) -> None:
    config, pipeline, models, _ = _components()
    snapshot = snapshot_factory()
    MarketSnapshotValidator(config.universe).validate(snapshot, snapshot.closed_at + timedelta(minutes=1))
    first = pipeline.transform(snapshot)
    second = pipeline.transform(snapshot)
    assert first == second
    assert first.feature_names == FEATURE_NAMES and first.feature_hash == FEATURE_HASH
    assert len(first.rows) == 11 and all(len(row.raw_values) == 39 for row in first.rows)
    predictions = models.predict(first)
    assert len(predictions.predictions) == 22


def test_snapshot_validation_rejects_stale_gaps_and_partial_candles(snapshot_factory) -> None:
    config, _, _, _ = _components()
    snapshot = snapshot_factory()
    validator = MarketSnapshotValidator(config.universe)
    with pytest.raises(SnapshotValidationError) as stale:
        validator.validate(snapshot, snapshot.closed_at + timedelta(hours=1))
    assert stale.value.status is ValidationStatus.NO_TRADE_DATA_STALE
    candle = replace(snapshot.series[0].candles[-1], is_closed=False)
    series = replace(snapshot.series[0], candles=(*snapshot.series[0].candles[:-1], candle))
    with pytest.raises(SnapshotValidationError):
        validator.validate(replace(snapshot, series=(series, *snapshot.series[1:])), snapshot.closed_at)


def test_layers_emit_all_six_semantics_and_fail_closed(snapshot_factory) -> None:
    _, pipeline, models, layers = _components()
    snapshot = snapshot_factory()
    features = pipeline.transform(snapshot)
    predictions = models.predict(features)
    outputs = layers.apply(predictions, ScientificContext("r", "c", snapshot.closed_at, "5m", snapshot.portfolio, features))
    assert len(outputs.results) == 11
    assert tuple(layer.value for layer in outputs.ordered_layers) == ("D3", "RV2", "TRRM", "QMAE", "EQM", "ECON1")
    assert all(0 <= result.trrm_compatibility <= 1 for result in outputs.results)
    with pytest.raises(ModelBundleError):
        models.predict(replace(features, feature_hash="tampered"))
