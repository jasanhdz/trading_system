from dataclasses import replace

import numpy as np

from aegis.decision import ScientificCandidateBuilder
from aegis.domain import PortfolioContext, ReasonCode, ScientificContext
from aegis.features import DeterministicFeaturePipeline
from aegis.layers import LayerSettings, OrderedScientificLayers
from aegis.models import DeterministicModelRuntime, QuantileHeadSpec
from aegis.training.train import calibration_metrics, fit_platt_calibrator
from aegis.utils import Sha256HashProvider


def _layers(snapshot, features, predictions):
    return OrderedScientificLayers(LayerSettings(0.70, 0.03, 0.0, 0.50)).apply(
        predictions,
        ScientificContext("request", "cycle", snapshot.closed_at, "5m", snapshot.portfolio, features),
    )


def test_missing_calibration_and_quantiles_fail_closed_without_stop_proxy(
    snapshot_factory, scenario_bundle_factory,
) -> None:
    snapshot = snapshot_factory()
    features = DeterministicFeaturePipeline().transform(snapshot)
    source = scenario_bundle_factory("SHORT")
    estimators = tuple(replace(item, qmae_quantiles=None) for item in source.estimators)
    bundle = replace(source, calibration=None, estimators=estimators)
    predictions = DeterministicModelRuntime(bundle).predict(features)
    outputs = _layers(snapshot, features, predictions)
    assert all(not result.eligible for result in outputs.results)
    assert all(ReasonCode.CALIBRATION_MISSING in result.reason_codes for result in outputs.results)
    assert all(ReasonCode.QMAE_QUANTILE_UNAVAILABLE in result.reason_codes for result in outputs.results)
    assert all(result.qmae_q90 is None for result in outputs.results)
    candidates = ScientificCandidateBuilder(Sha256HashProvider()).build("cycle", predictions, outputs)
    assert all(candidate.risk_intent.stop_distance_fraction is None for candidate in candidates.candidates)


def test_qmae_mean_is_not_exposed_as_q90_when_coverage_is_invalid(
    snapshot_factory, scenario_bundle_factory,
) -> None:
    snapshot = snapshot_factory()
    features = DeterministicFeaturePipeline().transform(snapshot)
    source = scenario_bundle_factory("SHORT")
    invalid = tuple(
        replace(
            item,
            qmae_mean=replace(item.qmae_mean, bias=0.025),
            qmae_quantiles=QuantileHeadSpec(
                item.qmae_quantiles.q50, item.qmae_quantiles.q90, 0.0, 0.80,  # type: ignore[union-attr]
            ),
        )
        for item in source.estimators
    )
    predictions = DeterministicModelRuntime(replace(source, estimators=invalid)).predict(features)
    assert all(item.qmae_mean == 0.025 for item in predictions.predictions)
    assert all(item.qmae_q90 is None and not item.qmae_valid for item in predictions.predictions)


def test_platt_calibrator_serializable_contract_matches_training_application() -> None:
    raw = np.asarray([0.05, 0.10, 0.20, 0.35, 0.45, 0.55, 0.70, 0.80, 0.90, 0.95])
    labels = np.asarray([0, 0, 0, 0, 1, 0, 1, 1, 1, 1], dtype=np.float64)
    calibrator = fit_platt_calibrator(raw, labels)
    runtime_values = np.asarray([calibrator.apply(float(value)) for value in raw])
    ece, brier = calibration_metrics(runtime_values, labels)
    assert abs(calibrator.ece - ece) <= 1e-15
    assert abs(calibrator.brier - brier) <= 1e-15
    assert calibrator.sample_count == len(raw)
    assert np.isfinite(runtime_values).all()
