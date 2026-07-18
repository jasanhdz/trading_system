"""Verified immutable model bundles and deterministic inference runtime."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Mapping, Protocol

from .domain import FeatureBatch, ModelPrediction, ModelPredictions, TradeSide
from .features import FrozenNormalizer, feature_contract
from .tree_models import TreeEnsemble, TreeModelError
from .utils import Sha256HashProvider


class ModelBundleError(ValueError):
    pass


class CalibrationMethod(str, Enum):
    IDENTITY = "IDENTITY"
    PLATT = "PLATT"
    ISOTONIC = "ISOTONIC"


class ModelRuntime(Protocol):
    bundle_id: str

    def predict(self, features: FeatureBatch) -> ModelPredictions: ...


@dataclass(frozen=True)
class LinearHead:
    bias: float
    weights: Mapping[str, float]

    def evaluate(self, values: Mapping[str, float]) -> float:
        result = self.bias + math.fsum(float(weight) * values.get(name, 0.0) for name, weight in self.weights.items())
        if not math.isfinite(result):
            raise ModelBundleError("non-finite linear head output")
        return result


@dataclass(frozen=True)
class TreeHead:
    ensemble_id: str
    output_kind: str

    def __post_init__(self) -> None:
        if not self.ensemble_id or self.output_kind not in {"RAW", "PROBABILITY"}:
            raise ModelBundleError("tree head contract is invalid")


PredictiveHead = LinearHead | TreeHead


@dataclass(frozen=True)
class CalibratorSpec:
    method: CalibrationMethod
    ece: float
    brier: float
    sample_count: int
    parameters: tuple[float, ...] = ()
    x: tuple[float, ...] = ()
    y: tuple[float, ...] = ()

    def __post_init__(self) -> None:
        if not 0.0 <= self.ece <= 1.0 or not 0.0 <= self.brier <= 1.0 or self.sample_count <= 0:
            raise ModelBundleError("calibrator metrics are invalid")
        if self.method is CalibrationMethod.PLATT and len(self.parameters) != 2:
            raise ModelBundleError("Platt calibrator requires slope and intercept")
        if self.method is CalibrationMethod.ISOTONIC:
            if len(self.x) < 2 or len(self.x) != len(self.y) or tuple(sorted(self.x)) != self.x:
                raise ModelBundleError("isotonic calibration knots are invalid")
            if any(not 0.0 <= value <= 1.0 for value in (*self.x, *self.y)):
                raise ModelBundleError("isotonic calibration knots must be probabilities")

    def apply(self, probability: float) -> float:
        value = min(1.0, max(0.0, float(probability)))
        if self.method is CalibrationMethod.IDENTITY:
            return value
        if self.method is CalibrationMethod.PLATT:
            slope, intercept = self.parameters
            clipped = min(1.0 - 1e-15, max(1e-15, value))
            logit = math.log(clipped / (1.0 - clipped))
            return _sigmoid(slope * logit + intercept)
        if value <= self.x[0]:
            return self.y[0]
        if value >= self.x[-1]:
            return self.y[-1]
        for index in range(1, len(self.x)):
            if value <= self.x[index]:
                left_x, right_x = self.x[index - 1], self.x[index]
                weight = 0.0 if right_x == left_x else (value - left_x) / (right_x - left_x)
                return self.y[index - 1] + weight * (self.y[index] - self.y[index - 1])
        raise AssertionError("unreachable isotonic interval")


@dataclass(frozen=True)
class CalibrationBlock:
    schema_version: str
    out_of_fold: bool
    long: CalibratorSpec
    short: CalibratorSpec
    neutral: CalibratorSpec
    tail_risk: CalibratorSpec
    quality: CalibratorSpec

    @property
    def valid(self) -> bool:
        return self.schema_version == "aegis-calibration-v1" and self.out_of_fold


@dataclass(frozen=True)
class QuantileHeadSpec:
    q50: PredictiveHead
    q90: PredictiveHead
    conformal_adjustment: float
    empirical_coverage: float
    coverage_min: float = 0.87
    coverage_max: float = 0.93

    @property
    def valid(self) -> bool:
        return (
            self.conformal_adjustment >= 0.0
            and 0.0 <= self.coverage_min <= self.empirical_coverage <= self.coverage_max <= 1.0
        )


@dataclass(frozen=True)
class EstimatorSpec:
    model_id: str
    horizon_bars: int
    long: PredictiveHead
    short: PredictiveHead
    neutral: PredictiveHead
    expected_return: PredictiveHead
    tail_risk: PredictiveHead
    qmae_mean: PredictiveHead
    quality: PredictiveHead
    qmae_quantiles: QuantileHeadSpec | None = None


@dataclass(frozen=True)
class BundleMetadata:
    purpose: str
    trained: bool
    training_window: tuple[str, str] | None
    validation_window: tuple[str, str] | None
    test_window: tuple[str, str] | None
    seed: int
    framework: str
    framework_version: str
    code_version: str
    calibration_method: str
    feature_count: int
    thresholds: Mapping[str, float]


@dataclass(frozen=True)
class ModelBundle:
    bundle_id: str
    schema_version: str
    feature_schema_version: str
    feature_hash: str
    universe_id: str
    symbol_set_hash: str
    timeframe: str
    approved: bool
    content_hash: str
    normalizer: FrozenNormalizer
    estimators: tuple[EstimatorSpec, ...]
    metadata: BundleMetadata
    calibration: CalibrationBlock | None = None
    tree_ensembles: tuple[TreeEnsemble, ...] = ()


def _head(payload: Mapping[str, Any], allowed_names: tuple[str, ...]) -> PredictiveHead:
    if "tree_ensemble_id" in payload:
        return TreeHead(str(payload["tree_ensemble_id"]), str(payload.get("output_kind", "RAW")))
    weights = payload.get("weights", {})
    if not isinstance(weights, Mapping):
        raise ModelBundleError("head weights must be a mapping")
    parsed = {str(name): float(value) for name, value in weights.items()}
    if not all(math.isfinite(value) for value in parsed.values()):
        raise ModelBundleError("head weights must be finite")
    unknown = set(parsed) - set(allowed_names)
    if unknown:
        raise ModelBundleError(f"head contains unknown features: {sorted(unknown)}")
    return LinearHead(float(payload.get("bias", 0.0)), parsed)


def _window(value: Any, name: str) -> tuple[str, str] | None:
    if value is None:
        return None
    if not isinstance(value, list) or len(value) != 2 or not all(isinstance(item, str) and item for item in value):
        raise ModelBundleError(f"{name} must be null or a two-item timestamp list")
    return value[0], value[1]


def _calibrator(payload: Mapping[str, Any]) -> CalibratorSpec:
    try:
        return CalibratorSpec(
            method=CalibrationMethod(str(payload["method"])),
            ece=float(payload["ece"]), brier=float(payload["brier"]),
            sample_count=int(payload["sample_count"]),
            parameters=tuple(float(value) for value in payload.get("parameters", ())),
            x=tuple(float(value) for value in payload.get("x", ())),
            y=tuple(float(value) for value in payload.get("y", ())),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ModelBundleError("calibrator contract is invalid") from exc


def _calibration(payload: Any) -> CalibrationBlock | None:
    if payload is None:
        return None
    if not isinstance(payload, Mapping) or not isinstance(payload.get("heads"), Mapping):
        raise ModelBundleError("calibration block is invalid")
    heads = payload["heads"]
    try:
        return CalibrationBlock(
            schema_version=str(payload["schema_version"]), out_of_fold=bool(payload["out_of_fold"]),
            long=_calibrator(heads["long"]), short=_calibrator(heads["short"]),
            neutral=_calibrator(heads["neutral"]), tail_risk=_calibrator(heads["tail_risk"]),
            quality=_calibrator(heads["quality"]),
        )
    except KeyError as exc:
        raise ModelBundleError("calibration block does not cover every probabilistic head") from exc


def _quantiles(payload: Any, allowed_names: tuple[str, ...]) -> QuantileHeadSpec | None:
    if payload is None:
        return None
    if not isinstance(payload, Mapping):
        raise ModelBundleError("QMAE quantile block is invalid")
    try:
        spec = QuantileHeadSpec(
            q50=_head(payload["q50"], allowed_names), q90=_head(payload["q90"], allowed_names),
            conformal_adjustment=float(payload["conformal_adjustment"]),
            empirical_coverage=float(payload["empirical_coverage"]),
            coverage_min=float(payload.get("coverage_min", 0.87)),
            coverage_max=float(payload.get("coverage_max", 0.93)),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ModelBundleError("QMAE quantile contract is invalid") from exc
    if not all(math.isfinite(value) for value in (
        spec.conformal_adjustment, spec.empirical_coverage, spec.coverage_min, spec.coverage_max,
    )):
        raise ModelBundleError("QMAE quantile metadata must be finite")
    return spec


def model_bundle_from_payload(payload: Mapping[str, Any], *, expected_bundle_id: str | None = None) -> ModelBundle:
    """Validate an in-memory JSON bundle using the same path as registry loading."""
    if not isinstance(payload, dict):
        raise ModelBundleError("bundle must be a JSON object")
    claimed_hash = str(payload.get("content_hash", ""))
    hash_payload = dict(payload)
    hash_payload.pop("content_hash", None)
    actual_hash = Sha256HashProvider().digest_value(hash_payload)
    if claimed_hash != actual_hash:
        raise ModelBundleError("bundle content hash mismatch")
    bundle_id = str(payload.get("bundle_id", ""))
    if expected_bundle_id is not None and bundle_id != expected_bundle_id:
        raise ModelBundleError("bundle ID mismatch")
    if payload.get("schema_version") not in {"aegis-model-bundle-v1", "aegis-model-bundle-v2"}:
        raise ModelBundleError("unsupported model bundle schema")
    if payload.get("feature_hash") is None or payload.get("feature_schema_version") is None:
        raise ModelBundleError("bundle feature contract is incomplete")
    try:
        feature_names, expected_feature_hash = feature_contract(str(payload["feature_schema_version"]))
    except ValueError as exc:
        raise ModelBundleError("unsupported bundle feature schema") from exc
    if str(payload["feature_hash"]) != expected_feature_hash:
        raise ModelBundleError("bundle feature hash does not match its schema")
    try:
        tree_ensembles = tuple(TreeEnsemble.from_payload(item) for item in payload.get("tree_ensembles", ()))
    except (TypeError, TreeModelError) as exc:
        raise ModelBundleError("tree ensemble block is invalid") from exc
    if len({item.ensemble_id for item in tree_ensembles}) != len(tree_ensembles):
        raise ModelBundleError("tree ensemble IDs must be unique")
    normalizer_data = payload.get("normalizer", {})
    if not isinstance(normalizer_data, Mapping):
        raise ModelBundleError("normalizer must be a mapping")
    metadata_data = payload.get("metadata")
    if not isinstance(metadata_data, Mapping):
        raise ModelBundleError("bundle metadata is required")
    try:
        estimators = tuple(
            EstimatorSpec(
                model_id=str(item["model_id"]), horizon_bars=int(item["horizon_bars"]),
                long=_head(item["heads"]["long"], feature_names), short=_head(item["heads"]["short"], feature_names),
                neutral=_head(item["heads"]["neutral"], feature_names), expected_return=_head(item["heads"]["expected_return"], feature_names),
                tail_risk=_head(item["heads"]["tail_risk"], feature_names), qmae_mean=_head(item["heads"]["qmae_mean"], feature_names),
                quality=_head(item["heads"]["quality"], feature_names),
                qmae_quantiles=_quantiles(item.get("qmae_quantiles"), feature_names),
            )
            for item in payload.get("estimators", ())
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ModelBundleError("estimator contract is incomplete or invalid") from exc
    if not estimators or len({item.model_id for item in estimators}) != len(estimators):
        raise ModelBundleError("bundle requires uniquely named estimators")
    if any(item.horizon_bars <= 0 for item in estimators):
        raise ModelBundleError("estimator horizon must be positive")
    tree_ids = {item.ensemble_id for item in tree_ensembles}
    referenced_tree_ids = {
        head.ensemble_id
        for estimator in estimators
        for head in (
            estimator.long, estimator.short, estimator.neutral, estimator.expected_return,
            estimator.tail_risk, estimator.qmae_mean, estimator.quality,
            *((estimator.qmae_quantiles.q50, estimator.qmae_quantiles.q90) if estimator.qmae_quantiles else ()),
        )
        if isinstance(head, TreeHead)
    }
    if not referenced_tree_ids <= tree_ids:
        raise ModelBundleError("estimator references an unknown tree ensemble")
    means = {str(k): float(v) for k, v in normalizer_data.get("means", {}).items()}
    scales = {str(k): float(v) for k, v in normalizer_data.get("scales", {}).items()}
    if set(means) != set(scales) or set(means) - set(feature_names):
        raise ModelBundleError("normalizer means/scales are incompatible")
    if not all(math.isfinite(value) for value in (*means.values(), *scales.values())) or any(value <= 0 for value in scales.values()):
        raise ModelBundleError("normalizer values are invalid")
    thresholds_data = metadata_data.get("thresholds", {})
    if not isinstance(thresholds_data, Mapping):
        raise ModelBundleError("metadata thresholds must be a mapping")
    metadata = BundleMetadata(
        purpose=str(metadata_data["purpose"]), trained=bool(metadata_data["trained"]),
        training_window=_window(metadata_data.get("training_window"), "training_window"),
        validation_window=_window(metadata_data.get("validation_window"), "validation_window"),
        test_window=_window(metadata_data.get("test_window"), "test_window"),
        seed=int(metadata_data["seed"]), framework=str(metadata_data["framework"]),
        framework_version=str(metadata_data["framework_version"]), code_version=str(metadata_data["code_version"]),
        calibration_method=str(metadata_data["calibration_method"]), feature_count=int(metadata_data["feature_count"]),
        thresholds={str(key): float(value) for key, value in thresholds_data.items()},
    )
    if metadata.feature_count != len(feature_names):
        raise ModelBundleError("bundle feature count mismatch")
    return ModelBundle(
        bundle_id=bundle_id, schema_version=str(payload["schema_version"]),
        feature_schema_version=str(payload["feature_schema_version"]), feature_hash=str(payload["feature_hash"]),
        universe_id=str(payload["universe_id"]), symbol_set_hash=str(payload["symbol_set_hash"]), timeframe=str(payload["timeframe"]),
        approved=bool(payload.get("approved", False)), content_hash=claimed_hash,
        normalizer=FrozenNormalizer(
            means=means,
            scales=scales,
            clip_absolute=float(normalizer_data.get("clip_absolute", 12.0)),
        ),
        estimators=estimators, metadata=metadata, calibration=_calibration(payload.get("calibration")),
        tree_ensembles=tree_ensembles,
    )


def load_model_bundle(path: Path, *, expected_bundle_id: str | None = None) -> ModelBundle:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return model_bundle_from_payload(payload, expected_bundle_id=expected_bundle_id)


def _sigmoid(value: float) -> float:
    if value >= 0:
        z = math.exp(-min(value, 700.0))
        return 1.0 / (1.0 + z)
    z = math.exp(max(value, -700.0))
    return z / (1.0 + z)


def _softmax(values: tuple[float, ...]) -> tuple[float, ...]:
    maximum = max(values)
    exponentials = tuple(math.exp(max(-700.0, value - maximum)) for value in values)
    denominator = math.fsum(exponentials)
    return tuple(value / denominator for value in exponentials)


@dataclass(frozen=True)
class DeterministicModelRuntime:
    bundle: ModelBundle
    direction_threshold: float = 0.50

    @property
    def bundle_id(self) -> str:
        return self.bundle.bundle_id

    def predict(self, features: FeatureBatch) -> ModelPredictions:
        if features.schema_version != self.bundle.feature_schema_version or features.feature_hash != self.bundle.feature_hash:
            raise ModelBundleError("feature schema does not match model bundle")
        predictions: list[ModelPrediction] = []
        tree_ensembles = {item.ensemble_id: item for item in self.bundle.tree_ensembles}

        def evaluate(head: PredictiveHead, values: Mapping[str, float], ordered: tuple[float, ...]) -> float:
            if isinstance(head, LinearHead):
                return head.evaluate(values)
            return tree_ensembles[head.ensemble_id].evaluate(ordered)

        for row in features.rows:
            values = dict(zip(features.feature_names, row.normalized_values))
            for estimator in self.bundle.estimators:
                ordered = row.normalized_values
                probabilities = _softmax((
                    evaluate(estimator.long, values, ordered), evaluate(estimator.short, values, ordered),
                    evaluate(estimator.neutral, values, ordered),
                ))
                calibration_valid = self.bundle.calibration is not None and self.bundle.calibration.valid
                if calibration_valid:
                    assert self.bundle.calibration is not None
                    calibrated = (
                        self.bundle.calibration.long.apply(probabilities[0]),
                        self.bundle.calibration.short.apply(probabilities[1]),
                        self.bundle.calibration.neutral.apply(probabilities[2]),
                    )
                    total = math.fsum(calibrated)
                    if total <= 0.0:
                        raise ModelBundleError("calibrated direction probabilities are degenerate")
                    probabilities = tuple(value / total for value in calibrated)
                long_probability, short_probability, neutral_probability = probabilities
                best = max(probabilities)
                if best < self.direction_threshold or neutral_probability == best:
                    side = TradeSide.NO_TRADE
                else:
                    side = TradeSide.LONG if long_probability >= short_probability else TradeSide.SHORT
                tail_value = evaluate(estimator.tail_risk, values, ordered)
                quality_value = evaluate(estimator.quality, values, ordered)
                raw_tail = tail_value if isinstance(estimator.tail_risk, TreeHead) and estimator.tail_risk.output_kind == "PROBABILITY" else _sigmoid(tail_value)
                raw_quality = quality_value if isinstance(estimator.quality, TreeHead) and estimator.quality.output_kind == "PROBABILITY" else _sigmoid(quality_value)
                tail_probability = self.bundle.calibration.tail_risk.apply(raw_tail) if calibration_valid else raw_tail  # type: ignore[union-attr]
                quality_probability = self.bundle.calibration.quality.apply(raw_quality) if calibration_valid else raw_quality  # type: ignore[union-attr]
                qmae_mean = max(0.0, evaluate(estimator.qmae_mean, values, ordered))
                qmae_q50 = qmae_q90 = coverage = None
                qmae_valid = estimator.qmae_quantiles is not None and estimator.qmae_quantiles.valid
                if qmae_valid:
                    assert estimator.qmae_quantiles is not None
                    qmae_q50 = max(0.0, evaluate(estimator.qmae_quantiles.q50, values, ordered))
                    qmae_q90 = max(qmae_q50, evaluate(estimator.qmae_quantiles.q90, values, ordered))
                    qmae_q90 += estimator.qmae_quantiles.conformal_adjustment
                    coverage = estimator.qmae_quantiles.empirical_coverage
                predictions.append(ModelPrediction(
                    model_id=estimator.model_id, symbol=row.symbol, horizon_bars=estimator.horizon_bars, side=side,
                    long_probability=long_probability, short_probability=short_probability, neutral_probability=neutral_probability,
                    expected_return=evaluate(estimator.expected_return, values, ordered), tail_risk_probability=tail_probability,
                    qmae_mean=qmae_mean, quality_probability=quality_probability, uncertainty=1.0 - best,
                    qmae_q50=qmae_q50, qmae_q90=qmae_q90, qmae_coverage=coverage,
                    calibration_valid=calibration_valid, qmae_valid=qmae_valid,
                ))
        return ModelPredictions(self.bundle.bundle_id, features.feature_hash, tuple(predictions))
