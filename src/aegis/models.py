"""Verified immutable model bundles and deterministic inference runtime."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Protocol

from .domain import FeatureBatch, ModelPrediction, ModelPredictions, TradeSide
from .features import FEATURE_NAMES, FrozenNormalizer
from .utils import Sha256HashProvider


class ModelBundleError(ValueError):
    pass


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
class EstimatorSpec:
    model_id: str
    horizon_bars: int
    long: LinearHead
    short: LinearHead
    neutral: LinearHead
    expected_return: LinearHead
    tail_risk: LinearHead
    qmae_q90: LinearHead
    quality: LinearHead


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


def _head(payload: Mapping[str, Any]) -> LinearHead:
    weights = payload.get("weights", {})
    if not isinstance(weights, Mapping):
        raise ModelBundleError("head weights must be a mapping")
    parsed = {str(name): float(value) for name, value in weights.items()}
    if not all(math.isfinite(value) for value in parsed.values()):
        raise ModelBundleError("head weights must be finite")
    unknown = set(parsed) - set(FEATURE_NAMES)
    if unknown:
        raise ModelBundleError(f"head contains unknown features: {sorted(unknown)}")
    return LinearHead(float(payload.get("bias", 0.0)), parsed)


def _window(value: Any, name: str) -> tuple[str, str] | None:
    if value is None:
        return None
    if not isinstance(value, list) or len(value) != 2 or not all(isinstance(item, str) and item for item in value):
        raise ModelBundleError(f"{name} must be null or a two-item timestamp list")
    return value[0], value[1]


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
    if payload.get("schema_version") != "aegis-model-bundle-v1":
        raise ModelBundleError("unsupported model bundle schema")
    if payload.get("feature_hash") is None or payload.get("feature_schema_version") is None:
        raise ModelBundleError("bundle feature contract is incomplete")
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
                long=_head(item["heads"]["long"]), short=_head(item["heads"]["short"]),
                neutral=_head(item["heads"]["neutral"]), expected_return=_head(item["heads"]["expected_return"]),
                tail_risk=_head(item["heads"]["tail_risk"]), qmae_q90=_head(item["heads"]["qmae_q90"]),
                quality=_head(item["heads"]["quality"]),
            )
            for item in payload.get("estimators", ())
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ModelBundleError("estimator contract is incomplete or invalid") from exc
    if not estimators or len({item.model_id for item in estimators}) != len(estimators):
        raise ModelBundleError("bundle requires uniquely named estimators")
    if any(item.horizon_bars <= 0 for item in estimators):
        raise ModelBundleError("estimator horizon must be positive")
    means = {str(k): float(v) for k, v in normalizer_data.get("means", {}).items()}
    scales = {str(k): float(v) for k, v in normalizer_data.get("scales", {}).items()}
    if set(means) != set(scales) or set(means) - set(FEATURE_NAMES):
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
    if metadata.feature_count != len(FEATURE_NAMES):
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
        estimators=estimators, metadata=metadata,
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
        for row in features.rows:
            values = dict(zip(features.feature_names, row.normalized_values))
            for estimator in self.bundle.estimators:
                probabilities = _softmax((estimator.long.evaluate(values), estimator.short.evaluate(values), estimator.neutral.evaluate(values)))
                long_probability, short_probability, neutral_probability = probabilities
                best = max(probabilities)
                if best < self.direction_threshold or neutral_probability == best:
                    side = TradeSide.NO_TRADE
                else:
                    side = TradeSide.LONG if long_probability >= short_probability else TradeSide.SHORT
                predictions.append(ModelPrediction(
                    model_id=estimator.model_id, symbol=row.symbol, horizon_bars=estimator.horizon_bars, side=side,
                    long_probability=long_probability, short_probability=short_probability, neutral_probability=neutral_probability,
                    expected_return=estimator.expected_return.evaluate(values), tail_risk_probability=_sigmoid(estimator.tail_risk.evaluate(values)),
                    qmae_q90=max(0.0, estimator.qmae_q90.evaluate(values)), quality_probability=_sigmoid(estimator.quality.evaluate(values)),
                    uncertainty=1.0 - best,
                ))
        return ModelPredictions(self.bundle.bundle_id, features.feature_hash, tuple(predictions))
