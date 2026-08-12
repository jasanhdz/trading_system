"""Canonical causal feature contract for the inactive V17 challenger."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from .domain import Candle
from .features import FEATURE_NAMES, DeterministicFeaturePipeline
from .research.decomposed_entry_v9 import (
    V9_FEATURE_NAMES,
    rolling_four_hour_context,
    v9_feature_vectors,
)
from .research.directional_contract_v15 import contract_indices
from .research.long_entry_v21_shadow import multitimeframe_long_features
from .research.regime_aware_directional_v6 import (
    classify_regime_axes,
    directional_role,
    regime_aware_feature_vector,
)
from .research.regime_entry_exit_v7 import v7_feature_vector
from .research.tail_aware_entry_v8 import v8_feature_vector
from .training.hybrid_directional import DirectionalSide
from .utils import Sha256HashProvider


class V17FeatureContractError(ValueError):
    pass


V17_FEATURE_SCHEMA = "aegis-v17-v9-directional-features-v1"
V17_REQUIRED_CLOSED_5M_BARS = 576
V17_DTYPE = "float64"
V17_NORMALIZATION_CONTRACT = {
    "clean_classifier": "FITTED_STANDARD_SCALER",
    "danger_classifier": "FITTED_STANDARD_SCALER",
    "mae_q90": "NONE_HISTOGRAM_GRADIENT_BOOSTING_NATIVE",
    "pairwise_ranker": "FITTED_STANDARD_SCALER",
}


@dataclass(frozen=True)
class V17FeatureVector:
    side: str
    schema_version: str
    schema_hash: str
    names: tuple[str, ...]
    values: tuple[float, ...]
    dtype: str = V17_DTYPE

    def validate(self) -> None:
        expected = contract_for_side(self.side)
        if self.schema_version != V17_FEATURE_SCHEMA:
            raise V17FeatureContractError("V17_FEATURE_VERSION_MISMATCH")
        if self.schema_hash != expected["schema_hash"]:
            raise V17FeatureContractError("V17_FEATURE_HASH_MISMATCH")
        if self.names != expected["feature_names"]:
            raise V17FeatureContractError("V17_FEATURE_ORDER_MISMATCH")
        if self.dtype != V17_DTYPE:
            raise V17FeatureContractError("V17_FEATURE_DTYPE_MISMATCH")
        if len(self.values) != len(self.names):
            raise V17FeatureContractError("V17_FEATURE_WIDTH_MISMATCH")
        if not all(isinstance(value, float) and math.isfinite(value) for value in self.values):
            raise V17FeatureContractError("V17_FEATURE_VALUE_INVALID")


def _side(side: str) -> DirectionalSide:
    try:
        return DirectionalSide(side)
    except ValueError as exc:
        raise V17FeatureContractError("V17_FEATURE_SIDE_INVALID") from exc


def contract_for_side(side: str) -> Mapping[str, Any]:
    normalized = _side(side).value
    from pathlib import Path

    import yaml

    root = Path(__file__).resolve().parents[2]
    config = yaml.safe_load(
        (root / "config/experiments/aegis_directional_contract_v15_research.yaml").read_text()
    )
    indices = contract_indices(config, normalized)
    names = tuple(V9_FEATURE_NAMES[index] for index in indices)
    material = {
        "schema_version": V17_FEATURE_SCHEMA,
        "side": normalized,
        "source_schema": "V9_SIDE_FEATURES_176",
        "feature_names": names,
        "dtype": V17_DTYPE,
        "normalization": V17_NORMALIZATION_CONTRACT,
    }
    return {
        **material,
        "feature_names": names,
        "source_indices": indices,
        "feature_count": len(names),
        "schema_hash": Sha256HashProvider().digest_value(material),
    }


def select_v17_features(side: str, v9_values: Sequence[Any]) -> V17FeatureVector:
    if len(v9_values) != len(V9_FEATURE_NAMES):
        raise V17FeatureContractError("V17_SOURCE_FEATURE_WIDTH_MISMATCH")
    try:
        source = tuple(float(value) for value in v9_values)
    except (TypeError, ValueError) as exc:
        raise V17FeatureContractError("V17_SOURCE_FEATURE_DTYPE_INVALID") from exc
    if not all(math.isfinite(value) for value in source):
        raise V17FeatureContractError("V17_SOURCE_FEATURE_VALUE_INVALID")
    contract = contract_for_side(side)
    vector = V17FeatureVector(
        side=side,
        schema_version=V17_FEATURE_SCHEMA,
        schema_hash=str(contract["schema_hash"]),
        names=tuple(contract["feature_names"]),
        values=tuple(source[index] for index in contract["source_indices"]),
    )
    vector.validate()
    return vector


def build_v17_runtime_features(
    *,
    side: str,
    base_features: Mapping[str, Any],
    history: Sequence[Candle],
    pipeline: DeterministicFeaturePipeline | None = None,
) -> V17FeatureVector:
    if tuple(base_features) != FEATURE_NAMES:
        missing = tuple(name for name in FEATURE_NAMES if name not in base_features)
        extra = tuple(name for name in base_features if name not in FEATURE_NAMES)
        code = "V17_BASE_FEATURE_ORDER_MISMATCH"
        if missing:
            code = "V17_BASE_FEATURE_MISSING"
        elif extra:
            code = "V17_BASE_FEATURE_EXTRA"
        raise V17FeatureContractError(f"{code}:missing={missing}:extra={extra}")
    if len(history) < V17_REQUIRED_CLOSED_5M_BARS or not all(
        candle.is_closed for candle in history[-V17_REQUIRED_CLOSED_5M_BARS:]
    ):
        raise V17FeatureContractError("V17_CLOSED_HISTORY_INSUFFICIENT")
    try:
        values = {name: float(base_features[name]) for name in FEATURE_NAMES}
    except (TypeError, ValueError) as exc:
        raise V17FeatureContractError("V17_BASE_FEATURE_DTYPE_INVALID") from exc
    if not all(math.isfinite(value) for value in values.values()):
        raise V17FeatureContractError("V17_BASE_FEATURE_VALUE_INVALID")

    direction = _side(side)
    causal_history = tuple(history[-V17_REQUIRED_CLOSED_5M_BARS:])
    feature_pipeline = pipeline or DeterministicFeaturePipeline()
    multitimeframe, context = multitimeframe_long_features(
        values, causal_history, pipeline=feature_pipeline
    )
    regime = classify_regime_axes(values, context)
    role = directional_role(direction, regime["direction"])
    v6 = regime_aware_feature_vector(
        multitimeframe_features=multitimeframe,
        base_features=values,
        side=direction,
        regime=regime,
    )
    v7, archetype, _ = v7_feature_vector(
        {
            "side": direction.value,
            "features": v6,
            "regime": regime,
            "directional_role": role.value,
        }
    )
    v8, _ = v8_feature_vector(
        {
            "v7_features": v7,
            "regime": regime,
            "directional_role": role.value,
            "v7_archetype": archetype.value,
        }
    )
    rolling = rolling_four_hour_context(causal_history)
    v9, _, _ = v9_feature_vectors(
        {
            "side": direction.value,
            "v7_features": v7,
            "v8_features": v8,
        },
        rolling,
    )
    return select_v17_features(direction.value, v9)
