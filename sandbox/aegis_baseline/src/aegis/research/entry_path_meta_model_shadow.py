"""Inspectable side-specific clean-entry scorer with Shadow-only authority."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from ..training.hybrid_directional import calibrator_from_mapping
from ..tree_models import TreeEnsemble
from ..utils import Sha256HashProvider, sha256_file
from .entry_methodology_v2_shadow import (
    ENTRY_PATH_MODEL_FEATURE_NAMES,
    entry_path_model_features,
)


class EntryPathMetaModelShadowError(ValueError):
    pass


@dataclass(frozen=True)
class _SideModel:
    model: TreeEnsemble
    calibrator: Any
    threshold: float
    validation_pass: bool


class EntryPathMetaModelShadow:
    """Score entry-path quality without selecting or mutating anything."""

    def __init__(self, path: Path) -> None:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise EntryPathMetaModelShadowError(
                "entry path artifact is unreadable"
            ) from exc
        if not isinstance(payload, Mapping):
            raise EntryPathMetaModelShadowError("entry path artifact is invalid")
        unsigned = dict(payload)
        claimed_hash = str(unsigned.pop("content_hash", ""))
        if (
            claimed_hash != Sha256HashProvider().digest_value(unsigned)
            or payload.get("schema_id")
            != "aegis-entry-path-meta-model-shadow-validation-v1"
            or payload.get("mode") != "SHADOW"
            or payload.get("target") != "CLEAN_FAST_SUCCESS"
            or tuple(payload.get("feature_names", ())) != ENTRY_PATH_MODEL_FEATURE_NAMES
            or payload.get("live_selection_effect") != "NONE"
            or payload.get("exchange_authority") is not False
            or int(payload.get("exchange_mutations", -1)) != 0
        ):
            raise EntryPathMetaModelShadowError(
                "entry path artifact authority is invalid"
            )
        raw_models = payload.get("models")
        if not isinstance(raw_models, Mapping) or set(raw_models) != {"LONG", "SHORT"}:
            raise EntryPathMetaModelShadowError("entry path side models are invalid")
        models: dict[str, _SideModel] = {}
        try:
            for side in ("LONG", "SHORT"):
                raw = raw_models[side]
                if not isinstance(raw, Mapping):
                    raise TypeError
                threshold = float(raw["selection_threshold"])
                if not math.isfinite(threshold) or not 0.0 <= threshold <= 1.0:
                    raise ValueError
                models[side] = _SideModel(
                    model=TreeEnsemble.from_payload(raw["model"]),
                    calibrator=calibrator_from_mapping(raw["calibrator"]),
                    threshold=threshold,
                    validation_pass=raw.get("validation_pass") is True,
                )
        except (KeyError, TypeError, ValueError) as exc:
            raise EntryPathMetaModelShadowError(
                "entry path side model is invalid"
            ) from exc
        self.path = path.resolve()
        self.sha256 = sha256_file(path)
        self.content_hash = claimed_hash
        self._models = models

    def assess(
        self,
        *,
        side: str,
        prediction: Mapping[str, Any],
        confirmation: Mapping[str, Any],
        confirmation_features: Mapping[str, Any],
        entry_intelligence: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        if side not in self._models:
            raise EntryPathMetaModelShadowError("entry path side is invalid")
        side_model = self._models[side]
        if not side_model.validation_pass:
            return {
                "schema_id": "aegis-entry-path-meta-model-shadow-score-v1",
                "mode": "SHADOW",
                "side": side,
                "status": "SIDE_MODEL_NOT_VALIDATED",
                "probability": None,
                "threshold": side_model.threshold,
                "counterfactual_pass": False,
                "selection_effect": "NONE",
                "exchange_authority": False,
                "exchange_mutations": 0,
            }
        features = entry_path_model_features(
            side=side,
            prediction=prediction,
            confirmation=confirmation,
            confirmation_features=confirmation_features,
            entry_intelligence=entry_intelligence,
        )
        raw_probability = side_model.model.evaluate(features)
        probability = side_model.calibrator.apply(raw_probability)
        return {
            "schema_id": "aegis-entry-path-meta-model-shadow-score-v1",
            "mode": "SHADOW",
            "side": side,
            "status": "VALIDATED_SHADOW_SCORE",
            "probability": probability,
            "threshold": side_model.threshold,
            "counterfactual_pass": probability >= side_model.threshold,
            "artifact_content_hash": self.content_hash,
            "selection_effect": "NONE",
            "exchange_authority": False,
            "exchange_mutations": 0,
        }
