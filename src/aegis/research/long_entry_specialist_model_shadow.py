"""Runtime evaluator for observational LONG specialist artifacts."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from ..training.hybrid_directional import calibrator_from_mapping
from ..tree_models import TreeEnsemble
from ..utils import Sha256HashProvider, sha256_file
from .long_entry_specialists_shadow import (
    LONG_SPECIALIST_FEATURE_NAMES,
    LongArchetype,
    classify_long_archetype,
    long_specialist_feature_vector,
)


class LongEntrySpecialistModelShadowError(ValueError):
    pass


@dataclass(frozen=True)
class _Specialist:
    model: TreeEnsemble
    calibrator: Any
    threshold: float
    validation_pass: bool


class LongEntrySpecialistModelShadow:
    """Expose model evidence while retaining zero execution authority."""

    def __init__(self, path: Path) -> None:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise LongEntrySpecialistModelShadowError(
                "LONG specialist artifact is unreadable"
            ) from exc
        if not isinstance(payload, Mapping):
            raise LongEntrySpecialistModelShadowError(
                "LONG specialist artifact is invalid"
            )
        unsigned = dict(payload)
        content_hash = str(unsigned.pop("content_hash", ""))
        if (
            content_hash != Sha256HashProvider().digest_value(unsigned)
            or payload.get("schema_id")
            != "aegis-long-entry-specialists-shadow-validation-v1"
            or payload.get("mode") != "SHADOW"
            or tuple(payload.get("feature_names", ())) != LONG_SPECIALIST_FEATURE_NAMES
            or payload.get("live_selection_effect") != "NONE"
            or payload.get("exchange_authority") is not False
            or int(payload.get("exchange_mutations", -1)) != 0
        ):
            raise LongEntrySpecialistModelShadowError(
                "LONG specialist artifact authority is invalid"
            )
        try:
            danger = payload["shared_danger_model"]
            if not isinstance(danger, Mapping):
                raise TypeError
            self._danger_model = TreeEnsemble.from_payload(danger["model"])
            self._danger_calibrator = calibrator_from_mapping(danger["calibrator"])
            self._danger_threshold = float(danger["maximum_probability"])
            if not math.isfinite(self._danger_threshold):
                raise ValueError
            raw_specialists = payload["specialists"]
            if not isinstance(raw_specialists, Mapping):
                raise TypeError
            specialists: dict[str, _Specialist] = {}
            for archetype in (
                LongArchetype.TREND_CONTINUATION.value,
                LongArchetype.CONFIRMED_REVERSAL.value,
            ):
                raw = raw_specialists[archetype]
                if (
                    not isinstance(raw, Mapping)
                    or raw.get("status") != "TRAINED_SHADOW_ONLY"
                ):
                    continue
                threshold = float(raw["success_threshold"])
                if not math.isfinite(threshold) or not 0.0 <= threshold <= 1.0:
                    raise ValueError
                specialists[archetype] = _Specialist(
                    TreeEnsemble.from_payload(raw["model"]),
                    calibrator_from_mapping(raw["calibrator"]),
                    threshold,
                    raw.get("validation_pass") is True,
                )
        except (KeyError, TypeError, ValueError) as exc:
            raise LongEntrySpecialistModelShadowError(
                "LONG specialist model is invalid"
            ) from exc
        self.path = path.resolve()
        self.sha256 = sha256_file(path)
        self.content_hash = content_hash
        self.validation_pass = payload.get("validation_pass") is True
        self._specialists = specialists

    def assess(self, features: Mapping[str, Any]) -> Mapping[str, Any]:
        archetype = classify_long_archetype(features)
        name = str(archetype["archetype"])
        specialist = self._specialists.get(name)
        if specialist is None:
            return {
                "schema_id": "aegis-long-entry-specialist-shadow-score-v1",
                "mode": "SHADOW",
                "archetype": name,
                "archetype_evidence": archetype,
                "status": "NO_TRAINED_SPECIALIST_FOR_ARCHETYPE",
                "success_probability": None,
                "danger_probability": None,
                "counterfactual_action": "OBSERVE_ONLY",
                "selection_effect": "NONE",
                "exchange_authority": False,
                "exchange_mutations": 0,
            }
        vector = long_specialist_feature_vector(features)
        success_probability = specialist.calibrator.apply(
            specialist.model.evaluate(vector)
        )
        danger_probability = self._danger_calibrator.apply(
            self._danger_model.evaluate(vector)
        )
        counterfactual_pass = (
            specialist.validation_pass
            and success_probability >= specialist.threshold
            and danger_probability <= self._danger_threshold
        )
        return {
            "schema_id": "aegis-long-entry-specialist-shadow-score-v1",
            "mode": "SHADOW",
            "archetype": name,
            "archetype_evidence": archetype,
            "status": (
                "VALIDATED_SHADOW_SCORE"
                if specialist.validation_pass
                else "RESEARCH_SCORE_NOT_VALIDATED"
            ),
            "success_probability": success_probability,
            "success_threshold": specialist.threshold,
            "danger_probability": danger_probability,
            "maximum_danger_probability": self._danger_threshold,
            "counterfactual_pass": counterfactual_pass,
            "counterfactual_action": (
                "COUNTERFACTUAL_ENTER" if counterfactual_pass else "OBSERVE_ONLY"
            ),
            "artifact_content_hash": self.content_hash,
            "selection_effect": "NONE",
            "exchange_authority": False,
            "exchange_mutations": 0,
        }
