"""Deterministic loader for non-authoritative V17 research artifacts."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from .tree_models import TreeEnsemble
from .utils import Sha256HashProvider
from .v17_feature_contract import V17FeatureVector, contract_for_side


class V17ResearchArtifactError(ValueError):
    pass


def _sigmoid(value: float) -> float:
    if value >= 0.0:
        z = math.exp(-min(value, 700.0))
        return 1.0 / (1.0 + z)
    z = math.exp(max(value, -700.0))
    return z / (1.0 + z)


@dataclass(frozen=True)
class FrozenLinearModel:
    feature_names: tuple[str, ...]
    means: tuple[float, ...]
    scales: tuple[float, ...]
    coefficients: tuple[float, ...]
    intercept: float
    output: str

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "FrozenLinearModel":
        unsigned = dict(payload)
        claimed = str(unsigned.pop("content_hash", ""))
        if len(claimed) != 64 or claimed != Sha256HashProvider().digest_value(unsigned):
            raise V17ResearchArtifactError("V17_LINEAR_MODEL_HASH_MISMATCH")
        if payload.get("schema_id") != "aegis-v17-frozen-linear-v1":
            raise V17ResearchArtifactError("V17_LINEAR_MODEL_SCHEMA_MISMATCH")
        try:
            result = cls(
                feature_names=tuple(str(value) for value in payload["feature_names"]),
                means=tuple(float(value) for value in payload["means"]),
                scales=tuple(float(value) for value in payload["scales"]),
                coefficients=tuple(float(value) for value in payload["coefficients"]),
                intercept=float(payload["intercept"]),
                output=str(payload["output"]),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise V17ResearchArtifactError("V17_LINEAR_MODEL_INVALID") from exc
        widths = {len(result.feature_names), len(result.means), len(result.scales), len(result.coefficients)}
        if len(widths) != 1 or not result.feature_names or len(set(result.feature_names)) != len(result.feature_names):
            raise V17ResearchArtifactError("V17_LINEAR_MODEL_WIDTH_MISMATCH")
        if result.output not in {"PROBABILITY", "RAW_SCORE"}:
            raise V17ResearchArtifactError("V17_LINEAR_MODEL_OUTPUT_INVALID")
        if not all(math.isfinite(value) for value in (*result.means, *result.scales, *result.coefficients, result.intercept)) or any(value <= 0.0 for value in result.scales):
            raise V17ResearchArtifactError("V17_LINEAR_MODEL_VALUE_INVALID")
        return result

    def evaluate(self, values: Sequence[float]) -> float:
        if len(values) != len(self.feature_names):
            raise V17ResearchArtifactError("V17_LINEAR_INPUT_WIDTH_MISMATCH")
        normalized = tuple((float(value) - mean) / scale for value, mean, scale in zip(values, self.means, self.scales))
        if not all(math.isfinite(value) for value in normalized):
            raise V17ResearchArtifactError("V17_LINEAR_INPUT_INVALID")
        raw = self.intercept + math.fsum(weight * value for weight, value in zip(self.coefficients, normalized))
        return _sigmoid(raw) if self.output == "PROBABILITY" else raw


@dataclass(frozen=True)
class V17SideArtifact:
    side: str
    status: str
    feature_schema_hash: str
    clean: FrozenLinearModel
    danger: FrozenLinearModel
    mae_q90: TreeEnsemble
    ranker: FrozenLinearModel
    gate_thresholds: Mapping[str, float]
    minimum_rank_score: float | None

    def score(self, vector: V17FeatureVector) -> Mapping[str, Any]:
        vector.validate()
        if vector.side != self.side or vector.schema_hash != self.feature_schema_hash:
            raise V17ResearchArtifactError("V17_ARTIFACT_FEATURE_CONTRACT_MISMATCH")
        names = tuple(vector.names)
        if any(model.feature_names != names for model in (self.clean, self.danger, self.ranker)) or self.mae_q90.feature_names != names:
            raise V17ResearchArtifactError("V17_ARTIFACT_FEATURE_ORDER_MISMATCH")
        clean = self.clean.evaluate(vector.values)
        danger = self.danger.evaluate(vector.values)
        mae = max(0.0, self.mae_q90.evaluate(vector.values))
        rank = self.ranker.evaluate(vector.values)
        selected = False
        if self.minimum_rank_score is not None:
            selected = (
                clean >= float(self.gate_thresholds["minimum_clean_probability"])
                and danger <= float(self.gate_thresholds["maximum_danger_probability"])
                and mae <= float(self.gate_thresholds["maximum_mae_q90"])
                and rank >= self.minimum_rank_score
            )
        return {
            "clean_probability": clean,
            "danger_probability": danger,
            "mae_q90": mae,
            "rank_score": rank,
            "selected": selected,
            "policy_status": self.status,
        }


@dataclass(frozen=True)
class V17ResearchArtifact:
    content_hash: str
    sides: Mapping[str, V17SideArtifact]
    promotion_authority: bool

    @classmethod
    def load(cls, path: Path) -> "V17ResearchArtifact":
        payload = json.loads(path.read_text(encoding="utf-8"))
        claimed = str(payload.get("content_hash", ""))
        unsigned = dict(payload)
        unsigned.pop("content_hash", None)
        if claimed != Sha256HashProvider().digest_value(unsigned):
            raise V17ResearchArtifactError("V17_ARTIFACT_HASH_MISMATCH")
        if payload.get("schema_id") != "aegis-v17-research-artifact-v1" or bool(payload.get("promotion_authority")):
            raise V17ResearchArtifactError("V17_ARTIFACT_AUTHORITY_INVALID")
        sides = {}
        for side in ("LONG", "SHORT"):
            raw = payload["sides"][side]
            contract = contract_for_side(side)
            if str(raw["feature_schema_hash"]) != contract["schema_hash"]:
                raise V17ResearchArtifactError("V17_ARTIFACT_SCHEMA_HASH_MISMATCH")
            policy = raw.get("policy")
            sides[side] = V17SideArtifact(
                side=side,
                status=str(raw["status"]),
                feature_schema_hash=str(raw["feature_schema_hash"]),
                clean=FrozenLinearModel.from_payload(raw["models"]["clean"]),
                danger=FrozenLinearModel.from_payload(raw["models"]["danger"]),
                mae_q90=TreeEnsemble.from_payload(raw["models"]["mae_q90"]),
                ranker=FrozenLinearModel.from_payload(raw["models"]["ranker"]),
                gate_thresholds={str(k): float(v) for k, v in raw["gate"]["thresholds"].items()},
                minimum_rank_score=(float(policy["minimum_score"]) if isinstance(policy, Mapping) else None),
            )
        return cls(claimed, sides, False)
