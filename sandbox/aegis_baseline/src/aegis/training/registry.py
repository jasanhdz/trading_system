"""Immutable explicit artifact publication; never auto-promotes."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Protocol

from .evaluate import EvaluationReport
from .train import ModelArtifact
from ..utils import HashProvider, to_primitive


class ArtifactRegistry(Protocol):
    def publish(self, artifact: ModelArtifact, evaluation: EvaluationReport) -> str: ...
    def load(self, artifact_id: str) -> ModelArtifact: ...


class FileArtifactRegistry:
    def __init__(self, root: Path, hashing: HashProvider) -> None:
        self.root = root
        self.hashing = hashing

    def publish(self, artifact: ModelArtifact, evaluation: EvaluationReport) -> str:
        if evaluation.artifact_id != artifact.artifact_id or not evaluation.accepted:
            raise ValueError("only explicitly accepted matching artifacts may be published")
        path = self.root / f"{artifact.artifact_id}.json"
        if path.exists():
            raise FileExistsError(f"immutable artifact already exists: {artifact.artifact_id}")
        self.root.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(to_primitive(artifact), sort_keys=True, separators=(",", ":")), encoding="utf-8")
        return artifact.artifact_id

    def load(self, artifact_id: str) -> ModelArtifact:
        payload = json.loads((self.root / f"{artifact_id}.json").read_text(encoding="utf-8"))
        artifact = ModelArtifact(
            artifact_id=payload["artifact_id"], model_family=payload["model_family"], dataset_id=payload["dataset_id"],
            feature_schema_version=payload["feature_schema_version"], feature_hash=payload["feature_hash"],
            coefficients={key: tuple(value) for key, value in payload["coefficients"].items()},
            intercepts={key: float(value) for key, value in payload["intercepts"].items()}, artifact_hash=payload["artifact_hash"],
        )
        expected = self.hashing.digest_value({"dataset": artifact.dataset_id, "features": artifact.feature_hash,
                                              "coefficients": artifact.coefficients, "intercepts": artifact.intercepts})
        if expected != artifact.artifact_hash:
            raise ValueError("published artifact hash mismatch")
        return artifact
