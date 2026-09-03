"""Fail-closed execution contract for the V17 research challenger.

This module deliberately does not load or train a model.  It defines the
canonical decision envelope that a future, reproducibly exported V17 artifact
must satisfy before its output can enter the existing TypeScript execution
pipeline.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Mapping

import yaml

from .utils import sha256_file


class V17ChallengerError(ValueError):
    pass


class V17RuntimeMode(str, Enum):
    OFF = "OFF"
    SHADOW = "SHADOW"
    READY_INACTIVE = "READY_INACTIVE"


@dataclass(frozen=True)
class V17ChallengerConfig:
    schema_version: str
    challenger_id: str
    mode: V17RuntimeMode
    execution_authority: bool
    model_artifact: str | None
    model_sha256: str | None
    research_artifact: str | None
    research_artifact_sha256: str | None
    promotion_gate_passed: bool
    promotion_blockers: tuple[str, ...]
    feature_schema: str
    long_feature_count: int
    short_feature_count: int
    required_closed_5m_bars: int

    @property
    def model_available(self) -> bool:
        return bool(self.model_artifact and self.model_sha256)

    @property
    def execution_ready(self) -> bool:
        return (
            self.mode is V17RuntimeMode.READY_INACTIVE
            and self.model_available
            and self.promotion_gate_passed
            and not self.promotion_blockers
        )

    def health(self) -> Mapping[str, Any]:
        blocker = self.promotion_blockers[0] if self.promotion_blockers else None
        if not self.model_available and blocker is None:
            blocker = "V17_EXECUTION_ARTIFACT_REQUIRED"
        return {
            "schema_version": self.schema_version,
            "challenger_id": self.challenger_id,
            "mode": self.mode.value,
            "execution_authority": self.execution_authority,
            "model_available": self.model_available,
            "research_artifact_available": bool(
                self.research_artifact and self.research_artifact_sha256
            ),
            "promotion_gate_passed": self.promotion_gate_passed,
            "promotion_blockers": self.promotion_blockers,
            "execution_contract_ready": self.execution_ready,
            "blocker": blocker,
            "feature_schema": self.feature_schema,
            "long_feature_count": self.long_feature_count,
            "short_feature_count": self.short_feature_count,
            "required_closed_5m_bars": self.required_closed_5m_bars,
        }


@dataclass(frozen=True)
class V17CanonicalDecision:
    symbol: str
    side: str
    selected: bool
    clean_probability: float
    danger_probability: float
    mae_q90: float
    rank_score: float
    minimum_clean_probability: float
    maximum_danger_probability: float
    maximum_mae_q90: float
    minimum_rank_score: float
    expected_price: float
    market_timestamp: str
    feature_hash: str
    model_identifier: str
    model_sha256: str
    policy_identifier: str

    def __post_init__(self) -> None:
        if self.side not in {"LONG", "SHORT"}:
            raise V17ChallengerError("V17 side must be LONG or SHORT")
        if not self.symbol.endswith("USDT"):
            raise V17ChallengerError("V17 symbol is invalid")
        probabilities = (self.clean_probability, self.danger_probability)
        finite = (
            *probabilities,
            self.mae_q90,
            self.rank_score,
            self.minimum_clean_probability,
            self.maximum_danger_probability,
            self.maximum_mae_q90,
            self.minimum_rank_score,
            self.expected_price,
        )
        if not all(math.isfinite(value) for value in finite):
            raise V17ChallengerError("V17 decision contains non-finite values")
        if not all(0.0 <= value <= 1.0 for value in probabilities):
            raise V17ChallengerError("V17 probabilities are outside [0, 1]")
        if min(self.mae_q90, self.maximum_mae_q90, self.expected_price) < 0.0:
            raise V17ChallengerError("V17 decision contains a negative magnitude")
        gate_passed = (
            self.clean_probability >= self.minimum_clean_probability
            and self.danger_probability <= self.maximum_danger_probability
            and self.mae_q90 <= self.maximum_mae_q90
            and self.rank_score >= self.minimum_rank_score
        )
        if self.selected != gate_passed:
            raise V17ChallengerError("V17 selected flag disagrees with frozen policy")
        for identity in (
            self.feature_hash,
            self.model_identifier,
            self.model_sha256,
            self.policy_identifier,
            self.market_timestamp,
        ):
            if not identity:
                raise V17ChallengerError("V17 decision identity is incomplete")

    def telemetry(self) -> Mapping[str, Any]:
        return {
            "schema_id": "aegis-v17-canonical-decision-v1",
            "symbol": self.symbol,
            "side": self.side,
            "selected": self.selected,
            "clean_probability": self.clean_probability,
            "danger_probability": self.danger_probability,
            "mae_q90": self.mae_q90,
            "rank_score": self.rank_score,
            "thresholds": {
                "minimum_clean_probability": self.minimum_clean_probability,
                "maximum_danger_probability": self.maximum_danger_probability,
                "maximum_mae_q90": self.maximum_mae_q90,
                "minimum_rank_score": self.minimum_rank_score,
            },
            "expected_price": self.expected_price,
            "market_timestamp": self.market_timestamp,
            "feature_hash": self.feature_hash,
            "model_identifier": self.model_identifier,
            "model_sha256": self.model_sha256,
            "policy_identifier": self.policy_identifier,
        }


def load_v17_challenger_config(path: Path) -> V17ChallengerConfig:
    payload = yaml.safe_load(path.read_text())
    if not isinstance(payload, Mapping):
        raise V17ChallengerError("V17 challenger config must be a mapping")
    config = V17ChallengerConfig(
        schema_version=str(payload.get("schema_version", "")),
        challenger_id=str(payload.get("challenger_id", "")),
        mode=V17RuntimeMode(str(payload.get("mode", ""))),
        execution_authority=bool(payload.get("execution_authority", False)),
        model_artifact=(str(payload["model_artifact"]) if payload.get("model_artifact") else None),
        model_sha256=(str(payload["model_sha256"]) if payload.get("model_sha256") else None),
        research_artifact=(
            str(payload["research_artifact"]) if payload.get("research_artifact") else None
        ),
        research_artifact_sha256=(
            str(payload["research_artifact_sha256"])
            if payload.get("research_artifact_sha256")
            else None
        ),
        promotion_gate_passed=bool(payload.get("promotion_gate_passed", False)),
        promotion_blockers=tuple(str(value) for value in payload.get("promotion_blockers", ())),
        feature_schema=str(payload.get("feature_schema", "")),
        long_feature_count=int(payload.get("long_feature_count", 0)),
        short_feature_count=int(payload.get("short_feature_count", 0)),
        required_closed_5m_bars=int(payload.get("required_closed_5m_bars", 0)),
    )
    if config.schema_version != "aegis-v17-execution-challenger-v1":
        raise V17ChallengerError("unexpected V17 challenger schema")
    if config.execution_authority:
        raise V17ChallengerError("V17 challenger must not have execution authority")
    if config.promotion_gate_passed:
        raise V17ChallengerError("V17 promotion gate must remain closed")
    if not config.promotion_blockers:
        raise V17ChallengerError("V17 promotion blockers must be explicit")
    if bool(config.research_artifact) != bool(config.research_artifact_sha256):
        raise V17ChallengerError("V17 research artifact identity is incomplete")
    if config.research_artifact:
        artifact_path = Path(config.research_artifact)
        if not artifact_path.is_absolute():
            artifact_path = path.resolve().parents[1] / artifact_path
        if not artifact_path.is_file():
            raise V17ChallengerError("V17 research artifact is missing")
        if sha256_file(artifact_path) != config.research_artifact_sha256:
            raise V17ChallengerError("V17 research artifact hash mismatch")
    if config.long_feature_count != 129 or config.short_feature_count != 168:
        raise V17ChallengerError("V17 directional feature contract changed")
    if config.required_closed_5m_bars < 576:
        raise V17ChallengerError("V17 history contract is insufficient")
    return config
