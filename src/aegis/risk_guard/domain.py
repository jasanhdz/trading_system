"""Domain types for the Risk Guard architecture."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any


class Direction(Enum):
    """Trading direction proposed by Aegis."""
    LONG = "LONG"
    SHORT = "SHORT"
    SKIP = "SKIP"


class RiskDecision(Enum):
    """Binary risk guard decision."""
    ALLOW = "ALLOW"
    BLOCK = "BLOCK"


class RiskGuardVerdict(Enum):
    """Full verdict including observation mode."""
    ALLOW = "ALLOW"
    BLOCK = "BLOCK"
    OBSERVED_BLOCK = "OBSERVED_BLOCK"


@dataclass(frozen=True)
class Signal:
    """An Aegis trading signal before risk evaluation."""
    signal_id: str
    timestamp: datetime
    symbol: str
    side: Direction
    direction_source: str
    direction_model_version: str
    turbo_score: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def timestamp_iso(self) -> str:
        return self.timestamp.isoformat()


@dataclass(frozen=True)
class RiskGuardResult:
    """Result of a risk guard evaluation."""
    decision: RiskDecision
    score: float
    threshold: float
    model_version: str
    feature_snapshot_hash: str
    reason: str
    evaluation_time_ms: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class EntryDecision:
    """Final entry decision combining direction + risk guard."""
    signal: Signal
    risk_result: RiskGuardResult
    verdict: RiskGuardVerdict
    enforced: bool
    observe_only: bool

    @property
    def would_block(self) -> bool:
        """True if the risk guard would block in enforce mode."""
        return self.risk_result.decision == RiskDecision.BLOCK

    @property
    def is_effective_block(self) -> bool:
        """True if the decision actually blocks the trade."""
        return self.verdict == RiskGuardVerdict.BLOCK

    def to_dict(self) -> dict[str, Any]:
        return {
            "signal_id": self.signal.signal_id,
            "timestamp": self.signal.timestamp_iso,
            "symbol": self.signal.symbol,
            "side": self.signal.side.value,
            "direction_source": self.signal.direction_source,
            "direction_model_version": self.signal.direction_model_version,
            "tail_risk_score": self.risk_result.score,
            "tail_risk_threshold": self.risk_result.threshold,
            "risk_decision": self.risk_result.decision.value,
            "verdict": self.verdict.value,
            "enforced": self.enforced,
            "observe_only": self.observe_only,
            "reason": self.risk_result.reason,
            "model_version": self.risk_result.model_version,
            "feature_snapshot_hash": self.risk_result.feature_snapshot_hash,
            "evaluation_time_ms": self.risk_result.evaluation_time_ms,
        }


@dataclass(frozen=True)
class RiskGuardConfig:
    """Configuration for the risk guard system.

    tail_risk_threshold is FROZEN at V1 value (0.4522452210875323).
    It cannot be changed at runtime.
    """
    enabled: bool = False
    mode: str = "observe_only"
    tail_risk_threshold: float = 0.4522452210875323
    models_joblib_path: str = ""
    models_joblib_sha256: str = ""
    feature_schema_path: str = ""
    feature_schema_sha256: str = ""
    candle_data_root: str = ""
    fail_closed: bool = True

    @property
    def enforce(self) -> bool:
        return self.enabled and self.mode == "enforce"

    @property
    def observe_only(self) -> bool:
        return self.enabled and self.mode == "observe_only"
