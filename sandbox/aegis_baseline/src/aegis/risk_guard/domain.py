"""Domain types for the Risk Guard architecture."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any

FROZEN_TAIL_RISK_THRESHOLD = 0.4522452210875323


class Direction(Enum):
    """Trading direction proposed by Aegis."""
    LONG = "LONG"
    SHORT = "SHORT"
    SKIP = "SKIP"


class RiskDecision(Enum):
    """Risk guard decision with feature availability states."""
    ALLOW = "ALLOW"
    BLOCK = "BLOCK"
    FEATURES_UNAVAILABLE = "FEATURES_UNAVAILABLE"
    STALE_DATA = "STALE_DATA"
    NON_CAUSAL_DATA = "NON_CAUSAL_DATA"
    FEATURE_BUILD_ERROR = "FEATURE_BUILD_ERROR"


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
    feature_available_at: datetime | None = None
    feature_build_latency_ms: float = 0.0
    feature_staleness_ms: float = 0.0
    source_feed_lag_ms: dict[str, float] | None = None


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
            "feature_available_at": self.risk_result.feature_available_at.isoformat() if self.risk_result.feature_available_at else None,
            "feature_build_latency_ms": self.risk_result.feature_build_latency_ms,
            "feature_staleness_ms": self.risk_result.feature_staleness_ms,
            "source_feed_lag_ms": self.risk_result.source_feed_lag_ms,
        }


@dataclass(frozen=True)
class RiskGuardConfig:
    """Configuration for the risk guard system.

    tail_risk_threshold is FROZEN at V1 value (0.4522452210875323).
    fail_closed is FROZEN at True for V1 — no fail-open allowed.
    Any attempt to set different values will raise ValueError.
    """
    enabled: bool = False
    mode: str = "observe_only"
    tail_risk_threshold: float = FROZEN_TAIL_RISK_THRESHOLD
    models_joblib_path: str = ""
    models_joblib_sha256: str = ""
    feature_schema_path: str = ""
    feature_schema_sha256: str = ""
    candle_data_root: str = ""
    fail_closed: bool = True

    _VALID_MODES = frozenset({"disabled", "observe_only", "enforce"})

    def __post_init__(self) -> None:
        if self.tail_risk_threshold != FROZEN_TAIL_RISK_THRESHOLD:
            raise ValueError(
                f"tail_risk_threshold must be FROZEN at {FROZEN_TAIL_RISK_THRESHOLD}, "
                f"got {self.tail_risk_threshold}. "
                f"Threshold cannot be changed in V1."
            )
        if self.fail_closed is not True:
            raise ValueError(
                "fail_closed must be True for E4 V1. "
                "Fail-open is not permitted."
            )
        if self.mode not in self._VALID_MODES:
            raise ValueError(
                f"Invalid mode '{self.mode}'. Must be one of: {sorted(self._VALID_MODES)}"
            )
        if self.enabled and self.mode == "disabled":
            raise ValueError(
                "Contradictory state: enabled=True with mode='disabled'. "
                "Use mode='observe_only' or mode='enforce' when enabled."
            )

    @property
    def enforce(self) -> bool:
        return self.enabled and self.mode == "enforce"

    @property
    def observe_only(self) -> bool:
        return self.enabled and self.mode == "observe_only"
