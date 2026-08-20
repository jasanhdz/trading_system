"""Observability and structured logging for Risk Guard decisions."""

from __future__ import annotations

import json
import logging
import threading
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .domain import EntryDecision, RiskDecision, RiskGuardVerdict

logger = logging.getLogger(__name__)


class RiskGuardObserver:
    """Logs structured risk guard decisions for counterfactual reconstruction.

    Every evaluation produces a log entry that allows later reconstruction of:
    - What Aegis proposed (direction, side, confidence)
    - What E4 scored (tail_risk_score, threshold)
    - What decision was made (ALLOW/BLOCK)
    - Whether it was enforced or observed-only

    Logs are written to:
    - Python logging (structured dict)
    - Optional JSONL file for offline analysis
    """

    def __init__(self, jsonl_path: str | Path | None = None) -> None:
        self._jsonl_path = Path(jsonl_path) if jsonl_path else None
        if self._jsonl_path:
            self._jsonl_path.parent.mkdir(parents=True, exist_ok=True)

    def record(self, decision: EntryDecision) -> None:
        """Record a risk guard decision."""
        entry = self._build_entry(decision)

        logger.info(
            "RISK_GUARD_DECISION %s",
            json.dumps(entry, default=str),
        )

        if self._jsonl_path:
            self._append_jsonl(entry)

    def _build_entry(self, decision: EntryDecision) -> dict[str, Any]:
        return {
            "event": "risk_guard_decision",
            "ts": datetime.now(timezone.utc).isoformat(),
            "signal_id": decision.signal.signal_id,
            "signal_timestamp": decision.signal.timestamp_iso,
            "symbol": decision.signal.symbol,
            "side": decision.signal.side.value,
            "direction_source": decision.signal.direction_source,
            "direction_model_version": decision.signal.direction_model_version,
            "tail_risk_score": decision.risk_result.score,
            "tail_risk_threshold": decision.risk_result.threshold,
            "risk_decision": decision.risk_result.decision.value,
            "verdict": decision.verdict.value,
            "enforced": decision.enforced,
            "observe_only": decision.observe_only,
            "would_block": decision.would_block,
            "reason": decision.risk_result.reason,
            "model_version": decision.risk_result.model_version,
            "feature_snapshot_hash": decision.risk_result.feature_snapshot_hash,
            "evaluation_time_ms": decision.risk_result.evaluation_time_ms,
            "feature_available_at": decision.risk_result.feature_available_at.isoformat() if decision.risk_result.feature_available_at else None,
            "feature_build_latency_ms": decision.risk_result.feature_build_latency_ms,
            "feature_staleness_ms": decision.risk_result.feature_staleness_ms,
            "source_feed_lag_ms": decision.risk_result.source_feed_lag_ms,
        }

    def _append_jsonl(self, entry: dict[str, Any]) -> None:
        try:
            with self._jsonl_path.open("a") as f:
                f.write(json.dumps(entry, default=str) + "\n")
        except Exception as exc:
            logger.warning("Failed to write risk guard log: %s", exc)


class RiskGuardMetrics:
    """In-memory metrics for the risk guard system.

    Tracks:
    - Total signals evaluated
    - ALLOW vs BLOCK counts
    - Observe-only blocks (would have blocked but didn't)
    - Score distributions
    - Per-symbol and per-side breakdowns
    """

    def __init__(self) -> None:
        self._reset()

    def _reset(self) -> None:
        self.total_evaluated = 0
        self.total_allowed = 0
        self.total_blocked = 0
        self.total_observed_blocks = 0
        self.total_enforced_blocks = 0
        self.total_skip = 0
        self.total_guard_disabled = 0
        self.total_feature_errors = 0
        self.total_stale_data = 0
        self.total_features_unavailable = 0
        self.scores: list[float] = []
        self.feature_build_latencies: list[float] = []
        self.per_symbol: dict[str, dict[str, int]] = {}
        self.per_side: dict[str, dict[str, int]] = {}
        self._start_time = time.monotonic()

    def record(self, decision: EntryDecision) -> None:
        """Record a decision for metrics."""
        self.total_evaluated += 1

        if decision.signal.side.value == "SKIP":
            self.total_skip += 1
            return

        if decision.risk_result.reason == "RISK_GUARD_DISABLED":
            self.total_guard_disabled += 1
            return

        if decision.risk_result.feature_build_latency_ms > 0:
            self.feature_build_latencies.append(decision.risk_result.feature_build_latency_ms)

        rd = decision.risk_result.decision
        if rd == RiskDecision.FEATURE_BUILD_ERROR:
            self.total_feature_errors += 1
        elif rd == RiskDecision.STALE_DATA:
            self.total_stale_data += 1
        elif rd == RiskDecision.FEATURES_UNAVAILABLE:
            self.total_features_unavailable += 1

        if not decision.risk_result.score != decision.risk_result.score:
            self.scores.append(decision.risk_result.score)

        if decision.verdict == RiskGuardVerdict.ALLOW:
            self.total_allowed += 1
        elif decision.verdict == RiskGuardVerdict.OBSERVED_BLOCK:
            self.total_observed_blocks += 1
            self.total_blocked += 1
        elif decision.verdict == RiskGuardVerdict.BLOCK:
            self.total_enforced_blocks += 1
            self.total_blocked += 1

        sym = decision.signal.symbol
        if sym not in self.per_symbol:
            self.per_symbol[sym] = {"evaluated": 0, "allowed": 0, "blocked": 0}
        self.per_symbol[sym]["evaluated"] += 1
        if decision.verdict == RiskGuardVerdict.ALLOW:
            self.per_symbol[sym]["allowed"] += 1
        else:
            self.per_symbol[sym]["blocked"] += 1

        side = decision.signal.side.value
        if side not in self.per_side:
            self.per_side[side] = {"evaluated": 0, "allowed": 0, "blocked": 0}
        self.per_side[side]["evaluated"] += 1
        if decision.verdict == RiskGuardVerdict.ALLOW:
            self.per_side[side]["allowed"] += 1
        else:
            self.per_side[side]["blocked"] += 1

    def summary(self) -> dict[str, Any]:
        """Return a summary of all metrics."""
        import numpy as np

        elapsed = time.monotonic() - self._start_time
        scores_arr = np.array(self.scores) if self.scores else np.array([float("nan")])

        return {
            "total_evaluated": self.total_evaluated,
            "total_allowed": self.total_allowed,
            "total_blocked": self.total_blocked,
            "total_observed_blocks": self.total_observed_blocks,
            "total_enforced_blocks": self.total_enforced_blocks,
            "total_skip": self.total_skip,
            "total_guard_disabled": self.total_guard_disabled,
            "total_feature_errors": self.total_feature_errors,
            "total_stale_data": self.total_stale_data,
            "total_features_unavailable": self.total_features_unavailable,
            "block_rate_pct": (
                (self.total_blocked / max(1, self.total_evaluated - self.total_skip - self.total_guard_disabled)) * 100.0
            ),
            "score_mean": float(np.nanmean(scores_arr)),
            "score_std": float(np.nanstd(scores_arr)),
            "score_p50": float(np.nanpercentile(scores_arr, 50)),
            "score_p90": float(np.nanpercentile(scores_arr, 90)),
            "score_p95": float(np.nanpercentile(scores_arr, 95)),
            "feature_build_latency_mean_ms": float(np.mean(self.feature_build_latencies)) if self.feature_build_latencies else 0.0,
            "feature_build_latency_p95_ms": float(np.percentile(self.feature_build_latencies, 95)) if self.feature_build_latencies else 0.0,
            "elapsed_seconds": elapsed,
            "per_symbol": self.per_symbol,
            "per_side": self.per_side,
        }

    def reset(self) -> None:
        """Reset all metrics."""
        self._reset()


class E4EvidenceRecorder:
    """Append-only JSONL evidence recorder for E4 evaluations.

    Records every E4 evaluation with full decision chain:
    Aegis direction → E4 score → ALLOW/BLOCK → execution result.

    Thread-safe. Uses file locking for concurrent writes.
    """

    def __init__(self, jsonl_path: str | Path | None = None) -> None:
        self._jsonl_path = Path(jsonl_path) if jsonl_path else None
        self._lock = threading.Lock()
        if self._jsonl_path:
            self._jsonl_path.parent.mkdir(parents=True, exist_ok=True)

    def record(self, entry: dict[str, Any]) -> None:
        """Record an E4 evidence entry."""
        entry = dict(entry)
        entry["recorded_at"] = datetime.now(timezone.utc).isoformat()

        logger.info(
            "E4_EVIDENCE %s",
            json.dumps(entry, default=str),
        )

        if self._jsonl_path:
            with self._lock:
                self._append_jsonl(entry)

    def record_precompute(
        self,
        *,
        decision_cycle_id: str,
        decision_at: datetime,
        symbol: str,
        side: str,
        e4_score: float,
        e4_threshold: float,
        e4_decision: str,
        e4_reason: str,
        e4_model_version: str,
        feature_snapshot_hash: str,
        feature_available_at: datetime | None,
        source_feed_lag_ms: dict[str, float] | None,
        snapshot_id: str,
        score_latency_ms: float,
    ) -> None:
        """Record published precompute telemetry without synthetic signal IDs."""
        self.record({
            "event": "e4_precompute_score",
            "decision_cycle_id": decision_cycle_id,
            "decision_at": decision_at.isoformat(),
            "symbol": symbol,
            "side": side,
            "e4_score": e4_score,
            "e4_threshold": e4_threshold,
            "e4_decision": e4_decision,
            "e4_reason": e4_reason,
            "e4_model_version": e4_model_version,
            "feature_snapshot_hash": feature_snapshot_hash,
            "feature_available_at": (
                feature_available_at.isoformat() if feature_available_at else None
            ),
            "source_feed_lag_ms": source_feed_lag_ms,
            "snapshot_id": snapshot_id,
            "score_latency_ms": score_latency_ms,
        })

    def record_evaluation(
        self,
        *,
        signal_id: str,
        decision_id: str,
        decision_cycle_id: str,
        decision_at: datetime,
        symbol: str,
        side: str,
        direction_source: str,
        e4_score: float,
        e4_threshold: float,
        e4_decision: str,
        e4_reason: str,
        e4_model_version: str,
        feature_snapshot_hash: str,
        feature_available_at: datetime | None,
        source_feed_lag_ms: dict[str, float] | None,
        snapshot_id: str,
        cache_age_ms: float,
        python_latency_ms: float,
        ts_guard_latency_ms: float | None = None,
        final_entry_decision: str | None = None,
        approval_status: str | None = None,
        execution_requested: bool = False,
        execution_result: str | None = None,
    ) -> None:
        """Record a complete E4 evaluation with full context."""
        entry = {
            "event": "e4_tail_risk_evaluation",
            "signal_id": signal_id,
            "decision_id": decision_id,
            "decision_cycle_id": decision_cycle_id,
            "decision_at": decision_at.isoformat() if isinstance(decision_at, datetime) else str(decision_at),
            "symbol": symbol,
            "side": side,
            "direction_source": direction_source,
            "e4_score": e4_score,
            "e4_threshold": e4_threshold,
            "e4_decision": e4_decision,
            "e4_reason": e4_reason,
            "e4_model_version": e4_model_version,
            "feature_snapshot_hash": feature_snapshot_hash,
            "feature_available_at": (
                feature_available_at.isoformat()
                if isinstance(feature_available_at, datetime)
                else str(feature_available_at) if feature_available_at else None
            ),
            "source_feed_lag_ms": source_feed_lag_ms,
            "snapshot_id": snapshot_id,
            "cache_age_ms": cache_age_ms,
            "python_latency_ms": python_latency_ms,
            "ts_guard_latency_ms": ts_guard_latency_ms,
            "final_entry_decision": final_entry_decision,
            "approval_status": approval_status,
            "execution_requested": execution_requested,
            "execution_result": execution_result,
        }
        self.record(entry)

    def record_blocked_outcome(
        self,
        *,
        signal_id: str,
        symbol: str,
        side: str,
        blocked_at: datetime,
        actual_outcome: str | None = None,
        actual_pnl_bps: float | None = None,
        blocked_correctly: bool | None = None,
    ) -> None:
        """Record the outcome of a blocked signal for counterfactual analysis."""
        entry = {
            "event": "e4_blocked_outcome",
            "signal_id": signal_id,
            "symbol": symbol,
            "side": side,
            "blocked_at": blocked_at.isoformat() if isinstance(blocked_at, datetime) else str(blocked_at),
            "actual_outcome": actual_outcome,
            "actual_pnl_bps": actual_pnl_bps,
            "blocked_correctly": blocked_correctly,
        }
        self.record(entry)

    def _append_jsonl(self, entry: dict[str, Any]) -> None:
        try:
            with self._jsonl_path.open("a") as f:
                f.write(json.dumps(entry, default=str) + "\n")
        except Exception as exc:
            logger.warning("Failed to write E4 evidence log: %s", exc)
