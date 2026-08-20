"""E4 Feature Precomputation Service.

Runs on a 5-minute cadence, precomputing E4 tail risk scores for all
11 symbols × 2 sides. Results are cached in memory and served via
the FastAPI endpoint.

Latency budget per cycle:
  - Candle fetch: ~2-3s (sequential REST, 11 symbols)
  - Feature build: ~100-200ms (FeatureBridge)
  - Model scoring: ~5ms (22 scores)
  - Total: ~3-4s every 5 minutes

Critical path per signal:
  - Cache lookup: <1ms
  - HTTP overhead: ~10ms
  - Total: <20ms

NO autonomous order placement. NO trade execution.
This module precomputes risk scores for the approval gate.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any

import pandas as pd

from .domain import (
    FROZEN_TAIL_RISK_THRESHOLD,
    RiskDecision,
    RiskGuardConfig,
)
from .e4_tail_risk_guard import E4TailRiskGuard
from .feature_bridge import FROZEN_E4_UNIVERSE, FROZEN_E4_TIMEFROZEN, FeatureBridge, FeatureRow
from .market_snapshot import MarketSnapshot, fetch_snapshot
from .observability import E4EvidenceRecorder

logger = logging.getLogger(__name__)

CADENCE_SECONDS = 300  # 5 minutes
STALENESS_TOLERANCE_S = 60  # allow 1 minute of staleness before rejecting


@dataclass(frozen=True)
class PrecomputedScore:
    """A precomputed E4 score for a specific (symbol, side, decision_at)."""
    symbol: str
    side: str
    decision_at: datetime
    score: float
    threshold: float
    risk_decision: str  # ALLOW or BLOCK
    reason: str
    model_version: str
    feature_snapshot_hash: str
    feature_available_at: datetime | None
    source_feed_lag_ms: dict[str, float] | None
    computed_at: datetime
    snapshot_id: str


@dataclass(frozen=True)
class PrecomputeCycleResult:
    """Result of a single precompute cycle."""
    decision_at: datetime
    snapshot_id: str
    snapshot_hash: str
    cycle_started_at: datetime
    cycle_completed_at: datetime
    cycle_latency_ms: float
    feature_build_latency_ms: float
    score_count: int
    scores: dict[str, dict[str, PrecomputedScore]]  # [symbol][side]
    error: str | None = None
    snapshot_fetch_ms: float = 0.0
    score_latency_ms: float = 0.0


class E4PrecomputeService:
    """Background service that precomputes E4 scores every 5 minutes.

    Thread-safe. Uses a lock for atomic cache updates.
    On cycle failure, the previous cache remains valid (no partial updates).
    """

    def __init__(
        self,
        config: RiskGuardConfig,
        evidence_recorder: E4EvidenceRecorder | None = None,
    ) -> None:
        self._config = config
        self._guard: E4TailRiskGuard | None = None
        self._lock = threading.Lock()
        self._cache: dict[str, PrecomputedScore] = {}  # key: "decision_at|symbol|side"
        self._last_cycle: PrecomputeCycleResult | None = None
        self._running = False
        self._thread: threading.Thread | None = None
        self._cycle_count = 0
        self._evidence_recorder = evidence_recorder or E4EvidenceRecorder()

    def initialize(self) -> None:
        """Load frozen artifacts and run initial computation.

        Defect #10: First cycle MUST succeed with 22/22 valid scores.
        Service will not start if initial precompute fails.
        """
        logger.info("E4 precompute service initializing...")

        self._guard = E4TailRiskGuard(self._config)
        self._guard.load()

        if not self._guard.is_available():
            raise RuntimeError("E4 model failed to load")

        self._verify_frozen_invariants()

        # Defect #10: First cycle must succeed with 22/22 valid scores
        result = self._run_cycle()

        if result.error is not None:
            raise RuntimeError(
                f"E4 precompute initialization failed: {result.error}. "
                f"Score count: {result.score_count}/22. "
                f"Service will not start with partial data."
            )

        EXPECTED_SCORE_COUNT = len(FROZEN_E4_UNIVERSE) * 2
        if result.score_count != EXPECTED_SCORE_COUNT:
            raise RuntimeError(
                f"E4 precompute initialization failed: expected {EXPECTED_SCORE_COUNT} scores, "
                f"got {result.score_count}. Service will not start with partial data."
            )

        logger.info(
            "E4 precompute service initialized",
            extra={
                "model_version": self._guard.version(),
                "threshold": FROZEN_TAIL_RISK_THRESHOLD,
                "universe": list(FROZEN_E4_UNIVERSE),
                "timeframes": FROZEN_E4_TIMEFROZEN,
                "initial_score_count": result.score_count,
                "cycle_latency_ms": result.cycle_latency_ms,
            },
        )

    def _verify_frozen_invariants(self) -> None:
        """Verify all frozen E4 V1 invariants."""
        if self._config.tail_risk_threshold != FROZEN_TAIL_RISK_THRESHOLD:
            raise ValueError(
                f"THRESHOLD_MISMATCH: expected {FROZEN_TAIL_RISK_THRESHOLD}, "
                f"got {self._config.tail_risk_threshold}"
            )
        if self._config.fail_closed is not True:
            raise ValueError("fail_closed must be True for E4 V1")
        if self._config.models_joblib_path:
            expected_hash = self._config.models_joblib_sha256
            if expected_hash:
                actual_hash = E4TailRiskGuard._sha256_file(
                    Path(self._config.models_joblib_path)
                )
                if actual_hash != expected_hash:
                    raise ValueError(
                        f"MODEL_HASH_MISMATCH: expected {expected_hash}, got {actual_hash}"
                    )

    def start_background(self) -> None:
        """Start the background precompute loop."""
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(
            target=self._background_loop, daemon=True, name="e4-precompute"
        )
        self._thread.start()
        logger.info("E4 precompute background loop started")

    def stop_background(self) -> None:
        """Stop the background precompute loop."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=10)
            self._thread = None
        logger.info("E4 precompute background loop stopped")

    def _background_loop(self) -> None:
        """Background loop aligned to UTC 5-minute boundaries.

        Prevents duplicate cycles per T by tracking last completed decision_at.
        Sleeps until the next 5-minute boundary after cycle completion.
        """
        last_completed_decision_at: datetime | None = None

        while self._running:
            try:
                result = self._run_cycle()

                # Prevent duplicate cycles for same decision_at
                if result.decision_at == last_completed_decision_at:
                    logger.debug(
                        "Skipping duplicate cycle for decision_at=%s",
                        result.decision_at.isoformat(),
                    )
                else:
                    last_completed_decision_at = result.decision_at

            except Exception:
                logger.exception("E4 precompute cycle failed")

            # Sleep until next 5-minute boundary
            now = datetime.now(timezone.utc)
            seconds_since_boundary = (now.minute % 5) * 60 + now.second
            sleep_seconds = CADENCE_SECONDS - seconds_since_boundary
            if sleep_seconds <= 0:
                sleep_seconds = CADENCE_SECONDS
            time.sleep(sleep_seconds)

    def _run_cycle(self) -> PrecomputeCycleResult:
        """Execute one precompute cycle. Thread-safe, atomic publish."""
        cycle_started = datetime.now(timezone.utc)
        self._cycle_count += 1

        decision_at = cycle_started.replace(
            minute=(cycle_started.minute // 5) * 5, second=0, microsecond=0
        )

        logger.info(
            "E4 precompute cycle %d starting",
            self._cycle_count,
            extra={"decision_at": decision_at.isoformat()},
        )

        try:
            snapshot_start = time.monotonic()
            snapshot = fetch_snapshot(decision_at)
            snapshot_ms = (time.monotonic() - snapshot_start) * 1000

            feature_start = time.monotonic()

            # Build ALL 22 feature rows in ONE panel pass (defect #3 fix)
            feature_rows = self._guard.bridge.from_market_candles_batch(
                snapshot.candles_by_symbol, decision_at
            )
            feature_build_ms = (time.monotonic() - feature_start) * 1000

            new_cache: dict[str, PrecomputedScore] = {}
            score_start = time.monotonic()

            for (symbol, side), feature_row in feature_rows.items():
                score = self._guard.score(
                    pd.DataFrame([feature_row.features])
                )

                # Defect #11: Validate score range
                if not (0.0 <= score <= 1.0):
                    raise ValueError(
                        f"INVALID_SCORE: {symbol}/{side} score={score} outside [0, 1]"
                    )

                # Defect #11: Validate threshold is exact frozen value
                if FROZEN_TAIL_RISK_THRESHOLD != 0.4522452210875323:
                    raise ValueError(
                        f"THRESHOLD_DRIFT: expected 0.4522452210875323, "
                        f"got {FROZEN_TAIL_RISK_THRESHOLD}"
                    )

                # Defect #11: Decision must match threshold comparison
                expected_decision = (
                    RiskDecision.ALLOW.value
                    if score < FROZEN_TAIL_RISK_THRESHOLD
                    else RiskDecision.BLOCK.value
                )
                risk_decision = expected_decision
                reason = (
                    f"TAIL_RISK_SCORE={score:.6f} < THRESHOLD={FROZEN_TAIL_RISK_THRESHOLD:.6f}"
                    if risk_decision == RiskDecision.ALLOW.value
                    else f"TAIL_RISK_SCORE={score:.6f} >= THRESHOLD={FROZEN_TAIL_RISK_THRESHOLD:.6f}"
                )

                computed = PrecomputedScore(
                    symbol=symbol,
                    side=side,
                    decision_at=decision_at,
                    score=score,
                    threshold=FROZEN_TAIL_RISK_THRESHOLD,
                    risk_decision=risk_decision,
                    reason=reason,
                    model_version=self._guard.version(),
                    feature_snapshot_hash=feature_row.feature_hash,
                    feature_available_at=feature_row.max_available_at,
                    source_feed_lag_ms=feature_row.source_feed_lag_ms,
                    computed_at=datetime.now(timezone.utc),
                    snapshot_id=snapshot.snapshot_id,
                )
                cache_key = _cache_key(decision_at, symbol, side)
                new_cache[cache_key] = computed

                # Defect #16: Wire evidence recorder
                self._evidence_recorder.record_evaluation(
                    signal_id=f"precompute_{decision_at.strftime('%Y%m%dT%H%M%SZ')}_{symbol}_{side}",
                    decision_id=cache_key,
                    decision_cycle_id=snapshot.snapshot_id,
                    decision_at=decision_at,
                    symbol=symbol,
                    side=side,
                    direction_source="precompute",
                    e4_score=score,
                    e4_threshold=FROZEN_TAIL_RISK_THRESHOLD,
                    e4_decision=risk_decision,
                    e4_reason=reason,
                    e4_model_version=self._guard.version(),
                    feature_snapshot_hash=feature_row.feature_hash,
                    feature_available_at=feature_row.max_available_at,
                    source_feed_lag_ms=feature_row.source_feed_lag_ms,
                    snapshot_id=snapshot.snapshot_id,
                    cache_age_ms=0.0,
                    python_latency_ms=score_ms,
                )

            score_ms = (time.monotonic() - score_start) * 1000

            # Defect #4: Atomic publish - require exactly 22/22 valid scores
            EXPECTED_SCORE_COUNT = len(FROZEN_E4_UNIVERSE) * 2  # 11 symbols × 2 sides
            if len(new_cache) != EXPECTED_SCORE_COUNT:
                raise RuntimeError(
                    f"ATOMIC_PUBLISH_FAILURE: expected {EXPECTED_SCORE_COUNT} scores, "
                    f"got {len(new_cache)}. Refusing to publish partial results."
                )

            # Validate all scores are valid (not NaN, not error scores)
            for key, score in new_cache.items():
                if score.risk_decision == RiskDecision.FEATURE_BUILD_ERROR.value:
                    raise RuntimeError(
                        f"ATOMIC_PUBLISH_FAILURE: score {key} has FEATURE_BUILD_ERROR. "
                        f"Refusing to publish partial results."
                    )
                if not (0.0 <= score.score <= 1.0):
                    raise RuntimeError(
                        f"ATOMIC_PUBLISH_FAILURE: score {key} has invalid score {score.score}. "
                        f"Refusing to publish partial results."
                    )

            cycle_completed = datetime.now(timezone.utc)
            cycle_latency_ms = (cycle_completed - cycle_started).total_seconds() * 1000

            result = PrecomputeCycleResult(
                decision_at=decision_at,
                snapshot_id=snapshot.snapshot_id,
                snapshot_hash=snapshot.snapshot_hash,
                cycle_started_at=cycle_started,
                cycle_completed_at=cycle_completed,
                cycle_latency_ms=cycle_latency_ms,
                feature_build_latency_ms=feature_build_ms,
                score_count=len(new_cache),
                scores=_organize_scores(new_cache),
                error=None,
                snapshot_fetch_ms=snapshot_ms,
                score_latency_ms=score_ms,
            )

            with self._lock:
                self._cache = new_cache
                self._last_cycle = result

            logger.info(
                "E4 precompute cycle %d completed",
                self._cycle_count,
                extra={
                    "decision_at": decision_at.isoformat(),
                    "cycle_latency_ms": cycle_latency_ms,
                    "snapshot_fetch_ms": snapshot_ms,
                    "feature_build_latency_ms": feature_build_ms,
                    "score_latency_ms": score_ms,
                    "score_count": len(new_cache),
                    "snapshot_id": snapshot.snapshot_id,
                },
            )

            return result

        except Exception as e:
            cycle_completed = datetime.now(timezone.utc)
            cycle_latency_ms = (cycle_completed - cycle_started).total_seconds() * 1000

            result = PrecomputeCycleResult(
                decision_at=decision_at,
                snapshot_id="",
                snapshot_hash="",
                cycle_started_at=cycle_started,
                cycle_completed_at=cycle_completed,
                cycle_latency_ms=cycle_latency_ms,
                feature_build_latency_ms=0.0,
                score_count=0,
                scores={},
                error=str(e),
            )

            with self._lock:
                self._last_cycle = result

            logger.error(
                "E4 precompute cycle %d failed: %s",
                self._cycle_count,
                e,
                extra={
                    "decision_at": decision_at.isoformat(),
                    "cycle_latency_ms": cycle_latency_ms,
                },
            )

            return result

    def lookup(
        self,
        symbol: str,
        side: str,
        decision_at: datetime,
    ) -> PrecomputedScore | None:
        """Look up a precomputed score. Returns None if not available.

        Enforces decision_at identity: the served score must correspond
        exactly to the requested decision_at.
        """
        cache_key = _cache_key(decision_at, symbol, side)
        with self._lock:
            return self._cache.get(cache_key)

    @property
    def last_cycle(self) -> PrecomputeCycleResult | None:
        with self._lock:
            return self._last_cycle

    @property
    def cycle_count(self) -> int:
        return self._cycle_count

    def health(self) -> dict[str, Any]:
        """Return service health status."""
        with self._lock:
            last = self._last_cycle
        return {
            "available": self._guard is not None and self._guard.is_available(),
            "cycle_count": self._cycle_count,
            "last_cycle_at": last.cycle_completed_at.isoformat() if last else None,
            "last_cycle_latency_ms": last.cycle_latency_ms if last else None,
            "last_cycle_error": last.error if last else None,
            "cache_entries": len(self._cache),
            "running": self._running,
        }


def _cache_key(decision_at: datetime, symbol: str, side: str) -> str:
    return f"{decision_at.isoformat()}|{symbol}|{side}"


def _organize_scores(
    cache: dict[str, PrecomputedScore],
) -> dict[str, dict[str, PrecomputedScore]]:
    """Organize flat cache into nested dict[symbol][side]."""
    result: dict[str, dict[str, PrecomputedScore]] = {}
    for key, score in cache.items():
        if score.symbol not in result:
            result[score.symbol] = {}
        result[score.symbol][score.side] = score
    return result
