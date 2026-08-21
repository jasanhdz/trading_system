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
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import pandas as pd

from .domain import (
    FROZEN_TAIL_RISK_THRESHOLD,
    RiskDecision,
    RiskGuardConfig,
)
from .e4_tail_risk_guard import E4TailRiskGuard
from .feature_bridge import FROZEN_E4_UNIVERSE, FROZEN_E4_TIMEFROZEN
from .market_snapshot import MarketSnapshot, fetch_snapshot
from .observability import E4EvidenceRecorder

logger = logging.getLogger(__name__)

CADENCE_SECONDS = 300  # 5 minutes
SOURCE_FEED_LAG_TOLERANCE_S = 60


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
        now_fn: Callable[[], datetime] | None = None,
        sleep_fn: Callable[[float], None] | None = None,
        snapshot_provider: Callable[[datetime], MarketSnapshot] | None = None,
    ) -> None:
        self._config = config
        self._guard: E4TailRiskGuard | None = None
        self._lock = threading.Lock()
        self._cycle_lock = threading.Lock()
        self._cache: dict[str, PrecomputedScore] = {}  # key: "decision_at|symbol|side"
        self._last_successful_cycle: PrecomputeCycleResult | None = None
        self._last_attempt: PrecomputeCycleResult | None = None
        self._running = False
        self._thread: threading.Thread | None = None
        self._cycle_count = 0
        self._evidence_recorder = evidence_recorder or E4EvidenceRecorder()
        self._now_fn = now_fn or (lambda: datetime.now(timezone.utc))
        self._sleep_fn = sleep_fn or time.sleep
        self._snapshot_provider = snapshot_provider or fetch_snapshot

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
        result = self._run_cycle(self.target_decision_at())

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
        with self._lock:
            if self._running:
                return
            self._running = True
            self._thread = threading.Thread(
                target=self._background_loop, daemon=True, name="e4-precompute"
            )
            thread = self._thread
        thread.start()
        logger.info("E4 precompute background loop started")

    def stop_background(self) -> None:
        """Stop the background precompute loop."""
        with self._lock:
            self._running = False
            thread = self._thread
        if thread:
            thread.join(timeout=10)
            with self._lock:
                if self._thread is thread:
                    self._thread = None
        logger.info("E4 precompute background loop stopped")

    def _background_loop(self) -> None:
        """Background loop aligned to UTC 5-minute boundaries.

        Prevents duplicate cycles per T by tracking last completed decision_at.
        Sleeps until the next 5-minute boundary after cycle completion.
        """
        while True:
            with self._lock:
                if not self._running:
                    return
            target = self.target_decision_at()
            try:
                if not self.has_published_cycle(target):
                    self._run_cycle(target)
            except Exception:
                logger.exception("E4 precompute cycle failed")
            self._sleep_fn(self.seconds_until_next_cycle())

    def target_decision_at(self, now: datetime | None = None) -> datetime:
        now = now or self._now_fn()
        if now.tzinfo is None:
            now = now.replace(tzinfo=timezone.utc)
        else:
            now = now.astimezone(timezone.utc)
        return now.replace(minute=(now.minute // 5) * 5, second=0, microsecond=0)

    def seconds_until_next_cycle(self, now: datetime | None = None) -> float:
        now = now or self._now_fn()
        target = self.target_decision_at(now)
        next_cycle = target.timestamp() + CADENCE_SECONDS
        return max(0.001, next_cycle - now.timestamp())

    def has_published_cycle(self, decision_at: datetime) -> bool:
        with self._lock:
            return (
                self._last_successful_cycle is not None
                and self._last_successful_cycle.decision_at == decision_at
            )

    def _run_cycle(self, decision_at: datetime | None = None) -> PrecomputeCycleResult:
        """Execute one precompute cycle. Thread-safe, atomic publish."""
        decision_at = decision_at or self.target_decision_at()
        with self._cycle_lock:
            with self._lock:
                if (
                    self._last_successful_cycle is not None
                    and self._last_successful_cycle.decision_at == decision_at
                ):
                    return self._last_successful_cycle
                self._cycle_count += 1
                cycle_number = self._cycle_count
            cycle_started = self._now_fn()
            return self._execute_cycle(decision_at, cycle_started, cycle_number)

    def _execute_cycle(
        self,
        decision_at: datetime,
        cycle_started: datetime,
        cycle_number: int,
    ) -> PrecomputeCycleResult:

        logger.info(
            "E4 precompute cycle %d starting",
            cycle_number,
            extra={"decision_at": decision_at.isoformat()},
        )

        try:
            snapshot_start = time.monotonic()
            snapshot = self._snapshot_provider(decision_at)
            snapshot_ms = (time.monotonic() - snapshot_start) * 1000

            feature_start = time.monotonic()

            # Build ALL 22 feature rows in ONE panel pass (defect #3 fix)
            feature_rows = self._guard.bridge.from_market_candles_batch(
                snapshot.candles_by_symbol, decision_at
            )
            feature_build_ms = (time.monotonic() - feature_start) * 1000

            new_cache: dict[str, PrecomputedScore] = {}
            score_start = time.monotonic()
            score_latencies: dict[str, float] = {}

            for (symbol, side), feature_row in feature_rows.items():
                score_one_start = time.monotonic()
                score = self._guard.score(
                    pd.DataFrame([feature_row.features])
                )
                score_one_ms = (time.monotonic() - score_one_start) * 1000

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
                    computed_at=self._now_fn(),
                    snapshot_id=snapshot.snapshot_id,
                )
                cache_key = _cache_key(decision_at, symbol, side)
                new_cache[cache_key] = computed
                score_latencies[cache_key] = score_one_ms

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

            cycle_completed = self._now_fn()
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
                self._last_successful_cycle = result
                self._last_attempt = result

            for cache_key, computed in new_cache.items():
                self._evidence_recorder.record_precompute(
                    decision_cycle_id=snapshot.snapshot_id,
                    decision_at=decision_at,
                    symbol=computed.symbol,
                    side=computed.side,
                    e4_score=computed.score,
                    e4_threshold=computed.threshold,
                    e4_decision=computed.risk_decision,
                    e4_reason=computed.reason,
                    e4_model_version=computed.model_version,
                    feature_snapshot_hash=computed.feature_snapshot_hash,
                    feature_available_at=computed.feature_available_at,
                    source_feed_lag_ms=computed.source_feed_lag_ms,
                    snapshot_id=computed.snapshot_id,
                    score_latency_ms=score_latencies[cache_key],
                )

            logger.info(
                "E4 precompute cycle %d completed",
                cycle_number,
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
            cycle_completed = self._now_fn()
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
                self._last_attempt = result

            logger.error(
                "E4 precompute cycle %d failed: %s",
                cycle_number,
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
            return self._last_successful_cycle

    @property
    def last_attempt(self) -> PrecomputeCycleResult | None:
        with self._lock:
            return self._last_attempt

    @property
    def cycle_count(self) -> int:
        return self._cycle_count

    def health(self) -> dict[str, Any]:
        """Return service health status."""
        with self._lock:
            last = self._last_successful_cycle
            attempt = self._last_attempt
            cache_entries = len(self._cache)
            running = self._running
        return {
            "available": self._guard is not None and self._guard.is_available(),
            "cycle_count": self._cycle_count,
            "last_decision_at": last.decision_at.isoformat() if last else None,
            "last_cycle_at": last.cycle_completed_at.isoformat() if last else None,
            "last_cycle_latency_ms": last.cycle_latency_ms if last else None,
            "last_snapshot_fetch_ms": last.snapshot_fetch_ms if last else None,
            "last_feature_build_ms": last.feature_build_latency_ms if last else None,
            "last_score_ms": last.score_latency_ms if last else None,
            "last_cycle_error": last.error if last else None,
            "last_attempt_at": attempt.cycle_completed_at.isoformat() if attempt else None,
            "last_attempt_error": attempt.error if attempt else None,
            "cache_entries": cache_entries,
            "running": running,
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
