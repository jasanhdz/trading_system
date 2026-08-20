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


class E4PrecomputeService:
    """Background service that precomputes E4 scores every 5 minutes.

    Thread-safe. Uses a lock for atomic cache updates.
    On cycle failure, the previous cache remains valid (no partial updates).
    """

    def __init__(self, config: RiskGuardConfig) -> None:
        self._config = config
        self._guard: E4TailRiskGuard | None = None
        self._lock = threading.Lock()
        self._cache: dict[str, PrecomputedScore] = {}  # key: "decision_at|symbol|side"
        self._last_cycle: PrecomputeCycleResult | None = None
        self._running = False
        self._thread: threading.Thread | None = None
        self._cycle_count = 0

    def initialize(self) -> None:
        """Load frozen artifacts and run initial computation."""
        logger.info("E4 precompute service initializing...")

        self._guard = E4TailRiskGuard(self._config)
        self._guard.load()

        if not self._guard.is_available():
            raise RuntimeError("E4 model failed to load")

        self._verify_frozen_invariants()

        self._run_cycle()

        logger.info(
            "E4 precompute service initialized",
            extra={
                "model_version": self._guard.version(),
                "threshold": FROZEN_TAIL_RISK_THRESHOLD,
                "universe": list(FROZEN_E4_UNIVERSE),
                "timeframes": FROZEN_E4_TIMEFROZEN,
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
        """Background loop that runs precompute cycles."""
        while self._running:
            try:
                self._run_cycle()
            except Exception:
                logger.exception("E4 precompute cycle failed")
            time.sleep(CADENCE_SECONDS)

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
            snapshot = fetch_snapshot(decision_at)
            feature_start = time.monotonic()

            panel_by_symbol: dict[str, pd.DataFrame] = {}
            for symbol in sorted(FROZEN_E4_UNIVERSE):
                panel_by_symbol[symbol] = snapshot.candles_by_symbol[symbol]

            new_cache: dict[str, PrecomputedScore] = {}
            feature_build_ms = 0.0
            all_features_built = True

            for symbol in sorted(FROZEN_E4_UNIVERSE):
                for side in ["LONG", "SHORT"]:
                    try:
                        t0 = time.monotonic()
                        feature_row = self._guard._feature_bridge.from_market_candles(
                            panel_by_symbol, symbol, side, decision_at
                        )
                        feature_build_ms = (time.monotonic() - t0) * 1000

                        t1 = time.monotonic()
                        score = self._guard._score_tail_risk(
                            pd.DataFrame([feature_row.features])
                        )
                        score_ms = (time.monotonic() - t1) * 1000

                        risk_decision = (
                            RiskDecision.ALLOW.value
                            if score < FROZEN_TAIL_RISK_THRESHOLD
                            else RiskDecision.BLOCK.value
                        )
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

                    except Exception as e:
                        all_features_built = False
                        error_score = PrecomputedScore(
                            symbol=symbol,
                            side=side,
                            decision_at=decision_at,
                            score=0.0,
                            threshold=FROZEN_TAIL_RISK_THRESHOLD,
                            risk_decision=RiskDecision.FEATURE_BUILD_ERROR.value,
                            reason=f"FEATURE_BUILD_ERROR: {e}",
                            model_version=self._guard.version(),
                            feature_snapshot_hash="",
                            feature_available_at=None,
                            source_feed_lag_ms=None,
                            computed_at=datetime.now(timezone.utc),
                            snapshot_id=snapshot.snapshot_id,
                        )
                        cache_key = _cache_key(decision_at, symbol, side)
                        new_cache[cache_key] = error_score

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
                error=None if all_features_built else "PARTIAL_BUILD_FAILURE",
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
                    "feature_build_latency_ms": feature_build_ms,
                    "score_count": len(new_cache),
                    "snapshot_id": snapshot.snapshot_id,
                    "error": result.error,
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
