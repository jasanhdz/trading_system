"""EntryDecisionOrchestrator — combines direction + risk guard into final decision."""

from __future__ import annotations

import logging
from typing import Any

from .domain import (
    Direction,
    EntryDecision,
    RiskDecision,
    RiskGuardConfig,
    RiskGuardResult,
    RiskGuardVerdict,
    Signal,
)
from .direction_provider import DirectionProvider
from .risk_guard import RiskGuard

logger = logging.getLogger(__name__)


class EntryDecisionOrchestrator:
    """Orchestrates direction provision → risk guard → entry decision.

    Flow:
        DirectionProvider.evaluate(symbol)
            → Signal (LONG/SHORT/SKIP)
                → RiskGuard.evaluate(signal)
                    → RiskGuardResult (ALLOW/BLOCK)
                        → EntryDecision (with enforce/observe_only logic)

    If direction is SKIP, no risk guard evaluation is performed.
    If risk guard is disabled, all signals pass through.
    """

    def __init__(
        self,
        direction_provider: DirectionProvider,
        risk_guard: RiskGuard | None,
        config: RiskGuardConfig,
    ) -> None:
        self._direction_provider = direction_provider
        self._risk_guard = risk_guard
        self._config = config

    def evaluate(self, symbol: str, context: dict[str, Any] | None = None) -> EntryDecision:
        """Evaluate a full entry decision for a symbol.

        1. Get direction from provider
        2. If SKIP, return immediately with ALLOW verdict (no trade anyway)
        3. If risk guard disabled, return ALLOW
        4. Evaluate risk guard
        5. Apply enforce/observe_only logic
        """
        signal = self._direction_provider.evaluate(symbol, context)

        if signal.side == Direction.SKIP:
            return EntryDecision(
                signal=signal,
                risk_result=RiskGuardResult(
                    decision=RiskDecision.ALLOW,
                    score=0.0,
                    threshold=0.0,
                    model_version="NONE",
                    feature_snapshot_hash="",
                    reason="DIRECTION_SKIP",
                ),
                verdict=RiskGuardVerdict.ALLOW,
                enforced=False,
                observe_only=False,
            )

        if self._risk_guard is None or not self._config.enabled:
            return EntryDecision(
                signal=signal,
                risk_result=RiskGuardResult(
                    decision=RiskDecision.ALLOW,
                    score=0.0,
                    threshold=self._config.tail_risk_threshold,
                    model_version="NONE",
                    feature_snapshot_hash="",
                    reason="RISK_GUARD_DISABLED",
                ),
                verdict=RiskGuardVerdict.ALLOW,
                enforced=False,
                observe_only=False,
            )

        risk_result = self._risk_guard.evaluate(signal, context)

        _feature_error_states = {
            RiskDecision.FEATURES_UNAVAILABLE,
            RiskDecision.STALE_DATA,
            RiskDecision.NON_CAUSAL_DATA,
            RiskDecision.FEATURE_BUILD_ERROR,
        }

        if risk_result.decision == RiskDecision.ALLOW:
            verdict = RiskGuardVerdict.ALLOW
        elif risk_result.decision in _feature_error_states:
            verdict = RiskGuardVerdict.ALLOW
        elif self._config.enforce:
            verdict = RiskGuardVerdict.BLOCK
        else:
            verdict = RiskGuardVerdict.OBSERVED_BLOCK

        decision = EntryDecision(
            signal=signal,
            risk_result=risk_result,
            verdict=verdict,
            enforced=self._config.enforce,
            observe_only=self._config.observe_only,
        )

        logger.info(
            "EntryDecision[%s] %s/%s: score=%.6f verdict=%s enforced=%s",
            signal.signal_id,
            signal.symbol,
            signal.side.value,
            risk_result.score,
            verdict.value,
            decision.enforced,
        )

        return decision
