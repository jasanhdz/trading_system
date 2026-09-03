"""PositionManager contract — defines the handoff from EntryDecision to execution.

The PositionManager is the final gatekeeper:

    EntryDecision with verdict ALLOW          → may execute
    EntryDecision with verdict OBSERVED_BLOCK  → executes (observe_only logs but doesn't block)
    EntryDecision with verdict BLOCK           → NUNCA reaches executor

    SKIP direction                            → NUNCA reaches PositionManager at all

This module defines the interface contract. The actual PositionManager
implementation lives in the existing codebase and is NOT modified by
this package.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any

from .domain import Direction, EntryDecision, RiskGuardVerdict

logger = logging.getLogger(__name__)


class PositionManagerContract(ABC):
    """Abstract contract for the position manager handoff.

    Rules:
        1. SKIP direction → can_execute=False (should never reach here)
        2. verdict=ALLOW → can_execute=True
        3. verdict=OBSERVED_BLOCK → can_execute=True (observe_only doesn't block)
        4. verdict=BLOCK → can_execute=False
    """

    @abstractmethod
    def can_execute(self, decision: EntryDecision) -> bool:
        """Check if an entry decision can be executed.

        Returns True for ALLOW and OBSERVED_BLOCK.
        Returns False for BLOCK and SKIP.
        """
        ...

    @abstractmethod
    def execute(self, decision: EntryDecision) -> PositionManagerResult:
        """Execute an entry decision.

        Raises ValueError if not can_execute().
        """
        ...


class PositionManagerResult:
    """Result of a position manager execution attempt."""

    def __init__(
        self,
        accepted: bool,
        reason: str,
        order_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self.accepted = accepted
        self.reason = reason
        self.order_id = order_id
        self.metadata = metadata or {}

    def to_dict(self) -> dict[str, Any]:
        return {
            "accepted": self.accepted,
            "reason": self.reason,
            "order_id": self.order_id,
            "metadata": self.metadata,
        }


class AllowOnlyPositionManager(PositionManagerContract):
    """Reference implementation with correct semantics.

    Flow:
        SKIP          → NUNCA ejecutar (should not reach here)
        LONG/SHORT + ALLOW           → ejecutar
        LONG/SHORT + OBSERVED_BLOCK  → ejecutar (observe_only)
        LONG/SHORT + BLOCK           → NO ejecutar
    """

    def can_execute(self, decision: EntryDecision) -> bool:
        # SKIP direction should never reach PositionManager
        if decision.signal.side == Direction.SKIP:
            return False

        # ALLOW and OBSERVED_BLOCK are both executable
        # OBSERVED_BLOCK means "E4 flagged it but we're observing — trade proceeds"
        if decision.verdict in (RiskGuardVerdict.ALLOW, RiskGuardVerdict.OBSERVED_BLOCK):
            return True

        # BLOCK is not executable
        return False

    def execute(self, decision: EntryDecision) -> PositionManagerResult:
        if not self.can_execute(decision):
            logger.warning(
                "PositionManager: rejecting %s decision for %s/%s (verdict=%s, side=%s)",
                decision.risk_result.decision.value,
                decision.signal.symbol,
                decision.signal.side.value,
                decision.verdict.value,
                decision.signal.side.value,
            )
            return PositionManagerResult(
                accepted=False,
                reason=f"VERDICT_NOT_ALLOW:{decision.verdict.value}:SIDE:{decision.signal.side.value}",
                metadata={
                    "signal_id": decision.signal.signal_id,
                    "tail_risk_score": decision.risk_result.score,
                },
            )

        logger.info(
            "PositionManager: accepting %s for %s/%s (verdict=%s, score=%.6f)",
            decision.risk_result.decision.value,
            decision.signal.symbol,
            decision.signal.side.value,
            decision.verdict.value,
            decision.risk_result.score,
        )
        return PositionManagerResult(
            accepted=True,
            reason=f"ALLOW:{decision.verdict.value}",
            metadata={
                "signal_id": decision.signal.signal_id,
                "tail_risk_score": decision.risk_result.score,
            },
        )
