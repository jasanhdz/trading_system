"""PositionManager contract — defines the handoff from EntryDecision to execution.

The PositionManager is the final gatekeeper:
- EntryDecision.ALLOW → may proceed to executor/PositionManager
- EntryDecision.BLOCK → NUNCA reaches executor/PositionManager

This module defines the interface contract. The actual PositionManager
implementation lives in the existing codebase and is NOT modified by
this package.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any

from .domain import EntryDecision, RiskGuardVerdict

logger = logging.getLogger(__name__)


class PositionManagerContract(ABC):
    """Abstract contract for the position manager handoff.

   任何实现了此接口的 PositionManager 必须遵守以下规则：
    1. 只接受 verdict == ALLOW 的 EntryDecision
    2. 对 verdict == BLOCK 的 EntryDecision 拒绝执行
    3. 对 verdict == OBSERVED_BLOCK 的 EntryDecision 记录但不执行
    """

    @abstractmethod
    def can_execute(self, decision: EntryDecision) -> bool:
        """Check if an entry decision can be executed.

        Returns True only if verdict == ALLOW.
        Returns False for BLOCK and OBSERVED_BLOCK.
        """
        ...

    @abstractmethod
    def execute(self, decision: EntryDecision) -> PositionManagerResult:
        """Execute an entry decision.

        Raises ValueError if decision.verdict != ALLOW.
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
    """Reference implementation that only allows ALLOW decisions.

    This demonstrates the correct contract behavior:
    - ALLOW → accepted
    - BLOCK → rejected
    - OBSERVED_BLOCK → rejected (logged but not executed)
    """

    def can_execute(self, decision: EntryDecision) -> bool:
        return decision.verdict == RiskGuardVerdict.ALLOW

    def execute(self, decision: EntryDecision) -> PositionManagerResult:
        if not self.can_execute(decision):
            logger.warning(
                "PositionManager: rejecting %s decision for %s/%s (verdict=%s)",
                decision.risk_result.decision.value,
                decision.signal.symbol,
                decision.signal.side.value,
                decision.verdict.value,
            )
            return PositionManagerResult(
                accepted=False,
                reason=f"VERDICT_NOT_ALLOW:{decision.verdict.value}",
                metadata={
                    "signal_id": decision.signal.signal_id,
                    "tail_risk_score": decision.risk_result.score,
                },
            )

        logger.info(
            "PositionManager: accepting %s for %s/%s (score=%.6f)",
            decision.risk_result.decision.value,
            decision.signal.symbol,
            decision.signal.side.value,
            decision.risk_result.score,
        )
        return PositionManagerResult(
            accepted=True,
            reason="ALLOW",
            metadata={
                "signal_id": decision.signal.signal_id,
                "tail_risk_score": decision.risk_result.score,
            },
        )
