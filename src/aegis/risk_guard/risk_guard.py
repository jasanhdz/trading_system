"""RiskGuard interface — abstracts risk evaluation for trading signals."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from .domain import RiskGuardResult, Signal


class RiskGuard(ABC):
    """Abstract base for risk guard evaluators.

    A RiskGuard evaluates a signal and returns ALLOW or BLOCK.
    It never changes the side or creates new signals.
    """

    @abstractmethod
    def evaluate(self, signal: Signal, context: dict[str, Any] | None = None) -> RiskGuardResult:
        """Evaluate risk for a signal.

        Returns a RiskGuardResult with decision=ALLOW or BLOCK.
        If the guard cannot evaluate (missing data, error), it must:
        - If fail_closed=True: return BLOCK
        - If fail_closed=False: return ALLOW (with reason indicating failure)
        """
        ...

    @abstractmethod
    def name(self) -> str:
        """Human-readable name of this risk guard."""
        ...

    @abstractmethod
    def version(self) -> str:
        """Version identifier for this risk guard (frozen model version)."""
        ...

    @abstractmethod
    def is_available(self) -> bool:
        """Check if the guard's model artifacts are loaded and ready."""
        ...
