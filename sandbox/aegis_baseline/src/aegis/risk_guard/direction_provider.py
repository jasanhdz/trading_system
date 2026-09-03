"""DirectionProvider interface — abstracts the source of directional signals."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from .domain import Direction, Signal


class DirectionProvider(ABC):
    """Abstract base for directional signal providers.

    Implementations produce Direction.LONG, Direction.SHORT, or Direction.SKIP.
    The RiskGuard never modifies the side — it only evaluates risk.
    """

    @abstractmethod
    def evaluate(self, symbol: str, context: dict[str, Any] | None = None) -> Signal:
        """Evaluate direction for a symbol.

        Returns a Signal with side = LONG/SHORT/SKIP.
        Must never return a Signal with side modified by risk considerations.
        """
        ...

    @abstractmethod
    def name(self) -> str:
        """Human-readable name of this direction provider."""
        ...

    @abstractmethod
    def version(self) -> str:
        """Version identifier for this direction provider."""
        ...
