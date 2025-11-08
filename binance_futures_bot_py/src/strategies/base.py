"""Base strategy interface."""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional
from dataclasses import dataclass

from ..core.types import Signal, BotState
from ..core.ports.exchange import Exchange
from ..core.ports.logger import Logger


@dataclass
class StrategyContext:
    """Context passed to strategy evaluation."""
    symbol: str
    exchange: Exchange
    config: Any
    state: Optional[BotState]
    now: int
    logger: Logger


class Strategy(ABC):
    """Base strategy interface."""
    
    def __init__(self, name: str, timeframe: str = "5m"):
        """Initialize strategy."""
        self.name = name
        self.timeframe = timeframe
    
    @abstractmethod
    async def evaluate(
        self,
        symbol: str,
        exchange: Exchange,
        config: Any,
        state: Optional[BotState],
        now: int,
        logger: Logger,
    ) -> Signal:
        """Evaluate strategy and return signal."""
        pass
