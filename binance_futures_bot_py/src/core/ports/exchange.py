"""Exchange interface definition."""

from abc import ABC, abstractmethod
from typing import List, Literal, Optional, Protocol

from ..types import Candle, OrderResult, PositionInfo, Side, SymbolFilters


class Exchange(Protocol):
    """Exchange interface for trading operations."""
    
    async def get_server_time(self) -> int:
        """Get current server timestamp."""
        ...
    
    async def get_candles(
        self, symbol: str, interval: str, limit: int
    ) -> List[Candle]:
        """Get historical candles."""
        ...
    
    async def get_mark_price(self, symbol: str) -> float:
        """Get current mark price."""
        ...
    
    async def read_liquidation_price(
        self, symbol: str, side: Side
    ) -> Optional[float]:
        """Get liquidation price for position."""
        ...
    
    async def get_usdt_balance(self) -> float:
        """Get available USDT balance."""
        ...
    
    async def set_leverage(self, symbol: str, leverage: int) -> None:
        """Set leverage for symbol."""
        ...
    
    async def get_symbol_filters(
        self, symbol: str, leverage: int
    ) -> SymbolFilters:
        """Get trading filters for symbol."""
        ...
    
    async def has_open_position(
        self, symbol: str, side: Literal["LONG", "SHORT", "ANY"]
    ) -> bool:
        """Check if position exists."""
        ...
    
    async def read_active_position(
        self, symbol: str, side_hint: Side
    ) -> Optional[PositionInfo]:
        """Read active position details."""
        ...
    
    async def market_open(
        self, symbol: str, side: Side, quantity: float
    ) -> OrderResult:
        """Open market position."""
        ...
    
    async def place_stop_close(
        self, symbol: str, side: Side, stop_price: float
    ) -> None:
        """Place stop loss order."""
        ...
    
    async def place_tp_close(
        self, symbol: str, side: Side, trigger_price: float
    ) -> None:
        """Place take profit order."""
        ...
    
    async def close_side_market_safe(
        self,
        symbol: str,
        side: Side,
        qty_abs: float,
        side_mode: Literal["BOTH", "LONG", "SHORT"],
    ) -> None:
        """Close position safely."""
        ...
    
    async def open_stop_for_side(
        self, symbol: str, side: Side
    ) -> Optional[dict]:
        """Get open stop order for side."""
        ...
    
    async def cancel_order_by_id(
        self, symbol: str, order_id: str
    ) -> None:
        """Cancel order by ID."""
        ...
