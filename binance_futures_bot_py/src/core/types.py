"""Core type definitions for the trading bot."""

from dataclasses import dataclass
from enum import Enum
from typing import Literal, Optional, TypedDict


class Side(str, Enum):
    """Trading side."""
    LONG = "LONG"
    SHORT = "SHORT"


class BotMode(str, Enum):
    """Bot operation mode."""
    IDLE = "IDLE"
    LONG_RIDE = "LONG_RIDE"
    SHORT_RIDE = "SHORT_RIDE"


@dataclass
class Candle:
    """OHLCV candle data."""
    open_time: int
    open: float
    high: float
    low: float
    close: float
    volume: float
    close_time: int

    @property
    def typical_price(self) -> float:
        """Calculate typical price (HLC/3)."""
        return (self.high + self.low + self.close) / 3

    @property
    def ohlc4(self) -> float:
        """Calculate OHLC/4 price."""
        return (self.open + self.high + self.low + self.close) / 4


@dataclass
class BotState:
    """Bot state persistence."""
    mode: BotMode
    last_side: Optional[Side] = None
    last_entry_price: Optional[float] = None
    last_leverage: Optional[float] = None
    last_entry_at: Optional[int] = None
    peak_roe: Optional[float] = None
    last_tp_at: Optional[int] = None
    last_exit_reason: Optional[str] = None
    last_exit_at: Optional[int] = None
    
    # Bracket order tracking
    brackets_armed_at: Optional[int] = None
    pos_side_mode: Optional[Literal["BOTH", "LONG", "SHORT"]] = None
    
    # Pyramiding support
    last_entry_qty: Optional[float] = None
    pyramid_units: Optional[int] = None
    last_pyramid_price: Optional[float] = None
    last_trail_stop: Optional[float] = None
    brackets_attached: Optional[bool] = None
    last_intelli_tp_at: Optional[int] = None
    intelli_tp_state: Optional[str] = None


class Signal(TypedDict):
    """Trading signal from strategy."""
    action: Literal["ENTER_LONG", "ENTER_SHORT", "EXIT", "IDLE"]
    reason: Optional[str]


@dataclass
class Trade:
    """Completed trade record."""
    side: Side
    entry_idx: int
    entry_ts: int
    entry_px: float
    exit_idx: int
    exit_ts: int
    exit_px: float
    exit: Literal["TP", "SL", "Timeout", "StrategyExit"]
    bars_held: int
    pnl_pct: float
    mfe_pct: float  # Maximum Favorable Excursion
    mae_pct: float  # Maximum Adverse Excursion
    reason: Optional[str] = None
    
    # Strategy-specific metrics
    adx: Optional[float] = None
    v_ratio: Optional[float] = None
    bb_upper: Optional[float] = None
    bb_lower: Optional[float] = None
    dist_top_pct: Optional[float] = None


@dataclass
class PositionInfo:
    """Active position information."""
    side_mode: Literal["BOTH", "LONG", "SHORT"]
    qty_abs: float  # Absolute quantity
    entry_price: float
    leverage: float
    unrealized_pnl: Optional[float] = None
    mark_price: Optional[float] = None
    liquidation_price: Optional[float] = None


@dataclass
class SymbolFilters:
    """Trading symbol constraints."""
    tick_size: float
    step_size: float
    price_precision: int
    qty_precision: int
    min_notional: float
    notional_cap: Optional[float] = None  # From risk bracket
    max_qty: Optional[float] = None
    min_qty: Optional[float] = None


@dataclass
class OrderResult:
    """Order execution result."""
    order_id: str
    avg_price: float
    executed_qty: float
    status: str
    side: Side
    type: str
