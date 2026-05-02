from __future__ import annotations

from dataclasses import dataclass

from aegis_alpha.config import RiskConfig


@dataclass
class Position:
    side: int = 0
    size: float = 0.0
    entry_price: float = 0.0
    entry_step: int = 0


def current_roe(position: Position, price: float, cfg: RiskConfig) -> float:
    if position.side == 0 or position.entry_price <= 0:
        return 0.0
    raw = (price - position.entry_price) / position.entry_price
    if position.side < 0:
        raw = -raw
    return raw * cfg.leverage


def position_notional(balance: float, cfg: RiskConfig) -> float:
    return balance * cfg.leverage * cfg.position_fraction


def open_position(balance: float, side: int, price: float, step: int, cfg: RiskConfig) -> tuple[float, Position, float]:
    notional = position_notional(balance, cfg)
    fee = notional * cfg.total_fee
    if balance <= fee * 1.5:
        return balance, Position(), 0.0
    size = notional / max(price, 1e-10)
    if side < 0:
        size = -size
    return balance - fee, Position(side=side, size=size, entry_price=price, entry_step=step), fee


def close_position(balance: float, position: Position, price: float, cfg: RiskConfig) -> tuple[float, float, float]:
    if position.side == 0:
        return balance, 0.0, 0.0
    pnl = abs(position.size) * (price - position.entry_price if position.side > 0 else position.entry_price - price)
    fee = abs(position.size) * price * cfg.total_fee
    return max(0.0, balance + pnl - fee), pnl - fee, fee
