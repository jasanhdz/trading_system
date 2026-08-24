from __future__ import annotations

from dataclasses import dataclass

from .models import Side


@dataclass(frozen=True, slots=True)
class CostScenario:
    name: str
    fee_bps_per_side: float
    slippage_bps_per_side: float


BASELINE = CostScenario("BASELINE", 5.0, 2.0)
STRESS_20 = CostScenario("STRESS_20", 5.0, 5.0)
STRESS_30 = CostScenario("STRESS_30", 5.0, 10.0)


def adverse_fill(base_price: float, transaction_side: Side, slippage_bps: float) -> float:
    multiplier = 1.0 + slippage_bps / 10_000.0 if transaction_side == "LONG" else 1.0 - slippage_bps / 10_000.0
    return base_price * multiplier


def gross_return(side: Side, entry_fill: float, exit_fill: float) -> float:
    direction = 1.0 if side == "LONG" else -1.0
    return direction * (exit_fill - entry_fill) / entry_fill


def fee_return(entry_fill: float, exit_fill: float, fee_bps_per_side: float = 5.0) -> float:
    fee = fee_bps_per_side / 10_000.0
    return fee * (1.0 + exit_fill / entry_fill)


def funding_return(side: Side, entry_fill: float, events: tuple[tuple[float, float], ...]) -> float:
    direction = 1.0 if side == "LONG" else -1.0
    return -direction * sum(rate * mark_price / entry_fill for rate, mark_price in events)
