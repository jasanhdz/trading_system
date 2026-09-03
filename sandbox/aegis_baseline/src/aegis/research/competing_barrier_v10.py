"""Outcome-only competing-barrier contracts for V10 entry research."""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import Any, Mapping, Sequence


class BarrierOutcome(str, Enum):
    FAVORABLE_FIRST = "FAVORABLE_FIRST"
    ADVERSE_FIRST = "ADVERSE_FIRST"
    SAME_BAR_AMBIGUOUS = "SAME_BAR_AMBIGUOUS"
    NEITHER_REACHED = "NEITHER_REACHED"


class BarrierResearchError(ValueError):
    pass


@dataclass(frozen=True)
class BarrierContract:
    name: str
    favorable_fraction: float
    adverse_fraction: float
    horizon_bars: int
    severe_cost_fraction: float

    def __post_init__(self) -> None:
        values = (
            self.favorable_fraction,
            self.adverse_fraction,
            self.severe_cost_fraction,
        )
        if (
            not self.name
            or not all(math.isfinite(value) and value >= 0.0 for value in values)
            or min(self.favorable_fraction, self.adverse_fraction) <= 0.0
            or self.horizon_bars <= 0
        ):
            raise BarrierResearchError("invalid competing-barrier contract")


def contracts_from_config(config: Mapping[str, Any]) -> tuple[BarrierContract, ...]:
    barriers = config["barriers"]
    leverage = float(barriers["leverage_for_roe"])
    severe_cost = float(config["costs"]["severe_round_trip_fraction"])
    if not math.isfinite(leverage) or leverage <= 0.0:
        raise BarrierResearchError("invalid reporting leverage")
    favorable = tuple(float(value) for value in barriers["favorable_roe"])
    adverse = tuple(abs(float(value)) for value in barriers["adverse_roe"])
    if len(favorable) != len(adverse) or any(
        not math.isclose(left, right, rel_tol=0.0, abs_tol=1e-12)
        for left, right in zip(favorable, adverse)
    ):
        raise BarrierResearchError("V10 barriers must be symmetric")
    result = tuple(
        BarrierContract(
            name=f"ROE_{round(roe * 100):02d}_H{int(horizon)}",
            favorable_fraction=roe / leverage,
            adverse_fraction=loss / leverage,
            horizon_bars=int(horizon),
            severe_cost_fraction=severe_cost,
        )
        for roe, loss in zip(favorable, adverse)
        for horizon in barriers["horizons_bars"]
    )
    if len(result) != 9 or len({contract.name for contract in result}) != 9:
        raise BarrierResearchError("V10 requires exactly nine unique contracts")
    return result


def _finite(value: Any, name: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise BarrierResearchError(f"non-finite {name}")
    return result


def evaluate_barrier_path(
    *,
    side: str,
    entry_price: float,
    future_bars: Sequence[Any],
    contract: BarrierContract,
) -> Mapping[str, Any]:
    """Classify one future path without consulting any causal feature."""

    entry = _finite(entry_price, "entry price")
    if side not in {"LONG", "SHORT"} or entry <= 0.0:
        raise BarrierResearchError("invalid barrier path identity")
    if len(future_bars) < contract.horizon_bars:
        raise BarrierResearchError("incomplete future barrier path")
    outcome = BarrierOutcome.NEITHER_REACHED
    event_bar: int | None = None
    terminal_price = entry
    for index, bar in enumerate(future_bars[: contract.horizon_bars], start=1):
        high = _finite(bar.high, "high")
        low = _finite(bar.low, "low")
        close = _finite(bar.close, "close")
        if min(high, low, close) <= 0.0 or high < low:
            raise BarrierResearchError("invalid future OHLC")
        terminal_price = close
        if side == "LONG":
            favorable = high / entry - 1.0 >= contract.favorable_fraction
            adverse = 1.0 - low / entry >= contract.adverse_fraction
        else:
            favorable = 1.0 - low / entry >= contract.favorable_fraction
            adverse = high / entry - 1.0 >= contract.adverse_fraction
        if favorable or adverse:
            event_bar = index
            outcome = (
                BarrierOutcome.SAME_BAR_AMBIGUOUS
                if favorable and adverse
                else BarrierOutcome.FAVORABLE_FIRST
                if favorable
                else BarrierOutcome.ADVERSE_FIRST
            )
            break
    side_terminal = (
        terminal_price / entry - 1.0
        if side == "LONG"
        else 1.0 - terminal_price / entry
    )
    realized_utility = {
        BarrierOutcome.FAVORABLE_FIRST: contract.favorable_fraction
        - contract.severe_cost_fraction,
        BarrierOutcome.ADVERSE_FIRST: -contract.adverse_fraction
        - contract.severe_cost_fraction,
        BarrierOutcome.SAME_BAR_AMBIGUOUS: -contract.adverse_fraction
        - contract.severe_cost_fraction,
        BarrierOutcome.NEITHER_REACHED: side_terminal
        - contract.severe_cost_fraction,
    }[outcome]
    return {
        "contract": contract.name,
        "outcome": outcome.value,
        "event_bar": event_bar,
        "terminal_side_return": side_terminal,
        "realized_utility": realized_utility,
        "favorable_fraction": contract.favorable_fraction,
        "adverse_fraction": contract.adverse_fraction,
        "horizon_bars": contract.horizon_bars,
    }


def primary_direction_label(
    long_outcome: Mapping[str, Any], short_outcome: Mapping[str, Any]
) -> str:
    long_favorable = long_outcome["outcome"] == BarrierOutcome.FAVORABLE_FIRST.value
    short_favorable = short_outcome["outcome"] == BarrierOutcome.FAVORABLE_FIRST.value
    if long_favorable == short_favorable:
        return "ABSTAIN"
    return "LONG" if long_favorable else "SHORT"


def conservative_utility(
    probabilities: Mapping[str, float],
    contract: BarrierContract,
    *,
    unknown_penalty_fraction: float,
) -> float:
    expected = {outcome.value for outcome in BarrierOutcome}
    if set(probabilities) != expected:
        raise BarrierResearchError("incomplete outcome probability vector")
    values = {name: _finite(value, name) for name, value in probabilities.items()}
    if any(value < 0.0 or value > 1.0 for value in values.values()) or not math.isclose(
        sum(values.values()), 1.0, rel_tol=0.0, abs_tol=1e-6
    ):
        raise BarrierResearchError("invalid outcome probability vector")
    if not math.isfinite(unknown_penalty_fraction) or unknown_penalty_fraction < 0.0:
        raise BarrierResearchError("invalid unknown-state penalty")
    unknown = (
        values[BarrierOutcome.SAME_BAR_AMBIGUOUS.value]
        + values[BarrierOutcome.NEITHER_REACHED.value]
    )
    return (
        values[BarrierOutcome.FAVORABLE_FIRST.value] * contract.favorable_fraction
        - values[BarrierOutcome.ADVERSE_FIRST.value] * contract.adverse_fraction
        - contract.severe_cost_fraction
        - unknown * unknown_penalty_fraction * contract.adverse_fraction
    )


def deterministic_episode_mask(
    rows: Sequence[Mapping[str, Any]], *, spacing_minutes: int
) -> tuple[bool, ...]:
    """Select first event then fixed spacing per symbol, independent of outcomes."""

    if spacing_minutes <= 0:
        raise BarrierResearchError("episode spacing must be positive")
    last: dict[str, Any] = {}
    selected: list[bool] = []
    for row in rows:
        symbol = str(row["symbol"])
        timestamp = row["timestamp_value"]
        keep = symbol not in last or (
            timestamp - last[symbol]
        ).total_seconds() >= spacing_minutes * 60
        selected.append(keep)
        if keep:
            last[symbol] = timestamp
    return tuple(selected)
