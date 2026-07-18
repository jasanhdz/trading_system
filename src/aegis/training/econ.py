"""Independent SHORT economic replay over canonical prices; never uses training labels."""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime
from typing import Mapping, Sequence

import numpy as np

from ..data import CanonicalBar
from ..domain import Regime, TradeSide
from ..utils import Sha256HashProvider


@dataclass(frozen=True)
class CostScenario:
    scenario_id: str
    fee_bps_per_side: float
    slippage_bps_per_side: float
    funding_bps_per_hour: float

    def cost_fraction(self, holding_bars: int) -> float:
        return (
            2.0 * (self.fee_bps_per_side + self.slippage_bps_per_side) / 10_000.0
            + self.funding_bps_per_hour / 10_000.0 * holding_bars * 5.0 / 60.0
        )


COST_SCENARIOS = (
    CostScenario("A_OPTIMISTIC", 4.0, 1.0, 0.5),
    CostScenario("B_BASE", 5.0, 2.0, 1.0),
    CostScenario("C_PESSIMISTIC", 5.0, 5.0, 2.0),
)


@dataclass(frozen=True)
class EconomicSignal:
    timestamp: datetime
    symbol: str
    side: TradeSide
    score: float
    strategy_id: str
    fold: int
    regime: Regime


@dataclass(frozen=True)
class EconomicTrade:
    signal: EconomicSignal
    scenario_id: str
    entry_timestamp: datetime
    exit_timestamp: datetime
    entry_price: float
    exit_price: float
    gross_return_fraction: float
    cost_fraction: float
    net_return_fraction: float
    mfe_fraction: float
    mae_fraction: float


@dataclass(frozen=True)
class EconomicMetrics:
    trades: int
    expectancy: float
    profit_factor: float
    win_rate: float
    maximum_drawdown: float
    cvar_05: float
    worst_trade: float
    turnover: int
    maximum_symbol_share: float
    bootstrap_ci_90: tuple[float, float]


@dataclass(frozen=True)
class EconomicReplayReport:
    schema_version: str
    holding_bars: int
    trades: tuple[EconomicTrade, ...]
    metrics: Mapping[str, Mapping[str, EconomicMetrics]]
    report_hash: str


def _metrics(trades: Sequence[EconomicTrade], *, bootstrap_repetitions: int, seed: int) -> EconomicMetrics:
    values = np.asarray([trade.net_return_fraction for trade in trades], dtype=np.float64)
    if not len(values):
        return EconomicMetrics(0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0, 0.0, (0.0, 0.0))
    gains = float(values[values > 0].sum()); losses = float(-values[values < 0].sum())
    equity = np.cumsum(values); peaks = np.maximum.accumulate(np.concatenate(([0.0], equity)))
    drawdown = float(np.max(peaks[1:] - equity))
    cutoff = max(1, math.ceil(len(values) * 0.05)); cvar = float(np.mean(np.sort(values)[:cutoff]))
    counts: dict[str, int] = {}
    for trade in trades:
        counts[trade.signal.symbol] = counts.get(trade.signal.symbol, 0) + 1
    weeks: dict[tuple[int, int], list[float]] = {}
    for trade in trades:
        iso = trade.signal.timestamp.isocalendar()
        weeks.setdefault((iso.year, iso.week), []).append(trade.net_return_fraction)
    rng = np.random.default_rng(seed); keys = sorted(weeks); means = []
    for _ in range(bootstrap_repetitions):
        sampled = rng.choice(len(keys), size=len(keys), replace=True)
        sample = [value for index in sampled for value in weeks[keys[int(index)]]]
        means.append(float(np.mean(sample)))
    ci = tuple(float(value) for value in np.quantile(means, (0.05, 0.95)))
    profit_factor = gains / max(losses, np.finfo(np.float64).eps) if gains > 0 else 0.0
    return EconomicMetrics(
        len(values), float(np.mean(values)), profit_factor,
        float(np.mean(values > 0)), drawdown, cvar, float(np.min(values)), len(values),
        max(counts.values()) / len(values), (ci[0], ci[1]),
    )


def replay_economics(
    signals: Sequence[EconomicSignal], prices: Mapping[str, Sequence[CanonicalBar]], *,
    holding_bars: int = 12, scenarios: Sequence[CostScenario] = COST_SCENARIOS,
    bootstrap_repetitions: int = 200, seed: int = 42,
) -> EconomicReplayReport:
    """Replay equal-budget signals at next-bar-open and fixed H12 close."""
    if holding_bars <= 0 or bootstrap_repetitions <= 0:
        raise ValueError("ECON replay parameters must be positive")
    by_symbol = {symbol: {bar.timestamp: index for index, bar in enumerate(rows)} for symbol, rows in prices.items()}
    trades = []
    for signal in sorted(signals, key=lambda item: (item.timestamp, item.strategy_id, item.symbol)):
        if signal.side is not TradeSide.SHORT:
            raise ValueError("ECON parity plan accepts SHORT signals only")
        rows = prices.get(signal.symbol); index = by_symbol.get(signal.symbol, {}).get(signal.timestamp)
        if rows is None or index is None or index + holding_bars >= len(rows):
            continue
        entry_bar = rows[index + 1]; exit_bar = rows[index + holding_bars]
        path = rows[index + 1:index + holding_bars + 1]
        if len(path) != holding_bars:
            continue
        entry = entry_bar.open; exit_price = exit_bar.close
        gross = (entry - exit_price) / entry
        mfe = max(0.0, (entry - min(bar.low for bar in path)) / entry)
        mae = max(0.0, (max(bar.high for bar in path) - entry) / entry)
        for scenario in scenarios:
            cost = scenario.cost_fraction(holding_bars)
            trades.append(EconomicTrade(
                signal, scenario.scenario_id, entry_bar.timestamp, exit_bar.timestamp,
                entry, exit_price, gross, cost, gross - cost, mfe, mae,
            ))
    strategies = sorted({trade.signal.strategy_id for trade in trades})
    scenario_ids = tuple(item.scenario_id for item in scenarios)
    metrics = {
        strategy: {
            scenario: _metrics(
                [trade for trade in trades if trade.signal.strategy_id == strategy and trade.scenario_id == scenario],
                bootstrap_repetitions=bootstrap_repetitions, seed=seed,
            )
            for scenario in scenario_ids
        }
        for strategy in strategies
    }
    payload = {"holding_bars": holding_bars, "trades": trades, "metrics": metrics}
    report_hash = Sha256HashProvider().digest_value(payload)
    return EconomicReplayReport("aegis-econ-replay-v1", holding_bars, tuple(trades), metrics, report_hash)


def equal_budget(signals_by_strategy: Mapping[str, Sequence[EconomicSignal]]) -> Mapping[str, tuple[EconomicSignal, ...]]:
    """Apply the same trade count to every compared strategy without resampling outcomes."""
    if not signals_by_strategy:
        return {}
    budget = min(len(signals) for signals in signals_by_strategy.values())
    return {
        strategy: tuple(sorted(signals, key=lambda item: (-item.score, item.timestamp, item.symbol))[:budget])
        for strategy, signals in sorted(signals_by_strategy.items())
    }
