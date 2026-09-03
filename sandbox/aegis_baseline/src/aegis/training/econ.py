"""Independent SHORT economic replay over canonical prices; never uses training labels."""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime, timedelta
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
    profit_factor: float | None
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


@dataclass(frozen=True)
class RawBaselineIndicators:
    ret_12: float
    close_to_ema24: float
    volatility_24: float


@dataclass(frozen=True)
class EconomicBaselineCycle:
    timestamp: datetime
    fold: int
    histories: Mapping[str, tuple[CanonicalBar, ...]]
    regimes: Mapping[str, Regime]
    trrm_probabilities: Mapping[str, float]
    eqm_scores: Mapping[str, float]
    trrm_survivors: tuple[str, ...]
    gated_symbols: tuple[str, ...]


@dataclass(frozen=True)
class EconomicBaselineSelection:
    signals: Mapping[str, tuple[EconomicSignal, ...]]
    fold_budgets: Mapping[int, int]
    eligible_counts: Mapping[str, Mapping[int, int]]
    selected_counts: Mapping[str, Mapping[int, int]]
    fold_status: Mapping[str, Mapping[int, str]]
    content_hash: str


BASELINE_STRATEGIES = (
    "no_trade", "random_directional_with_gates", "momentum_rule",
    "mean_reversion_rule", "volatility_rule", "eqm_only", "trrm_only",
)


def _metrics(trades: Sequence[EconomicTrade], *, bootstrap_repetitions: int, seed: int) -> EconomicMetrics:
    values = np.asarray([trade.net_return_fraction for trade in trades], dtype=np.float64)
    if not len(values):
        return EconomicMetrics(0, 0.0, None, 0.0, 0.0, 0.0, 0.0, 0, 0.0, (0.0, 0.0))
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
    strategy_ids: Sequence[str] = (),
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
    strategies = sorted({trade.signal.strategy_id for trade in trades} | set(strategy_ids))
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


def raw_baseline_indicators(history: Sequence[CanonicalBar]) -> RawBaselineIndicators | None:
    """Compute frozen ECON rule inputs directly from 288 contiguous raw bars."""
    if len(history) != 288:
        return None
    rows = tuple(history)
    if any(rows[index].timestamp - rows[index - 1].timestamp != timedelta(minutes=5) for index in range(1, len(rows))):
        return None
    closes = np.asarray([bar.close for bar in rows], dtype=np.float64)
    if not np.all(np.isfinite(closes)) or np.any(closes <= 0.0):
        return None
    ret_12 = float(closes[-1] / closes[-13] - 1.0)
    alpha = 2.0 / 25.0
    ema = float(closes[0])
    for close in closes[1:]:
        ema = alpha * float(close) + (1.0 - alpha) * ema
    true_ranges = []
    for index in range(len(rows) - 24, len(rows)):
        bar = rows[index]; previous_close = rows[index - 1].close
        true_range = max(bar.high - bar.low, abs(bar.high - previous_close), abs(bar.low - previous_close))
        true_ranges.append(true_range / bar.close)
    return RawBaselineIndicators(
        ret_12, float(closes[-1] / ema - 1.0), float(np.mean(true_ranges)),
    )


def _signal(cycle: EconomicBaselineCycle, symbol: str, score: float, strategy: str) -> EconomicSignal:
    return EconomicSignal(
        cycle.timestamp, symbol, TradeSide.SHORT, float(score), strategy, cycle.fold,
        cycle.regimes.get(symbol, Regime.UNKNOWN),
    )


def select_competition_baselines(
    cycles: Sequence[EconomicBaselineCycle], fold_budgets: Mapping[int, int], *, seed: int = 20260718,
) -> EconomicBaselineSelection:
    """Select the closed E3 baseline set with per-fold equal-budget caps."""
    if seed != 20260718:
        raise ValueError("ECON baseline seed is frozen at 20260718")
    rng = np.random.Generator(np.random.PCG64(seed))
    candidates: dict[str, list[EconomicSignal]] = {name: [] for name in BASELINE_STRATEGIES}
    eligible_counts = {name: {int(fold): 0 for fold in fold_budgets} for name in BASELINE_STRATEGIES}

    for cycle in sorted(cycles, key=lambda item: (item.timestamp, item.fold)):
        indicators = {
            symbol: value for symbol in sorted(cycle.histories)
            if (value := raw_baseline_indicators(cycle.histories[symbol])) is not None
        }
        if not indicators:
            continue
        survivor_set = set(cycle.trrm_survivors)
        gate_set = set(cycle.gated_symbols)
        rules: dict[str, list[tuple[float, str]]] = {
            "momentum_rule": [
                (-value.ret_12, symbol) for symbol, value in indicators.items()
                if value.ret_12 < 0.0 and value.close_to_ema24 < 0.0
            ],
            "mean_reversion_rule": [
                (value.ret_12, symbol) for symbol, value in indicators.items() if value.ret_12 > 0.0
            ],
            "volatility_rule": [(value.volatility_24, symbol) for symbol, value in indicators.items()],
            "eqm_only": [
                (float(cycle.eqm_scores[symbol]), symbol) for symbol in indicators
                if symbol in cycle.eqm_scores
            ],
            "trrm_only": [
                (1.0 - float(cycle.trrm_probabilities[symbol]), symbol) for symbol in indicators
                if symbol in survivor_set and symbol in cycle.trrm_probabilities
            ],
        }
        random_eligible = [symbol for symbol in sorted(indicators) if symbol in gate_set]
        random_rows = [(float(rng.random()), symbol) for symbol in random_eligible]
        rules["random_directional_with_gates"] = random_rows
        for strategy, rows in rules.items():
            eligible_counts[strategy].setdefault(cycle.fold, 0)
            eligible_counts[strategy][cycle.fold] += len(rows)
            if rows:
                score, symbol = sorted(rows, key=lambda item: (-item[0], item[1]))[0]
                candidates[strategy].append(_signal(cycle, symbol, score, strategy))

    selected: dict[str, tuple[EconomicSignal, ...]] = {"no_trade": ()}
    statuses: dict[str, dict[int, str]] = {name: {} for name in BASELINE_STRATEGIES}
    selected_counts: dict[str, dict[int, int]] = {name: {} for name in BASELINE_STRATEGIES}
    for strategy in BASELINE_STRATEGIES:
        if strategy == "no_trade":
            for fold, budget in sorted(fold_budgets.items()):
                statuses[strategy][fold] = "NO_TRADE_FOLD" if budget == 0 else "PERMANENT_ABSTENTION"
                selected_counts[strategy][fold] = 0
            continue
        chosen = []
        for fold, budget in sorted(fold_budgets.items()):
            fold_rows = [row for row in candidates[strategy] if row.fold == fold]
            ranked = sorted(fold_rows, key=lambda item: (-item.score, item.symbol, item.timestamp))
            count = min(int(budget), len(ranked))
            chosen.extend(ranked[:count])
            selected_counts[strategy][fold] = count
            statuses[strategy][fold] = "NO_TRADE_FOLD" if budget == 0 else ("BUDGET_FILLED" if count == budget else "ELIGIBILITY_DEFICIT")
        selected[strategy] = tuple(sorted(chosen, key=lambda item: (item.timestamp, item.symbol)))
    payload = {
        "signals": selected, "fold_budgets": dict(fold_budgets), "eligible_counts": eligible_counts,
        "selected_counts": selected_counts, "fold_status": statuses, "seed": seed,
    }
    return EconomicBaselineSelection(
        selected, dict(fold_budgets), eligible_counts, selected_counts, statuses,
        Sha256HashProvider().digest_value(payload),
    )
