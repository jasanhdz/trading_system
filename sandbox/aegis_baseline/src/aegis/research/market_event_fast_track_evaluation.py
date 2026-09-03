"""Economic evaluation and fail-closed gates for M1A pattern candidates."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Mapping, Sequence

from ..domain import TradeSide
from .market_event_fast_track_m1a import (
    DirectionAxis,
    FastTrackContractError,
    MicroPattern,
    MinuteBar,
    PatternCandidate,
    VolatilityAxis,
)


@dataclass(frozen=True)
class EvaluatedPattern:
    event_id: str
    pattern: MicroPattern
    side: TradeSide
    symbol: str
    timestamp_ms: int
    regime_direction: DirectionAxis
    regime_volatility: VolatilityAxis
    entry_timestamp_ms: int
    exit_timestamp_ms: int
    horizon_minutes: int
    entry_price: float
    exit_price: float
    gross_return_fraction: float
    cost_fraction: float
    net_return_fraction: float
    mae_fraction: float
    mfe_fraction: float


@dataclass(frozen=True)
class EconomicMetrics:
    events: int
    net_expectancy: float
    expectancy_ci_95: tuple[float, float]
    profit_factor: float
    profit_factor_ci_95: tuple[float, float]
    win_rate: float
    mean_mae: float
    mean_mfe: float
    maximum_drawdown: float
    cvar_05: float
    positive_temporal_thirds: int
    symbol_share_maximum: float


@dataclass(frozen=True)
class PatternGate:
    passed: bool
    blockers: tuple[str, ...]


def evaluate_candidate(
    candidate: PatternCandidate,
    future: Sequence[MinuteBar],
    *,
    horizon_minutes: int,
    fee_bps_per_side: float = 5.0,
    slippage_bps_per_side: float = 2.0,
    funding_bps_per_hour: float = 1.0,
) -> EvaluatedPattern:
    if candidate.side not in {TradeSide.LONG, TradeSide.SHORT}:
        raise FastTrackContractError("AEGIS_M1A_EVALUATION_SIDE_INVALID")
    if horizon_minutes <= 0 or len(future) < horizon_minutes:
        raise FastTrackContractError("AEGIS_M1A_EVALUATION_PATH_INCOMPLETE")
    path = tuple(future[:horizon_minutes])
    expected_start = candidate.timestamp_ms + 1
    if path[0].open_time_ms != expected_start:
        raise FastTrackContractError("AEGIS_M1A_NEXT_BAR_ENTRY_MISMATCH")
    if any(
        row.interval_minutes != 1
        or row.open_time_ms != path[0].open_time_ms + index * 60_000
        for index, row in enumerate(path)
    ):
        raise FastTrackContractError("AEGIS_M1A_EVALUATION_PATH_GAP")
    costs_raw = (fee_bps_per_side, slippage_bps_per_side, funding_bps_per_hour)
    if not all(math.isfinite(value) and value >= 0.0 for value in costs_raw):
        raise FastTrackContractError("AEGIS_M1A_EVALUATION_COST_INVALID")
    entry = path[0].open
    exit_price = path[-1].close
    sign = 1.0 if candidate.side is TradeSide.LONG else -1.0
    gross = sign * (exit_price - entry) / entry
    favorable = (
        max((row.high - entry) / entry for row in path)
        if sign > 0
        else max((entry - row.low) / entry for row in path)
    )
    adverse = (
        max((entry - row.low) / entry for row in path)
        if sign > 0
        else max((row.high - entry) / entry for row in path)
    )
    cost = (
        2.0 * (fee_bps_per_side + slippage_bps_per_side) / 10_000.0
        + funding_bps_per_hour / 10_000.0 * horizon_minutes / 60.0
    )
    event_id = f"{candidate.pattern.value}:{candidate.side.value}:{candidate.symbol}:{candidate.timestamp_ms}"
    return EvaluatedPattern(
        event_id=event_id,
        pattern=candidate.pattern,
        side=candidate.side,
        symbol=candidate.symbol,
        timestamp_ms=candidate.timestamp_ms,
        regime_direction=candidate.regime_direction,
        regime_volatility=candidate.regime_volatility,
        entry_timestamp_ms=path[0].open_time_ms,
        exit_timestamp_ms=path[-1].close_time_ms,
        horizon_minutes=horizon_minutes,
        entry_price=entry,
        exit_price=exit_price,
        gross_return_fraction=gross,
        cost_fraction=cost,
        net_return_fraction=gross - cost,
        mae_fraction=max(0.0, adverse),
        mfe_fraction=max(0.0, favorable),
    )


def _sample_metrics(rows: Sequence[EvaluatedPattern]) -> tuple[float, float]:
    values = [row.net_return_fraction for row in rows]
    mean = sum(values) / len(values)
    gains = sum(value for value in values if value > 0.0)
    losses = -sum(value for value in values if value < 0.0)
    profit_factor = gains / losses if losses > 0.0 else math.inf if gains > 0.0 else 0.0
    return mean, profit_factor


def _day_blocks(rows: Sequence[EvaluatedPattern]) -> tuple[tuple[EvaluatedPattern, ...], ...]:
    grouped: dict[int, list[EvaluatedPattern]] = {}
    for row in rows:
        grouped.setdefault(row.timestamp_ms // 86_400_000, []).append(row)
    return tuple(tuple(grouped[key]) for key in sorted(grouped))


def summarize_economics(
    rows: Sequence[EvaluatedPattern], *, seed: int = 180102, bootstrap_repetitions: int = 2000
) -> EconomicMetrics:
    if not rows or bootstrap_repetitions < 100:
        raise FastTrackContractError("AEGIS_M1A_METRICS_INPUT_INVALID")
    if len({row.event_id for row in rows}) != len(rows):
        raise FastTrackContractError("AEGIS_M1A_DUPLICATE_EVALUATED_EVENT")
    ordered = tuple(sorted(rows, key=lambda item: (item.timestamp_ms, item.event_id)))
    mean, profit_factor = _sample_metrics(ordered)
    blocks = _day_blocks(ordered)
    rng = random.Random(seed)
    bootstrap = []
    for _ in range(bootstrap_repetitions):
        sample = tuple(item for _ in blocks for item in blocks[rng.randrange(len(blocks))])
        bootstrap.append(_sample_metrics(sample))
    bootstrap.sort(key=lambda item: item[0])
    means = sorted(item[0] for item in bootstrap)
    profit_factors = sorted(item[1] for item in bootstrap)
    lower = int((bootstrap_repetitions - 1) * 0.025)
    upper = int((bootstrap_repetitions - 1) * 0.975)
    thirds = []
    for third in range(3):
        start = len(ordered) * third // 3
        end = len(ordered) * (third + 1) // 3
        thirds.append(sum(row.net_return_fraction for row in ordered[start:end]) / max(end - start, 1))
    equity = 0.0
    peak = 0.0
    drawdown = 0.0
    for row in ordered:
        equity += row.net_return_fraction
        peak = max(peak, equity)
        drawdown = max(drawdown, peak - equity)
    values = sorted(row.net_return_fraction for row in ordered)
    tail_count = max(1, math.ceil(len(values) * 0.05))
    symbol_counts: dict[str, int] = {}
    for row in ordered:
        symbol_counts[row.symbol] = symbol_counts.get(row.symbol, 0) + 1
    return EconomicMetrics(
        events=len(ordered),
        net_expectancy=mean,
        expectancy_ci_95=(means[lower], means[upper]),
        profit_factor=profit_factor,
        profit_factor_ci_95=(
            profit_factors[lower],
            profit_factors[upper],
        ),
        win_rate=sum(row.net_return_fraction > 0.0 for row in ordered) / len(ordered),
        mean_mae=sum(row.mae_fraction for row in ordered) / len(ordered),
        mean_mfe=sum(row.mfe_fraction for row in ordered) / len(ordered),
        maximum_drawdown=drawdown,
        cvar_05=sum(values[:tail_count]) / tail_count,
        positive_temporal_thirds=sum(value > 0.0 for value in thirds),
        symbol_share_maximum=max(symbol_counts.values()) / len(ordered),
    )


def assess_pattern_gate(
    metrics: EconomicMetrics,
    *,
    matched_random_expectancy: float,
    stress_expectancy: float,
) -> PatternGate:
    blockers = []
    if metrics.events < 100:
        blockers.append("EVENT_COUNT_LT_100")
    if metrics.expectancy_ci_95[0] <= 0.0:
        blockers.append("EXPECTANCY_CI_LOWER_NOT_POSITIVE")
    if metrics.profit_factor_ci_95[0] <= 1.0:
        blockers.append("PROFIT_FACTOR_CI_LOWER_NOT_ABOVE_ONE")
    if metrics.positive_temporal_thirds < 2:
        blockers.append("TEMPORAL_THIRDS_LT_2")
    if metrics.symbol_share_maximum > 0.25:
        blockers.append("SYMBOL_CONCENTRATION_GT_25_PERCENT")
    if metrics.net_expectancy <= matched_random_expectancy:
        blockers.append("DOES_NOT_BEAT_MATCHED_RANDOM")
    if stress_expectancy <= 0.0:
        blockers.append("FAILS_FIRST_COST_STRESS")
    return PatternGate(not blockers, tuple(blockers))


def summarize_by_pattern_side(
    rows: Sequence[EvaluatedPattern], *, bootstrap_repetitions: int = 2000
) -> Mapping[str, EconomicMetrics]:
    grouped: dict[str, list[EvaluatedPattern]] = {}
    for row in rows:
        key = f"{row.pattern.value}:{row.side.value}"
        grouped.setdefault(key, []).append(row)
    return {
        key: summarize_economics(
            values,
            seed=180102 + index,
            bootstrap_repetitions=bootstrap_repetitions,
        )
        for index, (key, values) in enumerate(sorted(grouped.items()))
    }
