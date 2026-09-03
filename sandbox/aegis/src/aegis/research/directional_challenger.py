"""Deterministic evaluation utilities for directional research challengers."""

from __future__ import annotations

import math
import random
import statistics
from dataclasses import dataclass
from datetime import datetime
from typing import Mapping, Sequence


@dataclass(frozen=True)
class DirectionalEvidenceRow:
    timestamp: datetime
    symbol: str
    score: float
    net_return: float
    mae: float
    bad_entry: bool
    regime: str

    def __post_init__(self) -> None:
        if self.timestamp.tzinfo is None or not self.symbol or not self.regime:
            raise ValueError("directional evidence identity is invalid")
        if not all(
            math.isfinite(value)
            for value in (self.score, self.net_return, self.mae)
        ):
            raise ValueError("directional evidence contains non-finite values")
        if not 0.0 <= self.score <= 1.0 or self.mae < 0.0:
            raise ValueError("directional evidence score or MAE is invalid")


@dataclass(frozen=True)
class DirectionalSelectionContract:
    schema_version: str
    probability_quantiles: tuple[float, ...]
    minimum_calibration_selections: int
    minimum_scoring_selections: int
    maximum_mean_mae: float
    maximum_symbol_concentration: float
    bootstrap_resamples: int
    bootstrap_seed: int
    bootstrap_block_minutes: int = 720
    minimum_calibration_blocks: int = 10
    minimum_scoring_blocks: int = 10

    def __post_init__(self) -> None:
        if self.schema_version != "aegis-directional-selection-contract-v1":
            raise ValueError("unsupported directional selection contract")
        if (
            not self.probability_quantiles
            or tuple(sorted(set(self.probability_quantiles)))
            != self.probability_quantiles
            or not all(0.0 < value < 1.0 for value in self.probability_quantiles)
        ):
            raise ValueError("selection quantiles are invalid")
        if min(
            self.minimum_calibration_selections,
            self.minimum_scoring_selections,
            self.bootstrap_resamples,
            self.bootstrap_block_minutes,
            self.minimum_calibration_blocks,
            self.minimum_scoring_blocks,
        ) <= 0:
            raise ValueError("selection sample requirements must be positive")
        if self.maximum_mean_mae <= 0.0:
            raise ValueError("maximum mean MAE must be positive")
        if not 0.0 < self.maximum_symbol_concentration <= 1.0:
            raise ValueError("maximum symbol concentration is invalid")


@dataclass(frozen=True)
class DirectionalSelectionMetrics:
    signals: int
    independent_blocks: int
    mean_net_expectancy: float
    block_mean_net_expectancy: float
    expectancy_ci95_low: float | None
    expectancy_ci95_high: float | None
    win_rate: float
    bad_entry_rate: float
    mean_mae: float
    p90_mae: float
    symbol_counts: Mapping[str, int]
    symbol_concentration: float


@dataclass(frozen=True)
class DerivedSelectionPolicy:
    threshold: float
    allowed_regimes: tuple[str, ...]
    calibration_metrics: DirectionalSelectionMetrics
    calibration_valid: bool
    evaluated_thresholds: tuple[
        tuple[float, DirectionalSelectionMetrics, bool], ...
    ]


def within_symbol_percentiles(
    reference: Sequence[DirectionalEvidenceRow],
    target: Sequence[DirectionalEvidenceRow],
) -> tuple[float, ...]:
    """Map scores to held-out within-symbol empirical percentiles."""
    references: dict[str, tuple[float, ...]] = {}
    for symbol in sorted({row.symbol for row in reference}):
        references[symbol] = tuple(
            sorted(row.score for row in reference if row.symbol == symbol)
        )
    percentiles: list[float] = []
    for row in target:
        values = references.get(row.symbol)
        if not values:
            raise ValueError("symbol calibration reference is missing")
        rank = sum(value <= row.score for value in values)
        percentiles.append(rank / len(values))
    return tuple(percentiles)


def within_symbol_percentiles_with_global_fallback(
    reference: Sequence[DirectionalEvidenceRow],
    target: Sequence[DirectionalEvidenceRow],
) -> tuple[tuple[float, ...], tuple[str, ...]]:
    """Use held-out global calibration only when a symbol has no reference rows."""
    if not reference:
        raise ValueError("percentile calibration reference is empty")
    references = {
        symbol: tuple(
            sorted(row.score for row in reference if row.symbol == symbol)
        )
        for symbol in sorted({row.symbol for row in reference})
    }
    global_reference = tuple(sorted(row.score for row in reference))
    fallback_symbols = set()
    percentiles = []
    for row in target:
        values = references.get(row.symbol)
        if not values:
            values = global_reference
            fallback_symbols.add(row.symbol)
        rank = sum(value <= row.score for value in values)
        percentiles.append(rank / len(values))
    return tuple(percentiles), tuple(sorted(fallback_symbols))


def select_one_per_timestamp(
    rows: Sequence[DirectionalEvidenceRow],
    percentiles: Sequence[float],
    *,
    threshold: float,
    allowed_regimes: Sequence[str] = (),
) -> tuple[int, ...]:
    if len(rows) != len(percentiles) or not 0.0 < threshold < 1.0:
        raise ValueError("selection inputs are invalid")
    allowed = set(allowed_regimes)
    by_timestamp: dict[datetime, list[int]] = {}
    for index, (row, percentile) in enumerate(zip(rows, percentiles)):
        if not math.isfinite(percentile) or not 0.0 <= percentile <= 1.0:
            raise ValueError("selection percentile is invalid")
        if percentile < threshold or (allowed and row.regime not in allowed):
            continue
        by_timestamp.setdefault(row.timestamp, []).append(index)
    return tuple(
        max(
            indices,
            key=lambda index: (
                percentiles[index],
                rows[index].score,
                rows[index].symbol,
            ),
        )
        for _, indices in sorted(by_timestamp.items())
    )


def _bootstrap_interval(
    values: Sequence[float],
    *,
    resamples: int,
    seed: int,
) -> tuple[float | None, float | None]:
    if len(values) < 2:
        return None, None
    rng = random.Random(seed)
    means = sorted(
        statistics.fmean(rng.choice(values) for _ in values)
        for _ in range(resamples)
    )
    low = means[max(0, int(0.025 * (len(means) - 1)))]
    high = means[min(len(means) - 1, int(0.975 * (len(means) - 1)))]
    return low, high


def _temporal_block_returns(
    rows: Sequence[DirectionalEvidenceRow],
    selected: Sequence[int],
    *,
    block_minutes: int,
) -> tuple[float, ...]:
    if block_minutes <= 0:
        raise ValueError("bootstrap block duration must be positive")
    grouped: dict[int, list[float]] = {}
    block_seconds = block_minutes * 60
    for index in selected:
        row = rows[index]
        block = int(row.timestamp.timestamp()) // block_seconds
        grouped.setdefault(block, []).append(row.net_return)
    return tuple(
        statistics.fmean(values) for _, values in sorted(grouped.items())
    )


def selection_metrics(
    rows: Sequence[DirectionalEvidenceRow],
    selected: Sequence[int],
    *,
    bootstrap_resamples: int,
    bootstrap_seed: int,
    bootstrap_block_minutes: int = 720,
) -> DirectionalSelectionMetrics:
    chosen = [rows[index] for index in selected]
    returns = [row.net_return for row in chosen]
    block_returns = _temporal_block_returns(
        rows,
        selected,
        block_minutes=bootstrap_block_minutes,
    )
    counts = {
        symbol: sum(row.symbol == symbol for row in chosen)
        for symbol in sorted({row.symbol for row in chosen})
    }
    low, high = _bootstrap_interval(
        block_returns,
        resamples=bootstrap_resamples,
        seed=bootstrap_seed,
    )
    ordered_mae = sorted(row.mae for row in chosen)
    p90_index = (
        min(len(ordered_mae) - 1, math.ceil(0.90 * len(ordered_mae)) - 1)
        if ordered_mae
        else 0
    )
    return DirectionalSelectionMetrics(
        signals=len(chosen),
        independent_blocks=len(block_returns),
        mean_net_expectancy=statistics.fmean(returns) if returns else 0.0,
        block_mean_net_expectancy=(
            statistics.fmean(block_returns) if block_returns else 0.0
        ),
        expectancy_ci95_low=low,
        expectancy_ci95_high=high,
        win_rate=(
            sum(value > 0.0 for value in returns) / len(returns)
            if returns
            else 0.0
        ),
        bad_entry_rate=(
            sum(row.bad_entry for row in chosen) / len(chosen)
            if chosen
            else 0.0
        ),
        mean_mae=(
            statistics.fmean(row.mae for row in chosen) if chosen else 0.0
        ),
        p90_mae=ordered_mae[p90_index] if ordered_mae else 0.0,
        symbol_counts=counts,
        symbol_concentration=(
            max(counts.values()) / len(chosen) if chosen else 1.0
        ),
    )


def derive_selection_policy(
    calibration_rows: Sequence[DirectionalEvidenceRow],
    calibration_percentiles: Sequence[float],
    contract: DirectionalSelectionContract,
    *,
    allowed_regimes: Sequence[str] = (),
) -> DerivedSelectionPolicy:
    evaluated = []
    for threshold in contract.probability_quantiles:
        selected = select_one_per_timestamp(
            calibration_rows,
            calibration_percentiles,
            threshold=threshold,
            allowed_regimes=allowed_regimes,
        )
        metrics = selection_metrics(
            calibration_rows,
            selected,
            bootstrap_resamples=contract.bootstrap_resamples,
            bootstrap_seed=contract.bootstrap_seed,
            bootstrap_block_minutes=contract.bootstrap_block_minutes,
        )
        valid = (
            metrics.signals >= contract.minimum_calibration_selections
            and metrics.independent_blocks
            >= contract.minimum_calibration_blocks
            and metrics.block_mean_net_expectancy > 0.0
            and metrics.expectancy_ci95_low is not None
            and metrics.expectancy_ci95_low > 0.0
            and metrics.mean_mae <= contract.maximum_mean_mae
            and metrics.symbol_concentration
            <= contract.maximum_symbol_concentration
        )
        evaluated.append((threshold, metrics, valid))
    valid_rows = [row for row in evaluated if row[2]]
    pool = valid_rows or evaluated
    threshold, metrics, valid = max(
        pool,
        key=lambda row: (
            row[2],
            row[1].mean_net_expectancy,
            -row[1].mean_mae,
            row[1].signals,
        ),
    )
    return DerivedSelectionPolicy(
        threshold=threshold,
        allowed_regimes=tuple(allowed_regimes),
        calibration_metrics=metrics,
        calibration_valid=valid,
        evaluated_thresholds=tuple(evaluated),
    )


def scoring_policy_passes(
    metrics: DirectionalSelectionMetrics,
    contract: DirectionalSelectionContract,
) -> bool:
    """Require economic value, activity, risk control, and diversification."""
    return (
        metrics.signals >= contract.minimum_scoring_selections
        and metrics.independent_blocks >= contract.minimum_scoring_blocks
        and metrics.block_mean_net_expectancy > 0.0
        and metrics.expectancy_ci95_low is not None
        and metrics.expectancy_ci95_low > 0.0
        and metrics.mean_mae <= contract.maximum_mean_mae
        and metrics.symbol_concentration <= contract.maximum_symbol_concentration
    )
