"""Synthetic statistical primitives required by E5 Phase 0."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

import numpy as np

from .constants import HOLM_TEST_IDS, LABEL_ECONOMICS_REGISTRY, MIN_VALID_REPETITIONS, REQUESTED_REPETITIONS
from .core import namespaced_seed, normalize_fold, normalize_horizon, type7_quantile
from .errors import Phase0Error


def validate_resampling_configuration(requested: int, minimum_valid: int) -> None:
    if requested != REQUESTED_REPETITIONS or minimum_valid != MIN_VALID_REPETITIONS:
        raise Phase0Error("UNAUTHORIZED_SCIENTIFIC_CHOICE", "resampling requires 10,000 requested and 9,500 valid")


@dataclass(frozen=True)
class FiniteValidResult:
    requested: int
    valid: int
    invalid: int
    lower: float
    upper: float


def finite_valid_ci90(values: Sequence[float], requested: int = REQUESTED_REPETITIONS) -> FiniteValidResult:
    if len(values) != requested:
        raise Phase0Error("BOOTSTRAP_VALIDITY_FAILURE", "replicate count differs from requested count")
    finite = [float(value) for value in values if math.isfinite(float(value))]
    if len(finite) < MIN_VALID_REPETITIONS:
        raise Phase0Error("BOOTSTRAP_VALIDITY_FAILURE", f"only {len(finite)} valid replicates")
    return FiniteValidResult(requested, len(finite), requested - len(finite), type7_quantile(finite, 0.05), type7_quantile(finite, 0.95))


def deterministic_bootstrap_indices(block_count: int, statistic: str, horizon: str, scope: str, replicate_index: int) -> tuple[int, ...]:
    if block_count < 1:
        raise Phase0Error("BOOTSTRAP_VALIDITY_FAILURE", "bootstrap requires source blocks")
    seed, _ = namespaced_seed("BOOTSTRAP", statistic, normalize_horizon(horizon), scope, replicate_index)
    generator = np.random.Generator(np.random.PCG64(seed))
    return tuple(int(item) for item in generator.integers(0, block_count, size=block_count))


def fold_centered_residuals(values_by_fold: Mapping[str, Sequence[float]], mrem: float = 0.0005) -> dict[str, tuple[float, ...]]:
    if mrem != 0.0005:
        raise Phase0Error("UNAUTHORIZED_SCIENTIFIC_CHOICE", "MREM is fixed")
    reconstructed: dict[str, tuple[float, ...]] = {}
    for fold in sorted(values_by_fold):
        values = tuple(float(value) for value in values_by_fold[fold])
        if not values or any(not math.isfinite(value) for value in values):
            raise Phase0Error("POWER_NOT_COMPUTABLE", fold)
        mean = sum(values) / len(values)
        residuals = tuple(value - mean for value in values)
        reconstructed[fold] = tuple(mean + residual + mrem for residual in residuals)
    return reconstructed


def score_deciles(rows: Sequence[tuple[float, str, int]]) -> tuple[int, ...]:
    """Return bin assignments in canonical score-descending row order."""
    if not rows:
        raise Phase0Error("OUTCOME_NOT_COMPUTABLE", "score deciles require rows")
    ordered = sorted(rows, key=lambda row: (-float(row[0]), row[1], row[2]))
    count = len(ordered)
    return tuple(10 - math.floor(10 * index / count) for index in range(count))


def complete_week_starts(fold_start_ms: int, fold_end_ms: int) -> tuple[int, ...]:
    day_ms = 86_400_000
    week_ms = 7 * day_ms
    first_day = np.datetime64(fold_start_ms, "ms").astype("datetime64[D]")
    weekday = int((first_day.astype(int) + 3) % 7)
    first_monday = int((first_day + np.timedelta64((7 - weekday) % 7, "D")).astype("datetime64[ms]").astype(int))
    starts: list[int] = []
    current = first_monday
    while current + week_ms - 1 <= fold_end_ms:
        starts.append(current)
        current += week_ms
    return tuple(starts)


def temporal_shift(test_name: str, horizon: str, fold: int | str, repetition_index: int, week_count: int) -> int:
    if week_count < 4:
        raise Phase0Error("INSUFFICIENT_COMPLETE_WEEKS", f"only {week_count} complete weeks")
    seed, _ = namespaced_seed(test_name, normalize_horizon(horizon), normalize_fold(fold), repetition_index)
    generator = np.random.Generator(np.random.PCG64(seed))
    return int(generator.integers(1, week_count))


def circular_shift(values: Sequence[object], shift: int) -> tuple[object, ...]:
    if not values or shift <= 0 or shift >= len(values):
        raise Phase0Error("PERMUTATION_VALIDITY_FAILURE", "shift must be nonzero and within block count")
    return tuple(values[-shift:]) + tuple(values[:-shift])


@dataclass(frozen=True)
class HolmDecision:
    test_id: str
    raw_p: float
    adjusted_p: float
    rejected: bool


def holm_adjust(raw_p_values: Mapping[str, float], alpha: float = 0.05) -> tuple[HolmDecision, ...]:
    if tuple(raw_p_values) != HOLM_TEST_IDS:
        missing = set(HOLM_TEST_IDS) - set(raw_p_values)
        extra = set(raw_p_values) - set(HOLM_TEST_IDS)
        raise Phase0Error("HOLM_FAMILY_INCOMPLETE", f"missing={sorted(missing)} extra={sorted(extra)} order={tuple(raw_p_values)}")
    indexed: list[tuple[int, str, float]] = []
    for position, test_id in enumerate(HOLM_TEST_IDS):
        value = float(raw_p_values[test_id])
        if not math.isfinite(value) or not 0.0 <= value <= 1.0:
            raise Phase0Error("HOLM_FAMILY_INCOMPLETE", f"invalid p-value for {test_id}")
        indexed.append((position, test_id, value))
    ordered = sorted(indexed, key=lambda item: (item[2], item[0]))
    adjusted_by_id: dict[str, float] = {}
    rejected_by_id: dict[str, bool] = {}
    running_adjusted = 0.0
    still_rejecting = True
    family_size = len(ordered)
    for rank, (_, test_id, value) in enumerate(ordered, start=1):
        multiplier = family_size - rank + 1
        running_adjusted = max(running_adjusted, min(1.0, multiplier * value))
        adjusted_by_id[test_id] = running_adjusted
        rejected = still_rejecting and value <= alpha / multiplier
        rejected_by_id[test_id] = rejected
        if not rejected:
            still_rejecting = False
    return tuple(HolmDecision(test_id, raw_p_values[test_id], adjusted_by_id[test_id], rejected_by_id[test_id]) for test_id in HOLM_TEST_IDS)


def validate_label_registry(registry: Mapping[str, Mapping[str, object]]) -> bytes:
    if dict(registry) != LABEL_ECONOMICS_REGISTRY:
        raise Phase0Error("LABEL_SCHEMA_AMBIGUITY", "D10 registry differs from authority")
    from .core import canonical_json_bytes

    return canonical_json_bytes(registry)


def label_classification(connected_direction: bool, ordering_valid: bool, favorable_folds: int, economic_difference: float) -> str:
    if not connected_direction or not ordering_valid or favorable_folds < 3:
        return "LABEL_ECONOMICS_DISCONNECTED"
    if economic_difference < 0.0005:
        return "LABEL_ECONOMICS_CONNECTED_EFFECT_TOO_SMALL"
    return "LABEL_ECONOMICS_CONNECTED_MATERIAL"


def pooled_positive_pnl_concentration(returns_by_symbol: Mapping[str, Sequence[float]]) -> tuple[float, str | None]:
    contributions = {
        symbol: sum(max(float(value), 0.0) for value in values)
        for symbol, values in sorted(returns_by_symbol.items())
    }
    denominator = sum(contributions.values())
    if denominator == 0.0:
        return 0.0, "NO_POSITIVE_PNL"
    return max(contributions.values(), default=0.0) / denominator, None


AUTHORITY_CLASSIFICATION = {
    "pooled_concentration": "MANDATORY_GATE",
    "fold_concentration": "DIAGNOSTIC_ONLY_EXCEPT_DATA_INTEGRITY",
    "spearman_ic": "DIAGNOSTIC_ONLY",
}


def validate_diagnostic_authority(classification: Mapping[str, str]) -> None:
    if dict(classification) != AUTHORITY_CLASSIFICATION:
        raise Phase0Error("UNAUTHORIZED_SCIENTIFIC_CHOICE", "diagnostic authority changed")


def average_ranks(values: Sequence[float]) -> tuple[float, ...]:
    if any(not math.isfinite(float(value)) for value in values):
        raise Phase0Error("IC_NOT_COMPUTABLE", "non-finite rank input")
    order = sorted(range(len(values)), key=lambda index: (values[index], index))
    ranks = [0.0] * len(values)
    cursor = 0
    while cursor < len(order):
        end = cursor + 1
        while end < len(order) and values[order[end]] == values[order[cursor]]:
            end += 1
        average = (cursor + 1 + end) / 2.0
        for position in order[cursor:end]:
            ranks[position] = average
        cursor = end
    return tuple(ranks)


def spearman(values_x: Sequence[float], values_y: Sequence[float]) -> float:
    if len(values_x) != len(values_y) or len(values_x) < 2:
        raise Phase0Error("IC_NOT_COMPUTABLE", "Spearman needs at least two pairs")
    x = average_ranks(values_x)
    y = average_ranks(values_y)
    mean_x = sum(x) / len(x)
    mean_y = sum(y) / len(y)
    numerator = sum((left - mean_x) * (right - mean_y) for left, right in zip(x, y))
    denominator = math.sqrt(sum((value - mean_x) ** 2 for value in x) * sum((value - mean_y) ** 2 for value in y))
    if denominator == 0.0:
        raise Phase0Error("IC_NOT_COMPUTABLE", "constant ranks")
    return numerator / denominator
