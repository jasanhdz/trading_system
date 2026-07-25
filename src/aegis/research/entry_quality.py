"""Offline entry-quality scoring, calibration, and promotion assessment."""

from __future__ import annotations

import math
import statistics
from dataclasses import dataclass
from datetime import datetime
from typing import Mapping, Sequence

from ..models import CalibratorSpec
from .regime_v2 import DirectionRegime, RegimeV2Result, StructureRegime


@dataclass(frozen=True)
class HierarchicalProbabilityCalibrator:
    """Blend held-out global and symbol calibrators without hiding sample size."""

    schema_version: str
    global_calibrator: CalibratorSpec
    symbol_calibrators: Mapping[str, CalibratorSpec]
    symbol_sample_counts: Mapping[str, int]
    shrinkage_sample_count: int

    def __post_init__(self) -> None:
        if self.schema_version != "aegis-hierarchical-symbol-calibration-v1":
            raise ValueError("unsupported hierarchical calibration schema")
        if self.shrinkage_sample_count <= 0:
            raise ValueError("calibration shrinkage must be positive")
        if set(self.symbol_calibrators) != set(self.symbol_sample_counts):
            raise ValueError("symbol calibrator sample metadata is incomplete")
        if any(value <= 0 for value in self.symbol_sample_counts.values()):
            raise ValueError("symbol calibrator sample counts must be positive")

    def apply(self, symbol: str, raw_probability: float) -> float:
        if not math.isfinite(raw_probability):
            raise ValueError("raw opportunity probability must be finite")
        global_value = self.global_calibrator.apply(raw_probability)
        local = self.symbol_calibrators.get(symbol)
        if local is None:
            return global_value
        local_value = local.apply(raw_probability)
        count = self.symbol_sample_counts[symbol]
        local_weight = count / (count + self.shrinkage_sample_count)
        return global_value * (1.0 - local_weight) + local_value * local_weight


@dataclass(frozen=True)
class MaeAwareScoreContract:
    schema_version: str
    qmae_penalty: float
    tail_risk_penalty: float
    maximum_qmae_fraction: float
    maximum_tail_probability: float
    require_bearish_trend_context: bool

    def __post_init__(self) -> None:
        if self.schema_version != "aegis-mae-aware-score-contract-v1":
            raise ValueError("unsupported MAE-aware score schema")
        if min(self.qmae_penalty, self.tail_risk_penalty) < 0.0:
            raise ValueError("score penalties cannot be negative")
        if not 0.0 < self.maximum_qmae_fraction < 1.0:
            raise ValueError("maximum QMAE fraction is invalid")
        if not 0.0 < self.maximum_tail_probability < 1.0:
            raise ValueError("maximum tail probability is invalid")


@dataclass(frozen=True)
class EntryQualityInputs:
    symbol: str
    expected_short_return: float
    opportunity_probability: float
    qmae_q90: float
    tail_risk_probability: float
    qmae_valid: bool
    calibration_valid: bool
    regime: RegimeV2Result | None = None

    def __post_init__(self) -> None:
        values = (
            self.expected_short_return,
            self.opportunity_probability,
            self.qmae_q90,
            self.tail_risk_probability,
        )
        if not self.symbol or not all(math.isfinite(value) for value in values):
            raise ValueError("entry-quality inputs are invalid")
        if not 0.0 <= self.opportunity_probability <= 1.0:
            raise ValueError("opportunity probability is invalid")
        if self.qmae_q90 < 0.0 or not 0.0 <= self.tail_risk_probability <= 1.0:
            raise ValueError("entry-risk inputs are invalid")


@dataclass(frozen=True)
class EntryQualityScore:
    schema_version: str
    symbol: str
    eligible: bool
    score: float
    expected_clean_return: float
    qmae_penalty: float
    tail_risk_penalty: float
    reason_codes: tuple[str, ...]


def score_entry_quality(
    inputs: EntryQualityInputs,
    contract: MaeAwareScoreContract,
    calibrator: HierarchicalProbabilityCalibrator | None = None,
) -> EntryQualityScore:
    """Compute a ranking score; no threshold is selected inside this function."""
    reasons: list[str] = []
    if not inputs.calibration_valid:
        reasons.append("OPPORTUNITY_CALIBRATION_INVALID")
    if not inputs.qmae_valid:
        reasons.append("QMAE_INVALID")
    if inputs.qmae_q90 > contract.maximum_qmae_fraction:
        reasons.append("QMAE_LIMIT_EXCEEDED")
    if inputs.tail_risk_probability > contract.maximum_tail_probability:
        reasons.append("TAIL_RISK_LIMIT_EXCEEDED")
    if contract.require_bearish_trend_context:
        if (
            inputs.regime is None
            or not inputs.regime.evidence_ready
            or inputs.regime.direction is not DirectionRegime.BEARISH
            or inputs.regime.structure is not StructureRegime.TREND
        ):
            reasons.append("BEARISH_TREND_CONTEXT_NOT_ESTABLISHED")

    probability = (
        calibrator.apply(inputs.symbol, inputs.opportunity_probability)
        if calibrator is not None
        else inputs.opportunity_probability
    )
    expected_clean = probability * max(0.0, inputs.expected_short_return)
    qmae_penalty = contract.qmae_penalty * inputs.qmae_q90
    tail_penalty = contract.tail_risk_penalty * inputs.tail_risk_probability
    score = expected_clean - qmae_penalty - tail_penalty
    return EntryQualityScore(
        schema_version="aegis-mae-aware-entry-quality-score-v1",
        symbol=inputs.symbol,
        eligible=not reasons,
        score=score,
        expected_clean_return=expected_clean,
        qmae_penalty=qmae_penalty,
        tail_risk_penalty=tail_penalty,
        reason_codes=tuple(reasons) if reasons else ("ELIGIBLE_FOR_OFFLINE_RANKING",),
    )


@dataclass(frozen=True)
class EntryQualityEvidenceRow:
    timestamp: datetime
    symbol: str
    score: float
    qmae_q90: float
    net_return: float
    mae: float

    def __post_init__(self) -> None:
        if self.timestamp.tzinfo is None or not self.symbol:
            raise ValueError("entry-quality evidence identity is invalid")
        if not all(
            math.isfinite(value)
            for value in (self.score, self.qmae_q90, self.net_return, self.mae)
        ):
            raise ValueError("entry-quality evidence contains non-finite values")
        if self.qmae_q90 < 0.0 or self.mae < 0.0:
            raise ValueError("entry-quality risk evidence cannot be negative")


@dataclass(frozen=True)
class EntryQualityPromotionCriteria:
    schema_version: str
    minimum_rows: int
    minimum_symbols: int
    minimum_rows_per_evaluated_symbol: int
    minimum_score_return_correlation: float
    minimum_top_bottom_return_spread: float
    minimum_qmae_mae_correlation: float
    maximum_mean_mae: float
    maximum_negative_symbol_fraction: float

    def __post_init__(self) -> None:
        if self.schema_version != "aegis-entry-quality-promotion-criteria-v1":
            raise ValueError("unsupported entry-quality promotion schema")
        if min(self.minimum_rows, self.minimum_symbols, self.minimum_rows_per_evaluated_symbol) <= 0:
            raise ValueError("promotion sample requirements must be positive")
        if not -1.0 <= self.minimum_score_return_correlation <= 1.0:
            raise ValueError("score correlation requirement is invalid")
        if not -1.0 <= self.minimum_qmae_mae_correlation <= 1.0:
            raise ValueError("QMAE correlation requirement is invalid")
        if self.maximum_mean_mae <= 0.0:
            raise ValueError("maximum mean MAE must be positive")
        if not 0.0 <= self.maximum_negative_symbol_fraction <= 1.0:
            raise ValueError("negative-symbol allowance is invalid")


@dataclass(frozen=True)
class EntryQualityAssessment:
    schema_version: str
    row_count: int
    symbol_count: int
    mean_net_return: float
    mean_mae: float
    score_return_correlation: float | None
    qmae_mae_correlation: float | None
    top_bottom_return_spread: float | None
    negative_symbol_fraction: float
    per_symbol_mean_return: Mapping[str, float]
    passed: bool
    failures: tuple[str, ...]


def _correlation(left: Sequence[float], right: Sequence[float]) -> float | None:
    if len(left) < 3 or len(left) != len(right):
        return None
    left_mean = statistics.fmean(left)
    right_mean = statistics.fmean(right)
    numerator = math.fsum((x - left_mean) * (y - right_mean) for x, y in zip(left, right))
    left_scale = math.sqrt(math.fsum((x - left_mean) ** 2 for x in left))
    right_scale = math.sqrt(math.fsum((y - right_mean) ** 2 for y in right))
    if left_scale == 0.0 or right_scale == 0.0:
        return None
    return numerator / (left_scale * right_scale)


def assess_entry_quality(
    rows: Sequence[EntryQualityEvidenceRow],
    criteria: EntryQualityPromotionCriteria,
) -> EntryQualityAssessment:
    """Fail-closed assessment for an already frozen out-of-sample population."""
    ordered = sorted(rows, key=lambda row: (row.timestamp, row.symbol))
    symbols = sorted({row.symbol for row in ordered})
    per_symbol_rows = {symbol: [row for row in ordered if row.symbol == symbol] for symbol in symbols}
    evaluated_symbols = {
        symbol: values
        for symbol, values in per_symbol_rows.items()
        if len(values) >= criteria.minimum_rows_per_evaluated_symbol
    }
    per_symbol_mean = {
        symbol: statistics.fmean(row.net_return for row in values)
        for symbol, values in evaluated_symbols.items()
    }
    negative_fraction = (
        sum(value < 0.0 for value in per_symbol_mean.values()) / len(per_symbol_mean)
        if per_symbol_mean
        else 1.0
    )
    score_correlation = _correlation(
        [row.score for row in ordered], [row.net_return for row in ordered]
    )
    qmae_correlation = _correlation([row.qmae_q90 for row in ordered], [row.mae for row in ordered])
    bucket_size = max(1, len(ordered) // 10)
    ranked = sorted(ordered, key=lambda row: (row.score, row.symbol, row.timestamp))
    spread = (
        statistics.fmean(row.net_return for row in ranked[-bucket_size:])
        - statistics.fmean(row.net_return for row in ranked[:bucket_size])
        if len(ranked) >= 2
        else None
    )
    mean_net = statistics.fmean(row.net_return for row in ordered) if ordered else 0.0
    mean_mae = statistics.fmean(row.mae for row in ordered) if ordered else math.inf
    failures: list[str] = []
    if len(ordered) < criteria.minimum_rows:
        failures.append("INSUFFICIENT_ROWS")
    if len(evaluated_symbols) < criteria.minimum_symbols:
        failures.append("INSUFFICIENT_SYMBOL_COVERAGE")
    if score_correlation is None or score_correlation < criteria.minimum_score_return_correlation:
        failures.append("SCORE_RETURN_MONOTONICITY_FAILED")
    if spread is None or spread < criteria.minimum_top_bottom_return_spread:
        failures.append("TOP_BOTTOM_RETURN_SPREAD_FAILED")
    if qmae_correlation is None or qmae_correlation < criteria.minimum_qmae_mae_correlation:
        failures.append("QMAE_MAE_ORDERING_FAILED")
    if mean_mae > criteria.maximum_mean_mae:
        failures.append("MEAN_MAE_EXCEEDED")
    if negative_fraction > criteria.maximum_negative_symbol_fraction:
        failures.append("SYMBOL_HETEROGENEITY_EXCEEDED")
    return EntryQualityAssessment(
        schema_version="aegis-entry-quality-assessment-v1",
        row_count=len(ordered),
        symbol_count=len(evaluated_symbols),
        mean_net_return=mean_net,
        mean_mae=mean_mae,
        score_return_correlation=score_correlation,
        qmae_mae_correlation=qmae_correlation,
        top_bottom_return_spread=spread,
        negative_symbol_fraction=negative_fraction,
        per_symbol_mean_return=per_symbol_mean,
        passed=not failures,
        failures=tuple(failures),
    )


@dataclass(frozen=True)
class RegimeEvidenceRow:
    symbol: str
    direction: DirectionRegime
    structure: StructureRegime
    future_return: float
    net_short_return: float


def assess_regime_economics(rows: Sequence[RegimeEvidenceRow]) -> Mapping[str, Mapping[str, float | int]]:
    """Report regime economics without converting the regime into a Live gate."""
    groups: dict[str, list[RegimeEvidenceRow]] = {}
    for row in rows:
        key = f"{row.direction.value}:{row.structure.value}"
        groups.setdefault(key, []).append(row)
    result: dict[str, Mapping[str, float | int]] = {}
    for key, values in sorted(groups.items()):
        direction = values[0].direction
        direction_correct = [
            row.future_return < 0.0
            if direction is DirectionRegime.BEARISH
            else row.future_return > 0.0
            if direction is DirectionRegime.BULLISH
            else True
            for row in values
        ]
        result[key] = {
            "count": len(values),
            "directional_accuracy": sum(direction_correct) / len(values),
            "mean_future_return": statistics.fmean(row.future_return for row in values),
            "mean_net_short_return": statistics.fmean(row.net_short_return for row in values),
        }
    return result

