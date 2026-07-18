"""Pure ordered scientific layers for regime, risk, quality, and economics."""

from __future__ import annotations

import math
import statistics
from dataclasses import dataclass
from typing import Protocol

from .domain import (
    LayerOutputs,
    LayerResult,
    ModelPredictions,
    ReasonCode,
    Regime,
    ScientificContext,
    ScientificLayerName,
    TradeSide,
)


class ScientificLayers(Protocol):
    def apply(self, predictions: ModelPredictions, context: ScientificContext) -> LayerOutputs: ...


@dataclass(frozen=True)
class LayerSettings:
    trrm_max_tail_probability: float
    qmae_max_fraction: float
    eqm_min_score: float
    estimated_round_trip_cost_fraction: float
    direction_threshold: float


def _mean(values: list[float]) -> float:
    return math.fsum(values) / len(values) if values else 0.0


def _clip(value: float) -> float:
    return max(0.0, min(1.0, value))


@dataclass(frozen=True)
class OrderedScientificLayers:
    """Execute REGIME -> RV2 -> TRRM -> QMAE -> EQM -> ECON1 deterministically."""

    settings: LayerSettings

    def apply(self, predictions: ModelPredictions, context: ScientificContext) -> LayerOutputs:
        results: list[LayerResult] = []
        for row in context.features.rows:
            symbol_predictions = list(predictions.for_symbol(row.symbol))
            if not symbol_predictions:
                results.append(self._unavailable(row.symbol))
                continue
            features = dict(zip(context.features.feature_names, row.raw_values))
            regime, regime_confidence = classify_market_regime(features)
            long_probability = _mean([item.long_probability for item in symbol_predictions])
            short_probability = _mean([item.short_probability for item in symbol_predictions])
            direction_probability = max(long_probability, short_probability)
            if direction_probability < self.settings.direction_threshold:
                side = TradeSide.NO_TRADE
            else:
                side = TradeSide.LONG if long_probability >= short_probability else TradeSide.SHORT

            rv2_tail = _mean([item.tail_risk_probability for item in symbol_predictions])
            trrm_compatibility = 1.0 - rv2_tail
            qmae_values = [item.qmae_q90 for item in symbol_predictions if item.qmae_valid and item.qmae_q90 is not None]
            qmae_valid = len(qmae_values) == len(symbol_predictions)
            qmae_q90 = _mean(qmae_values) if qmae_valid else None
            qmae_quality = (
                _clip(1.0 - qmae_q90 / max(self.settings.qmae_max_fraction, 1e-12))
                if qmae_q90 is not None else 0.0
            )
            expected_returns = [item.expected_return for item in symbol_predictions]
            directional_returns = [value if side is TradeSide.LONG else -value for value in expected_returns]
            expected_edge = _mean(directional_returns) if side is not TradeSide.NO_TRADE else 0.0
            clean_probability = _mean([item.quality_probability for item in symbol_predictions])
            eqm_score = clean_probability * max(0.0, expected_edge)
            directional_probabilities = [
                item.long_probability if side is TradeSide.LONG else item.short_probability
                for item in symbol_predictions
            ]
            disagreement = statistics.pstdev(directional_probabilities) if len(directional_probabilities) > 1 else 0.0
            econ_edge = expected_edge - self.settings.estimated_round_trip_cost_fraction

            reasons: list[ReasonCode] = []
            if side is TradeSide.NO_TRADE:
                reasons.append(ReasonCode.MODEL_NO_DIRECTION)
            if side is TradeSide.LONG:
                reasons.append(ReasonCode.SIDE_NOT_ENABLED)
            if not all(item.calibration_valid for item in symbol_predictions):
                reasons.append(ReasonCode.CALIBRATION_MISSING)
            if rv2_tail > self.settings.trrm_max_tail_probability:
                reasons.append(ReasonCode.TRRM_TAIL_RISK_VETO)
            if not qmae_valid:
                reasons.append(ReasonCode.QMAE_QUANTILE_UNAVAILABLE)
            elif qmae_q90 is not None and qmae_q90 > self.settings.qmae_max_fraction:
                reasons.append(ReasonCode.QMAE_ADVERSE_EXCURSION_HIGH)
            if eqm_score < self.settings.eqm_min_score:
                reasons.append(ReasonCode.EQM_QUALITY_LOW)
            if econ_edge <= 0:
                reasons.append(ReasonCode.ECON1_EDGE_BELOW_COST)
            eligible = not reasons
            if eligible:
                reasons.append(ReasonCode.ELIGIBLE)
            calibrated_score = _clip(
                direction_probability
                * trrm_compatibility
                * qmae_quality
                * clean_probability
                * (1.0 - min(1.0, disagreement))
                * regime_confidence
            )
            results.append(LayerResult(
                symbol=row.symbol, side=side, regime=regime, regime_confidence=regime_confidence,
                rv2_tail_risk=rv2_tail, trrm_compatibility=trrm_compatibility,
                qmae_q90=qmae_q90, qmae_quality=qmae_quality, eqm_score=eqm_score,
                model_disagreement=disagreement, econ_edge=econ_edge,
                calibrated_score=calibrated_score, eligible=eligible,
                reason_codes=tuple(reasons),
                diagnostics=(
                    ("direction_probability", direction_probability),
                    ("clean_probability", clean_probability),
                    ("qmae_mean", _mean([item.qmae_mean for item in symbol_predictions])),
                    ("qmae_quantile_valid", qmae_valid),
                    ("expected_directional_return", expected_edge),
                    ("estimated_round_trip_cost", self.settings.estimated_round_trip_cost_fraction),
                ),
            ))
        return LayerOutputs(tuple(ScientificLayerName), tuple(results))

    @staticmethod
    def _unavailable(symbol: str) -> LayerResult:
        return LayerResult(
            symbol=symbol, side=TradeSide.NO_TRADE, regime=Regime.UNKNOWN, regime_confidence=0.0,
            rv2_tail_risk=1.0, trrm_compatibility=0.0, qmae_q90=None, qmae_quality=0.0,
            eqm_score=0.0, model_disagreement=1.0, econ_edge=-1.0, calibrated_score=0.0,
            eligible=False, reason_codes=(ReasonCode.MODEL_UNAVAILABLE,),
        )


def classify_market_regime(features: dict[str, float]) -> tuple[Regime, float]:
    """Classify market context without claiming the historical D3 data discipline."""
    trend = features["market_direction_6"]
    volatility = features["range_mean_24"]
    expansion = features["range_expansion"]
    chop = features["chop_12"]
    if volatility > 0.035 or expansion > 1.0:
        regime = Regime.HIGH_VOLATILITY
    elif chop > 0.70:
        regime = Regime.RANGE
    elif trend > 0.002:
        regime = Regime.BULL_TREND
    elif trend < -0.002:
        regime = Regime.BEAR_TREND
    else:
        regime = Regime.TRANSITION
    confidence = _clip(0.50 + min(0.45, abs(trend) * 25.0 + abs(expansion) * 0.10 + abs(chop - 0.5) * 0.20))
    return regime, confidence
