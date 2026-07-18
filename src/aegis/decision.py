"""Candidate construction, global selection, and immutable decision freeze."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from time import perf_counter
from typing import Callable, Protocol

from .domain import (
    Candidate, CandidateSet, DecisionStatus, FeatureBatch, FrozenDecision, LayerOutputs,
    ModelPredictions, PortfolioContext, RankedCandidate, ReasonCode, RiskIntent,
    ScientificContext, SelectionResult, TradeSide,
)
from .layers import ScientificLayers
from .models import ModelRuntime
from .utils import HashProvider


class CandidateBuilder(Protocol):
    def build(self, decision_cycle_id: str, predictions: ModelPredictions, layers: LayerOutputs) -> CandidateSet: ...


class SelectionPolicy(Protocol):
    def select(self, candidates: CandidateSet, context: PortfolioContext, now: datetime) -> SelectionResult: ...


@dataclass(frozen=True)
class ScientificPipelineResult:
    """Complete authoritative result before decision freeze and evidence persistence."""

    predictions: ModelPredictions
    layers: LayerOutputs
    candidates: CandidateSet
    selection: SelectionResult


def evaluate_scientific_pipeline(
    *, model_runtime: ModelRuntime, scientific_layers: ScientificLayers,
    candidate_builder: CandidateBuilder, selection_policy: SelectionPolicy,
    request_id: str, decision_cycle_id: str, closed_at: datetime, timeframe: str,
    portfolio: PortfolioContext, features: FeatureBatch, now: datetime,
    stage_observer: Callable[[str, float], None] | None = None,
) -> ScientificPipelineResult:
    """Single training/serving path for every scientific transformation and gate."""
    stage = perf_counter()
    predictions = model_runtime.predict(features)
    if stage_observer is not None:
        stage_observer("models", perf_counter() - stage)
    context = ScientificContext(request_id, decision_cycle_id, closed_at, timeframe, portfolio, features)
    stage = perf_counter()
    layers = scientific_layers.apply(predictions, context)
    if stage_observer is not None:
        stage_observer("layers", perf_counter() - stage)
    stage = perf_counter()
    candidates = candidate_builder.build(decision_cycle_id, predictions, layers)
    if stage_observer is not None:
        stage_observer("candidates", perf_counter() - stage)
    stage = perf_counter()
    selection = selection_policy.select(candidates, portfolio, now)
    if stage_observer is not None:
        stage_observer("selection", perf_counter() - stage)
    return ScientificPipelineResult(predictions, layers, candidates, selection)


@dataclass(frozen=True)
class ScientificCandidateBuilder:
    hashing: HashProvider

    def build(self, decision_cycle_id: str, predictions: ModelPredictions, layers: LayerOutputs) -> CandidateSet:
        candidates: list[Candidate] = []
        for result in sorted(layers.results, key=lambda item: item.symbol):
            symbol_predictions = predictions.for_symbol(result.symbol)
            if not symbol_predictions:
                continue
            directional = [
                item.long_probability if result.side is TradeSide.LONG else item.short_probability
                for item in symbol_predictions
            ]
            raw_score = sum(directional) / len(directional) if result.side is not TradeSide.NO_TRADE else 0.0
            expected = next((float(value) for key, value in result.diagnostics if key == "expected_directional_return"), 0.0)
            payload = {
                "cycle": decision_cycle_id, "symbol": result.symbol, "side": result.side,
                "raw_score": raw_score, "calibrated_score": result.calibrated_score,
                "bundle": predictions.bundle_id, "feature_hash": predictions.feature_hash,
                "layers": result,
            }
            candidate_hash = self.hashing.digest_value(payload)
            candidates.append(Candidate(
                candidate_id=f"candidate-{candidate_hash[:20]}", symbol=result.symbol, side=result.side,
                raw_score=raw_score, calibrated_score=result.calibrated_score,
                confidence=max(0.0, min(1.0, 1.0 - result.model_disagreement)),
                uncertainty=max(0.0, min(1.0, result.model_disagreement)), regime=result.regime,
                compatibility=result.trrm_compatibility, expected_return=expected,
                horizon_bars=symbol_predictions[0].horizon_bars,
                risk_intent=RiskIntent(
                    stop_distance_fraction=None,
                    target_distance_fraction=max(0.0, expected),
                    volatility_multiple=None, target_risk_ratio=None,
                    maximum_holding_bars=symbol_predictions[0].horizon_bars,
                    scientific_invalidation="REGIME/RV2/TRRM/QMAE/EQM/ECON1 contract invalidated",
                    relative_priority=result.calibrated_score,
                ),
                reason_codes=result.reason_codes,
                evidence_references=(f"features:{predictions.feature_hash}", f"bundle:{predictions.bundle_id}"),
                model_bundle_id=predictions.bundle_id, feature_hash=predictions.feature_hash,
                candidate_hash=candidate_hash, eligible=result.eligible,
            ))
        return CandidateSet(decision_cycle_id, tuple(candidates))


@dataclass(frozen=True)
class GlobalSelectionPolicy:
    selection_threshold: float
    maximum_selected: int = 1

    def select(self, candidates: CandidateSet, context: PortfolioContext, now: datetime) -> SelectionResult:
        evaluated: list[Candidate] = []
        for candidate in candidates.candidates:
            reasons = list(candidate.reason_codes)
            eligible = candidate.eligible
            if candidate.side is not TradeSide.SHORT:
                eligible = False
                reasons.append(ReasonCode.SIDE_NOT_ENABLED)
            if candidate.calibrated_score < self.selection_threshold:
                eligible = False
                reasons.append(ReasonCode.NO_TRADE_THRESHOLD)
            if candidate.symbol in context.blocked_symbols:
                eligible = False
                reasons.append(ReasonCode.SYMBOL_BLOCKED)
            if candidate.symbol in context.occupied_symbols:
                eligible = False
                reasons.append(ReasonCode.SYMBOL_OCCUPIED)
            cooldown = context.active_cooldowns.get(candidate.symbol)
            if cooldown is not None and cooldown > now:
                eligible = False
                reasons.append(ReasonCode.ACTIVE_COOLDOWN)
            evaluated.append(Candidate(**{**candidate.__dict__, "eligible": eligible, "reason_codes": tuple(dict.fromkeys(reasons))}))

        ordered = sorted(evaluated, key=lambda item: (-item.calibrated_score, item.symbol, item.candidate_hash))
        ranking = tuple(RankedCandidate(
            rank=index + 1, symbol=item.symbol, candidate_hash=item.candidate_hash,
            score=item.calibrated_score, eligible=item.eligible, reason_codes=item.reason_codes,
        ) for index, item in enumerate(ordered))
        if context.available_slots <= 0:
            return SelectionResult(DecisionStatus.NO_TRADE, (), ranking, (ReasonCode.NO_AVAILABLE_SLOT,))
        selected = tuple(item for item in ordered if item.eligible)[: min(self.maximum_selected, context.available_slots)]
        if not selected:
            return SelectionResult(DecisionStatus.NO_TRADE, (), ranking, (ReasonCode.NO_TRADE_NO_CANDIDATE,))
        return SelectionResult(DecisionStatus.SELECTED, selected, ranking, (ReasonCode.ELIGIBLE,))


@dataclass(frozen=True)
class DecisionFreezer:
    hashing: HashProvider
    maximum_age_seconds: int

    def freeze(
        self, selection: SelectionResult, *, decision_cycle_id: str, generated_at: datetime,
        model_bundle_id: str, feature_hash: str, config_hash: str, evidence_hash: str,
    ) -> FrozenDecision:
        expires_at = generated_at + timedelta(seconds=self.maximum_age_seconds)
        base = {
            "decision_cycle_id": decision_cycle_id, "generated_at": generated_at,
            "expires_at": expires_at, "status": selection.status, "selected": selection.selected,
            "ranking": selection.ranking, "reason_codes": selection.reason_codes,
            "model_bundle_id": model_bundle_id, "feature_hash": feature_hash,
            "config_hash": config_hash, "evidence_hash": evidence_hash,
        }
        decision_hash = self.hashing.digest_value(base)
        return FrozenDecision(
            decision_id=f"decision-{decision_hash[:24]}", decision_cycle_id=decision_cycle_id,
            generated_at=generated_at, expires_at=expires_at, status=selection.status,
            selected=selection.selected, ranking=selection.ranking, reason_codes=selection.reason_codes,
            model_bundle_id=model_bundle_id, feature_hash=feature_hash, config_hash=config_hash,
            evidence_hash=evidence_hash, decision_hash=decision_hash,
        )
