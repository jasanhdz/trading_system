"""Orchestration of the complete scientific decision pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .config import BrainConfig, load_brain_config
from .decision import DecisionFreezer, GlobalSelectionPolicy, ScientificCandidateBuilder
from .domain import BrainManifest, DecisionRequest, DecisionResponse, ScientificContext, ScientificEvidenceEvent
from .evidence import AppendOnlyEvidenceRecorder, EvidenceRecorder, InMemoryEvidenceRecorder
from .features import DeterministicFeaturePipeline, MarketSnapshotValidator
from .layers import LayerSettings, OrderedScientificLayers
from .models import DeterministicModelRuntime, load_model_bundle
from .utils import HashProvider, Sha256HashProvider, SystemUtcClock, UtcClock


@dataclass
class BrainRuntime:
    config: BrainConfig
    validator: MarketSnapshotValidator
    features: DeterministicFeaturePipeline
    models: DeterministicModelRuntime
    layers: OrderedScientificLayers
    candidate_builder: ScientificCandidateBuilder
    selection_policy: GlobalSelectionPolicy
    freezer: DecisionFreezer
    evidence: EvidenceRecorder
    hashing: HashProvider
    clock: UtcClock

    @property
    def ready(self) -> bool:
        bundle = self.models.bundle
        return bool(bundle.approved and bundle.bundle_id == self.config.models.model_bundle_id
                    and bundle.feature_hash == self.features.feature_hash
                    and bundle.symbol_set_hash == self.config.universe.symbol_set_hash
                    and bundle.timeframe == self.config.universe.timeframe)

    def manifest(self) -> BrainManifest:
        return BrainManifest(
            contract_version=self.config.contract_version, universe_id=self.config.universe.universe_id,
            symbols=self.config.universe.symbols, symbol_set_hash=self.config.universe.symbol_set_hash,
            timeframe=self.config.universe.timeframe, config_version=self.config.config_version,
            config_hash=self.config.config_hash, model_bundle_id=self.models.bundle_id,
            feature_schema_version=self.features.schema_version, feature_hash=self.features.feature_hash,
            capabilities=("SCIENTIFIC_EVALUATION", "GLOBAL_SELECTION", "DECISION_FREEZE", "OUTCOME_EVIDENCE"),
            build_id=self.config.build_id, ready=self.ready,
        )

    def evaluate(self, request: DecisionRequest) -> DecisionResponse:
        if request.contract_version != self.config.contract_version or request.config_version != self.config.config_version:
            raise ValueError("request contract or configuration version mismatch")
        if not self.ready:
            raise RuntimeError("scientific brain is not ready")
        now = self.clock.now()
        self.validator.validate(request.snapshot, now)
        features = self.features.transform(request.snapshot)
        predictions = self.models.predict(features)
        context = ScientificContext(request.request_id, request.decision_cycle_id, request.snapshot.closed_at,
                                    request.snapshot.timeframe, request.snapshot.portfolio, features)
        layers = self.layers.apply(predictions, context)
        candidates = self.candidate_builder.build(request.decision_cycle_id, predictions, layers)
        selection = self.selection_policy.select(candidates, request.snapshot.portfolio, now)
        evidence_payload = {
            "request_hash": self.hashing.digest_value(request), "validation": "VALID",
            "feature_batch": features, "predictions": predictions, "layers": layers,
            "candidates": candidates, "selection": selection,
        }
        evidence_hash = self.hashing.digest_value(evidence_payload)
        frozen = self.freezer.freeze(
            selection, decision_cycle_id=request.decision_cycle_id, generated_at=request.snapshot.closed_at,
            model_bundle_id=predictions.bundle_id, feature_hash=features.feature_hash,
            config_hash=self.config.config_hash, evidence_hash=evidence_hash,
        )
        response = DecisionResponse(
            contract_version=self.config.contract_version, decision_id=frozen.decision_id,
            decision_cycle_id=frozen.decision_cycle_id, generated_at=frozen.generated_at,
            expires_at=frozen.expires_at, status=frozen.status,
            universe_id=self.config.universe.universe_id, symbol_set_hash=self.config.universe.symbol_set_hash,
            config_version=self.config.config_version, model_bundle_id=frozen.model_bundle_id,
            feature_schema_version=features.schema_version, evidence_hash=evidence_hash,
            selected=frozen.selected, ranking=frozen.ranking, reason_codes=frozen.reason_codes,
        )
        self.evidence.record(ScientificEvidenceEvent(
            event_id=f"decision-{evidence_hash[:24]}", decision_id=frozen.decision_id,
            decision_cycle_id=request.decision_cycle_id, event_type="DECISION_EVALUATED", occurred_at=now,
            payload={**evidence_payload, "frozen": frozen, "response": response},
        ))
        return response


def build_runtime(config_dir: Path, *, clock: UtcClock | None = None, persist_evidence: bool | None = None) -> BrainRuntime:
    """Composition root; freezes configuration and bundle for the runtime lifetime."""
    config = load_brain_config(config_dir)
    hashing = Sha256HashProvider()
    bundle_path = config.models.artifact_registry / f"{config.models.model_bundle_id}.json"
    bundle = load_model_bundle(bundle_path, expected_bundle_id=config.models.model_bundle_id)
    if bundle.universe_id != config.universe.universe_id or bundle.feature_schema_version != config.models.feature_schema_version:
        raise ValueError("bundle does not match configured universe or feature schema")
    features = DeterministicFeaturePipeline(normalizer=bundle.normalizer)
    use_persistence = config.persistence_enabled if persist_evidence is None else persist_evidence
    evidence = AppendOnlyEvidenceRecorder(hashing, config.evidence_path) if use_persistence else InMemoryEvidenceRecorder(hashing)
    settings = LayerSettings(
        trrm_max_tail_probability=config.models.trrm_max_tail_probability,
        qmae_max_fraction=config.models.qmae_max_fraction, eqm_min_score=config.models.eqm_min_score,
        estimated_round_trip_cost_fraction=config.models.estimated_round_trip_cost_fraction,
        direction_threshold=config.models.direction_threshold,
    )
    return BrainRuntime(
        config=config, validator=MarketSnapshotValidator(config.universe), features=features,
        models=DeterministicModelRuntime(bundle, config.models.direction_threshold),
        layers=OrderedScientificLayers(settings), candidate_builder=ScientificCandidateBuilder(hashing),
        selection_policy=GlobalSelectionPolicy(config.models.selection_threshold),
        freezer=DecisionFreezer(hashing, config.models.maximum_decision_age_seconds),
        evidence=evidence, hashing=hashing, clock=clock or SystemUtcClock(),
    )
