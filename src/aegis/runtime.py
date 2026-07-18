"""Orchestration of the complete scientific decision pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from threading import Lock
from time import perf_counter
from typing import Mapping

from .config import BrainConfig, load_brain_config
from .decision import (
    DecisionFreezer, GlobalSelectionPolicy, ScientificCandidateBuilder,
    evaluate_scientific_pipeline,
)
from .domain import BrainManifest, DecisionRequest, DecisionResponse, ScientificEvidenceEvent
from .evidence import AppendOnlyEvidenceRecorder, EvidenceRecorder, InMemoryEvidenceRecorder
from .features import DeterministicFeaturePipeline, MarketSnapshotValidator
from .layers import LayerSettings, OrderedScientificLayers
from .models import DeterministicModelRuntime, load_model_bundle
from .utils import HashProvider, Sha256HashProvider, SystemUtcClock, UtcClock


class RuntimeMetrics:
    """In-process scientific counters; values never influence a decision."""

    def __init__(self) -> None:
        self._lock = Lock()
        self._counters: dict[str, int] = {"requests": 0, "errors": 0, "no_trade": 0, "selected": 0, "outcomes": 0}
        self._latency_seconds: dict[str, float] = {}
        self._stage_calls: dict[str, int] = {}

    def increment(self, name: str) -> None:
        with self._lock:
            self._counters[name] = self._counters.get(name, 0) + 1

    def latency(self, stage: str, seconds: float) -> None:
        with self._lock:
            self._latency_seconds[stage] = self._latency_seconds.get(stage, 0.0) + seconds
            self._stage_calls[stage] = self._stage_calls.get(stage, 0) + 1

    def snapshot(self) -> Mapping[str, Mapping[str, float | int]]:
        with self._lock:
            means = {stage: self._latency_seconds[stage] / self._stage_calls[stage] for stage in self._latency_seconds}
            return {"counters": dict(self._counters), "mean_latency_seconds": means}


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
    metrics: RuntimeMetrics = field(default_factory=RuntimeMetrics)
    _response_cache: dict[str, DecisionResponse] = field(default_factory=dict, init=False, repr=False)
    _cache_lock: Lock = field(default_factory=Lock, init=False, repr=False)

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
        started = perf_counter()
        self.metrics.increment("requests")
        request_hash = self.hashing.digest_value({
            "request_id": request.request_id, "decision_cycle_id": request.decision_cycle_id,
            "schema_version": request.schema_version, "contract_version": request.contract_version,
            "config_version": request.config_version,
            "snapshot": {
                "closed_at": request.snapshot.closed_at, "timeframe": request.snapshot.timeframe,
                "symbol_set_hash": request.snapshot.symbol_set_hash,
                "series": tuple(sorted(request.snapshot.series, key=lambda item: item.symbol)),
                "portfolio": request.snapshot.portfolio,
            },
        })
        with self._cache_lock:
            cached = self._response_cache.get(request_hash)
        if cached is not None:
            self.metrics.latency("total", perf_counter() - started)
            return cached
        try:
            if request.contract_version != self.config.contract_version or request.config_version != self.config.config_version:
                raise ValueError("request contract or configuration version mismatch")
            if not self.ready:
                raise RuntimeError("scientific brain is not ready")
            now = self.clock.now()
            stage = perf_counter(); self.validator.validate(request.snapshot, now); self.metrics.latency("validation", perf_counter() - stage)
            stage = perf_counter(); features = self.features.transform(request.snapshot); self.metrics.latency("features", perf_counter() - stage)
        except Exception:
            self.metrics.increment("errors")
            self.metrics.latency("total", perf_counter() - started)
            raise
        stage = perf_counter()
        pipeline = evaluate_scientific_pipeline(
            model_runtime=self.models, scientific_layers=self.layers,
            candidate_builder=self.candidate_builder, selection_policy=self.selection_policy,
            request_id=request.request_id, decision_cycle_id=request.decision_cycle_id,
            closed_at=request.snapshot.closed_at, timeframe=request.snapshot.timeframe,
            portfolio=request.snapshot.portfolio, features=features, now=now,
            stage_observer=self.metrics.latency,
        )
        self.metrics.latency("scientific_pipeline", perf_counter() - stage)
        evidence_payload = {
            "request_hash": request_hash, "validation": "VALID",
            "feature_batch": features, "predictions": pipeline.predictions, "layers": pipeline.layers,
            "candidates": pipeline.candidates, "selection": pipeline.selection,
        }
        evidence_hash = self.hashing.digest_value(evidence_payload)
        frozen = self.freezer.freeze(
            pipeline.selection, decision_cycle_id=request.decision_cycle_id, generated_at=request.snapshot.closed_at,
            model_bundle_id=pipeline.predictions.bundle_id, feature_hash=features.feature_hash,
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
        with self._cache_lock:
            self._response_cache.setdefault(request_hash, response)
            response = self._response_cache[request_hash]
        self.metrics.increment("no_trade" if response.status.value == "NO_TRADE" else "selected")
        self.metrics.latency("total", perf_counter() - started)
        return response


def build_runtime(config_dir: Path, *, clock: UtcClock | None = None, persist_evidence: bool | None = None) -> BrainRuntime:
    """Composition root; freezes configuration and bundle for the runtime lifetime."""
    config = load_brain_config(config_dir)
    hashing = Sha256HashProvider()
    bundle_path = config.models.artifact_registry / f"{config.models.model_bundle_id}.json"
    bundle = load_model_bundle(bundle_path, expected_bundle_id=config.models.model_bundle_id)
    if bundle.universe_id != config.universe.universe_id or bundle.feature_schema_version != config.models.feature_schema_version:
        raise ValueError("bundle does not match configured universe or feature schema")
    features = DeterministicFeaturePipeline(
        normalizer=bundle.normalizer, schema_version=bundle.feature_schema_version,
    )
    use_persistence = config.persistence_enabled if persist_evidence is None else persist_evidence
    evidence = AppendOnlyEvidenceRecorder(hashing, config.evidence_path) if use_persistence else InMemoryEvidenceRecorder(hashing)
    settings = LayerSettings(
        trrm_max_tail_probability=config.models.trrm_max_tail_probability,
        qmae_max_fraction=config.models.qmae_max_fraction, eqm_min_score=config.models.eqm_min_score,
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
