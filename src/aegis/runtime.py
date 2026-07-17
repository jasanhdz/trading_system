"""Declarative orchestration point for the scientific pipeline."""

from dataclasses import dataclass

from .config import BrainConfig
from .decision import CandidateBuilder, SelectionPolicy
from .domain import DecisionRequest, DecisionResponse
from .evidence import EvidenceRecorder
from .features import FeaturePipeline
from .layers import ScientificLayers
from .models import ModelRuntime
from .training.registry import ArtifactRegistry
from .utils import HashProvider, UtcClock


@dataclass
class BrainRuntime:
    """Wires the future pipeline without providing scientific behavior."""

    config: BrainConfig
    features: FeaturePipeline
    models: ModelRuntime
    layers: ScientificLayers
    candidate_builder: CandidateBuilder
    selection_policy: SelectionPolicy
    evidence: EvidenceRecorder
    artifacts: ArtifactRegistry
    hashing: HashProvider
    clock: UtcClock

    def evaluate(self, request: DecisionRequest) -> DecisionResponse:
        """TODO: validate, transform, predict, layer, select, freeze, and record."""
        raise NotImplementedError("Scientific runtime orchestration is not implemented")
