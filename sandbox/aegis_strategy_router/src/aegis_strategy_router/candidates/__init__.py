"""Deterministic, label-free Phase 2 candidate contracts and generators."""

from aegis_strategy_router.candidates.contracts import (
    CandidateEvaluation,
    CandidateSetup,
    CandidateStatus,
    CandidateSubstate,
    FrozenDecisionGap,
    Strategy,
    SubstateDisposition,
)
from aegis_strategy_router.candidates.registry import CandidateGeneratorRegistry

__all__ = [
    "CandidateEvaluation",
    "CandidateSetup",
    "CandidateGeneratorRegistry",
    "CandidateStatus",
    "CandidateSubstate",
    "FrozenDecisionGap",
    "Strategy",
    "SubstateDisposition",
]
