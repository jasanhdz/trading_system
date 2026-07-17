"""Candidate construction, global selection, and decision-freeze boundaries.

TODO migration references: study ``gen2_selection_policy.py`` and
``gen2_system_freeze.py`` on ``feature/wraith-phantom-v8`` for verified
selection and freeze semantics; do not recover their operational tooling.
"""

from typing import Protocol

from .domain import (
    CandidateSet,
    DecisionRequest,
    DecisionResponse,
    FrozenDecision,
    LayerOutputs,
    ModelPredictions,
    PortfolioContext,
)


class CandidateBuilder(Protocol):
    def build(
        self,
        predictions: ModelPredictions,
        layers: LayerOutputs,
    ) -> CandidateSet:
        """TODO: build one scientific candidate per symbol."""
        ...


class SelectionPolicy(Protocol):
    def select(
        self,
        candidates: CandidateSet,
        context: PortfolioContext,
    ) -> FrozenDecision:
        """TODO: rank globally, select deterministically, and freeze once."""
        ...


class DecisionEngine(Protocol):
    def evaluate(self, request: DecisionRequest) -> DecisionResponse:
        """TODO: execute the complete scientific pipeline."""
        ...
