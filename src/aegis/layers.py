"""Ordered D3, RV2, TRRM, QMAE, EQM, and ECON1 layer boundary.

TODO migration references: verify semantics against the prior Gen2 D3/RV2,
EQM1, and ECON1 specs and tools on ``feature/wraith-phantom-v8``. No formula
from that branch is part of this scaffold.
"""

from typing import Protocol

from .domain import LayerOutputs, ModelPredictions, ScientificContext


class ScientificLayers(Protocol):
    def apply(
        self,
        predictions: ModelPredictions,
        context: ScientificContext,
    ) -> LayerOutputs:
        """TODO: migrate verified layer semantics from the prior research branch."""
        ...
