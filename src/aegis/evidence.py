"""Append-only scientific evidence boundary.

TODO migration reference: inspect prior forward collectors and the Gen2
outcome resolver only to define a normalized evidence schema.
"""

from typing import Protocol

from .domain import DecisionOutcome, ScientificEvidenceEvent


class EvidenceRecorder(Protocol):
    def record(self, event: ScientificEvidenceEvent) -> None:
        """TODO: persist normalized scientific evidence append-only."""
        ...

    def record_outcome(self, outcome: DecisionOutcome) -> None:
        """TODO: link normalized TypeScript outcomes without changing policy."""
        ...
