"""Approved artifact publication and loading boundary.

TODO migration reference: retain checksum and immutable-bundle ideas from the
prior Gen2 freeze tooling, but design a fresh registry contract before use.
"""

from typing import Protocol

from .evaluate import EvaluationReport
from .train import ModelArtifact


class ArtifactRegistry(Protocol):
    def publish(
        self,
        artifact: ModelArtifact,
        evaluation: EvaluationReport,
    ) -> str:
        """TODO: publish an immutable approved bundle and return its ID."""
        ...

    def load(self, bundle_id: str) -> ModelArtifact:
        """TODO: verify checksums and load one immutable approved bundle."""
        ...
