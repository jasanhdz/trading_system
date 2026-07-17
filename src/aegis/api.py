"""Transport-neutral signatures for the future minimal Python API."""

from dataclasses import dataclass
from typing import Mapping

from .domain import BrainManifest, DecisionOutcome, DecisionRequest, DecisionResponse
from .runtime import BrainRuntime


@dataclass
class BrainApi:
    """Declares endpoint behavior without binding an HTTP framework."""

    runtime: BrainRuntime

    def health(self) -> Mapping[str, str]:
        """TODO: expose liveness only."""
        raise NotImplementedError("Health transport is not implemented")

    def ready(self) -> Mapping[str, str | bool]:
        """TODO: expose configuration, bundle, feature, and universe readiness."""
        raise NotImplementedError("Readiness transport is not implemented")

    def manifest(self) -> BrainManifest:
        """TODO: expose the versioned compatibility handshake."""
        raise NotImplementedError("Manifest transport is not implemented")

    def evaluate(self, request: DecisionRequest) -> DecisionResponse:
        """TODO: delegate a validated request to BrainRuntime."""
        raise NotImplementedError("Decision transport is not implemented")

    def submit_outcome(self, outcome: DecisionOutcome) -> None:
        """TODO: accept normalized evidence without operational authority."""
        raise NotImplementedError("Outcome transport is not implemented")
