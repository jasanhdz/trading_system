"""Scientific brain contracts for the Aegis clean rebuild.

This package intentionally exposes structure only. Scientific algorithms,
artifact loading, persistence, and transport remain TODO.
"""

from .domain import BrainManifest, DecisionRequest, DecisionResponse

__all__ = ["BrainManifest", "DecisionRequest", "DecisionResponse"]
