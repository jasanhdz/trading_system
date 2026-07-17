"""Feature-pipeline boundary for deterministic scientific transformations."""

from typing import Protocol

from .domain import FeatureBatch, MarketSnapshot


class FeaturePipeline(Protocol):
    def transform(self, snapshot: MarketSnapshot) -> FeatureBatch:
        """TODO: produce deterministic, versioned features without market I/O."""
        ...
