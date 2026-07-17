"""Hashing boundary used by manifests, evidence, and decision freeze."""

from typing import Protocol


class HashProvider(Protocol):
    def digest_bytes(self, payload: bytes) -> str:
        """TODO: return a canonical digest for immutable contract payloads."""
        ...
