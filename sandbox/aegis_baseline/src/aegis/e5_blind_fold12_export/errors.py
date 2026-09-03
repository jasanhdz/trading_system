"""Stable fail-closed errors for E5 Phase 1A."""

from __future__ import annotations


class BlindExportError(RuntimeError):
    """A clean-room-safe contract failure.

    Messages must contain field names, source ordinals, hashes, or paths only.
    Scientific values and serialized source payloads are forbidden.
    """

    def __init__(self, code: str, message: str) -> None:
        super().__init__(f"{code}: {message}")
        self.code = code
        self.message = message


class BlindExportInterrupted(BlindExportError):
    """A deterministic test interruption with a sealed checkpoint."""

    def __init__(self, message: str = "sealed checkpoint available") -> None:
        super().__init__("E5_BLIND_EXPORT_INTERRUPTED", message)
