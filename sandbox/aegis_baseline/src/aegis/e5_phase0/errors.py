"""Stable fail-closed errors for E5 Phase 0."""

from __future__ import annotations


class Phase0Error(RuntimeError):
    """An engineering contract violation with a stable failure code."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(f"{code}: {message}")
        self.code = code
        self.message = message
