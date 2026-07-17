"""UTC clock boundary for deterministic tests and expiry handling."""

from datetime import datetime
from typing import Protocol


class UtcClock(Protocol):
    def now(self) -> datetime:
        """TODO: return a timezone-aware UTC timestamp."""
        ...
