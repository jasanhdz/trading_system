"""UTC clock abstractions for deterministic runtime and tests."""

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Protocol


class UtcClock(Protocol):
    def now(self) -> datetime: ...


class SystemUtcClock:
    def now(self) -> datetime:
        return datetime.now(timezone.utc)


@dataclass(frozen=True)
class FixedUtcClock:
    value: datetime

    def __post_init__(self) -> None:
        if self.value.tzinfo is None or self.value.utcoffset() is None:
            raise ValueError("FixedUtcClock requires a timezone-aware datetime")

    def now(self) -> datetime:
        return self.value.astimezone(timezone.utc)
