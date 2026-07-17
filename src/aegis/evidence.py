"""Append-only scientific evidence with a deterministic SHA-256 chain."""

from __future__ import annotations

import json
import os
import threading
from dataclasses import replace
from pathlib import Path
from typing import Protocol

from .domain import DecisionOutcome, ScientificEvidenceEvent
from .utils import HashProvider, canonical_json


class EvidencePersistenceError(RuntimeError):
    pass


class EvidenceRecorder(Protocol):
    def record(self, event: ScientificEvidenceEvent) -> ScientificEvidenceEvent: ...
    def record_outcome(self, outcome: DecisionOutcome) -> ScientificEvidenceEvent: ...


class AppendOnlyEvidenceRecorder:
    """Thread-safe recorder; persistence errors are explicit and fail closed."""

    def __init__(self, hashing: HashProvider, path: Path | None = None) -> None:
        self._hashing = hashing
        self._path = path
        self._events: list[ScientificEvidenceEvent] = []
        self._outcomes: dict[str, ScientificEvidenceEvent] = {}
        self._lock = threading.RLock()

    @property
    def events(self) -> tuple[ScientificEvidenceEvent, ...]:
        return tuple(self._events)

    def record(self, event: ScientificEvidenceEvent) -> ScientificEvidenceEvent:
        with self._lock:
            previous = self._events[-1].event_hash if self._events else None
            unsigned = replace(event, previous_event_hash=previous, event_hash="")
            signed = replace(unsigned, event_hash=self._hashing.digest_value(unsigned))
            if self._path is not None:
                self._append(signed)
            self._events.append(signed)
            return signed

    def record_outcome(self, outcome: DecisionOutcome) -> ScientificEvidenceEvent:
        event_hash = self._hashing.digest_value(outcome)
        with self._lock:
            existing = self._outcomes.get(event_hash)
            if existing is not None:
                return existing
            recorded = self.record(ScientificEvidenceEvent(
                event_id=f"outcome-{event_hash[:24]}", decision_id=outcome.decision_id,
                decision_cycle_id=outcome.decision_cycle_id, event_type="DECISION_OUTCOME",
                occurred_at=outcome.occurred_at, payload={"outcome": outcome},
            ))
            self._outcomes[event_hash] = recorded
            return recorded

    def _append(self, event: ScientificEvidenceEvent) -> None:
        assert self._path is not None
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            with self._path.open("a", encoding="utf-8") as handle:
                handle.write(canonical_json(event) + "\n")
                handle.flush()
                os.fsync(handle.fileno())
        except OSError as exc:
            raise EvidencePersistenceError(f"unable to persist scientific evidence: {type(exc).__name__}") from exc


class InMemoryEvidenceRecorder(AppendOnlyEvidenceRecorder):
    def __init__(self, hashing: HashProvider) -> None:
        super().__init__(hashing, None)
