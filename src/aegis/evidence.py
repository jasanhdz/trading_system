"""Append-only scientific evidence with a deterministic SHA-256 chain."""

from __future__ import annotations

import json
import os
import threading
from dataclasses import replace
from datetime import datetime
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
        if self._path is not None and self._path.exists():
            self._recover()

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
        event_hash = self._hashing.digest_value({"decision_id": outcome.decision_id, "event_type": "DECISION_OUTCOME"})
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

    def _recover(self) -> None:
        assert self._path is not None
        previous: str | None = None
        seen_ids: set[str] = set()
        try:
            lines = self._path.read_text(encoding="utf-8").splitlines()
            for line_number, line in enumerate(lines, start=1):
                if not line.strip():
                    raise EvidencePersistenceError(f"blank evidence line at {line_number}")
                payload = json.loads(line)
                event = ScientificEvidenceEvent(
                    event_id=str(payload["event_id"]), decision_id=str(payload["decision_id"]),
                    decision_cycle_id=str(payload["decision_cycle_id"]), event_type=str(payload["event_type"]),
                    occurred_at=datetime.fromisoformat(str(payload["occurred_at"]).replace("Z", "+00:00")),
                    payload=payload.get("payload", {}), previous_event_hash=payload.get("previous_event_hash"),
                    event_hash=str(payload.get("event_hash", "")),
                )
                if event.event_id in seen_ids or event.previous_event_hash != previous:
                    raise EvidencePersistenceError(f"evidence chain linkage failure at line {line_number}")
                expected = self._hashing.digest_value(replace(event, event_hash=""))
                if event.event_hash != expected:
                    raise EvidencePersistenceError(f"evidence hash mismatch at line {line_number}")
                self._events.append(event); seen_ids.add(event.event_id); previous = event.event_hash
                if event.event_type == "DECISION_OUTCOME":
                    key = self._hashing.digest_value({"decision_id": event.decision_id, "event_type": event.event_type})
                    self._outcomes[key] = event
        except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
            if isinstance(exc, EvidencePersistenceError):
                raise
            raise EvidencePersistenceError("unable to recover scientific evidence chain") from exc

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
