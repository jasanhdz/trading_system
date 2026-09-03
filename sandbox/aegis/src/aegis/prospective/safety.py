"""Fail-closed historical E5 boundary for prospective tooling."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping


HISTORICAL_CLOSURE = "HISTORICAL_E5_NON_EXECUTABLE_MISSING_CONTEMPORANEOUS_ROW_TARGETS"


class HistoricalE5ClosureError(RuntimeError):
    pass


def load_historical_closure(path: Path) -> Mapping[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise HistoricalE5ClosureError("E5_HISTORICAL_CLOSURE_INVALID") from exc
    if (
        value.get("classification") != HISTORICAL_CLOSURE
        or value.get("status") != "CLOSED_NON_EXECUTABLE"
        or value.get("lockbox") != "NOT_CONSUMED"
        or value.get("consumed_queries") != []
        or value.get("budget_remaining") != 1
    ):
        raise HistoricalE5ClosureError("E5_HISTORICAL_CLOSURE_INVALID")
    return value


def deny_historical_e5_stage(stage: str) -> None:
    normalized = stage.strip().upper()
    if normalized in {"DISCOVERY", "CONFIRMATION"}:
        raise HistoricalE5ClosureError("E5_HISTORICAL_EXECUTION_CLOSED")
    raise HistoricalE5ClosureError("E5_HISTORICAL_STAGE_UNSUPPORTED")
