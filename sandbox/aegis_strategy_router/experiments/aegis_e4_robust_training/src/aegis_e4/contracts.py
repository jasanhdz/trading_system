"""Frozen E4 feature, label, split, and leakage contracts."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


FEATURE_PREFIX = "feature__"
TARGET_PREFIX = "target__"
FORBIDDEN_FEATURE_TOKENS = (
    "target__", "future", "outcome", "realized", "pnl", "exit_reason",
    "mfe", "mae", "barrier_first", "time_to_favorable", "time_to_adverse",
)


def load_config(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def stable_hash(payload: Any) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode()
    return hashlib.sha256(encoded).hexdigest()


def feature_schema(feature_families: dict[str, str]) -> dict[str, Any]:
    payload = {
        "schema": "aegis-e4-feature-contract-v1",
        "features": [
            {"name": name, "family": family, "available": "AT_OR_BEFORE_DECISION_AT"}
            for name, family in sorted(feature_families.items())
        ],
        "missing_policy": "EXPLICIT_FLAGS_AND_FAIL_CLOSED_FOR_REQUIRED_BASE",
        "forbidden_tokens": list(FORBIDDEN_FEATURE_TOKENS),
    }
    payload["sha256"] = stable_hash(payload)
    return payload


def assert_feature_allowlist(columns: list[str]) -> None:
    invalid = [name for name in columns if not name.startswith(FEATURE_PREFIX)]
    leaked = [
        name for name in columns
        if any(token in name.lower() for token in FORBIDDEN_FEATURE_TOKENS)
    ]
    if invalid:
        raise ValueError(f"NON_ALLOWLIST_FEATURES:{sorted(invalid)}")
    if leaked:
        raise ValueError(f"FUTURE_LEAKAGE_FEATURES:{sorted(leaked)}")


def split_for(timestamp: Any, splits: dict[str, list[str]]) -> str:
    import pandas as pd

    value = pd.Timestamp(timestamp)
    for name, bounds in splits.items():
        if pd.Timestamp(bounds[0]) <= value < pd.Timestamp(bounds[1]):
            return name
    return "OUTSIDE"
