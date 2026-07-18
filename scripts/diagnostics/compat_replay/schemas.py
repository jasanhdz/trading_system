"""Closed diagnostic replay contracts."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import yaml


REPLAY_ID = "aegis-gen2-compatibility-replay-v1"
STAGES = ("STAGE_0", "STAGE_1", "STAGE_2", "STAGE_3", "STAGE_4", "STAGE_5")
FORBIDDEN_OUTPUT_KINDS = {"CANDIDATE", "SELECTION_POLICY", "SYSTEM_FREEZE"}


@dataclass(frozen=True)
class ReplayConfig:
    path: Path
    payload: Mapping[str, Any]

    @classmethod
    def load(cls, path: Path) -> "ReplayConfig":
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
        required = {"schema_version", "replay_id", "lifecycle", "scope", "inputs", "stages", "reference", "output_root"}
        if not isinstance(payload, Mapping) or set(payload) != required:
            raise ValueError("COMPAT_REPLAY_CONFIG_INVALID: unexpected or missing top-level keys")
        if payload["replay_id"] != REPLAY_ID or payload["lifecycle"] != "DIAGNOSTIC_DEV_ONLY":
            raise ValueError("COMPAT_REPLAY_CONFIG_INVALID: identity/lifecycle")
        if tuple(payload["stages"]) != STAGES:
            raise ValueError("COMPAT_REPLAY_CONFIG_INVALID: stages are a closed ordered list")
        scope = payload["scope"]
        if scope != {"dev_end": "2026-04-26T23:59:59Z", "semi_blind": "FORBIDDEN", "lockbox": "FORBIDDEN"}:
            raise ValueError("COMPAT_REPLAY_CONFIG_INVALID: scope")
        return cls(path.resolve(), payload)
