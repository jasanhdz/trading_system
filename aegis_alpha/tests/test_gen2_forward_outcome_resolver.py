#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import aegis_alpha.tools.gen2_canary_core as core
import aegis_alpha.tools.gen2_forward_outcome_resolver as resolver

CID = "gen2-20260711T202935Z"


def setup(tmp: Path) -> None:
    core.CANARY_ROOT = tmp / "live_canary"
    core.FREEZE_PATH = tmp / "freeze.json"
    resolver.execv2.core.CANARY_ROOT = core.CANARY_ROOT
    resolver.execv2.core.FREEZE_PATH = core.FREEZE_PATH
    core.FREEZE_PATH.write_text(json.dumps({"candidate_id": CID, "trrm_v2_sha256": "a", "eqm1_sha256": "b", "d3_dataset_sha256": "c", "feature_hash": "d"}))
    core.init_canary(CID)


def test_outcome_resolver_uses_only_mature_h12_and_never_changes_policy() -> None:
    with tempfile.TemporaryDirectory() as t:
        setup(Path(t))
        now = pd.Timestamp("2026-07-12T02:00:00Z")
        result = resolver.resolve_records(CID, [
            {"timestamp": "2026-07-12T00:30:00Z"},
            {"timestamp": "2026-07-12T01:30:00Z"},
        ], now)
        assert result["mature_h12"] == 1
        assert result["unresolved_immature"] == 1
        assert result["policy_changed"] is False
        assert result["models_retrained"] is False
        assert result["orders_submitted"] == 0


if __name__ == "__main__":
    test_outcome_resolver_uses_only_mature_h12_and_never_changes_policy()
    print("test_gen2_forward_outcome_resolver: OK")
