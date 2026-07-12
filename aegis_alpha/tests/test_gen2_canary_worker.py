#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import aegis_alpha.tools.gen2_canary_core as core
import aegis_alpha.tools.gen2_canary_worker as worker

CID = "gen2-20260711T202935Z"


def setup(tmp: Path) -> None:
    core.CANARY_ROOT = tmp / "live_canary"
    core.FREEZE_PATH = tmp / "freeze.json"
    worker.core.CANARY_ROOT = core.CANARY_ROOT
    worker.core.FREEZE_PATH = core.FREEZE_PATH
    worker.execv2.core.CANARY_ROOT = core.CANARY_ROOT
    worker.execv2.core.FREEZE_PATH = core.FREEZE_PATH
    core.FREEZE_PATH.write_text(json.dumps({"candidate_id": CID, "trrm_v2_sha256": "a", "eqm1_sha256": "b", "d3_dataset_sha256": "c", "feature_hash": "d"}))


def test_worker_starts_unarmed_and_submits_no_orders() -> None:
    with tempfile.TemporaryDirectory() as t:
        setup(Path(t))
        result = worker.run_once(CID)
        assert result["armed"] is False
        assert result["real_order_submission_enabled"] is False
        assert result["status"] == "NO_OPPORTUNITY"


def test_worker_records_paper_live_parity_without_order_when_unarmed() -> None:
    with tempfile.TemporaryDirectory() as t:
        setup(Path(t))
        result = worker.run_once(CID, {"opportunity_id": "o1", "signal_id": "s1", "timestamp": "2026-07-12T00:00:00Z",
                                       "symbol": "ADAUSDT", "features": {"a": 1}, "paper_decision": "CANDIDATE_SHORT"})
        assert result["status"] == "UNARMED_NO_ORDER"
        assert result["enforcement_action"] == "NONE"
        assert "feature_snapshot_hash" in result
        assert (core.canary_dir(CID) / "paper_live_parity.jsonl").exists()


if __name__ == "__main__":
    test_worker_starts_unarmed_and_submits_no_orders()
    test_worker_records_paper_live_parity_without_order_when_unarmed()
    print("test_gen2_canary_worker: OK")
