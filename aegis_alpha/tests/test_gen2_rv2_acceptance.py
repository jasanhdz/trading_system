#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import aegis_alpha.tools.gen2_rv2_acceptance as acc
import aegis_alpha.tools.gen2_rv2_train as rv2
from test_gen2_rv2_train import make_dense, small_candidates, write_lockbox


def run_fixture_training(tmp: Path):
    dense = make_dense(tmp)
    lockbox = write_lockbox(tmp)
    rv2.LOCKBOX_MANIFEST_PATH = lockbox
    rv2.RV2_ROOT = tmp / "rv2"
    original = rv2.trrm_candidates
    rv2.trrm_candidates = small_candidates
    try:
        payload = rv2.run_training(rv2.parse_args(["--dataset-csv", str(dense)]))
    finally:
        rv2.trrm_candidates = original
    return payload, lockbox


def test_acceptance_decision_and_lockbox_query() -> None:
    with tempfile.TemporaryDirectory() as t:
        tmp = Path(t)
        payload, lockbox = run_fixture_training(tmp)
        artifact_dir = Path(payload["frozen_candidate"]["pickle"]).parent
        # patch the candidate factory so the semi-blind reload path matches training
        result = acc.run_acceptance(acc.parse_args(["--artifact-dir", str(artifact_dir), "--lockbox-path", str(lockbox)]))
        assert result["decision"] in {"GEN2_TRRM_READY", "GEN2_TRRM_PROMISING", "GEN2_TRRM_PARTIAL"}
        assert result["hypotheses"]["H1_signal"] is True
        assert result["semi_blind"]["status"] == "QUERY_EXECUTED_AND_LOGGED"
        box = json.loads(lockbox.read_text())
        assert box["current_query_count"] == 1
        assert len(box["query_log"]) == 1
        assert box["query_log"][0]["candidate_hash"] == payload["frozen_candidate"]["pickle_sha256"]
        # second acceptance run must NOT get a second query for the same candidate
        result2 = acc.run_acceptance(acc.parse_args(["--artifact-dir", str(artifact_dir), "--lockbox-path", str(lockbox)]))
        assert result2["semi_blind"]["status"] == "QUERY_BUDGET_EXHAUSTED"
        box2 = json.loads(lockbox.read_text())
        assert box2["current_query_count"] == 1


def test_rejected_when_no_winner() -> None:
    with tempfile.TemporaryDirectory() as t:
        tmp = Path(t)
        payload, lockbox = run_fixture_training(tmp)
        artifact_dir = Path(payload["frozen_candidate"]["pickle"]).parent
        report = json.loads((artifact_dir / "training_report.json").read_text())
        report["trrm_winner"] = None
        (artifact_dir / "training_report.json").write_text(json.dumps(report, default=str))
        result = acc.run_acceptance(acc.parse_args(["--artifact-dir", str(artifact_dir), "--lockbox-path", str(lockbox), "--skip-semi-blind"]))
        assert result["decision"] == "GEN2_TRRM_REJECTED"


if __name__ == "__main__":
    test_acceptance_decision_and_lockbox_query()
    test_rejected_when_no_winner()
    print("test_gen2_rv2_acceptance: OK")
