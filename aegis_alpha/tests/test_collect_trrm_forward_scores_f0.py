#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import aegis_alpha.tools.trrm_forward_common_f0 as common
from aegis_alpha.tests.test_freeze_trrm_forward_candidate_f0 import make_artifacts
from aegis_alpha.tools.collect_trrm_forward_scores_f0 import parse_args as collect_args
from aegis_alpha.tools.collect_trrm_forward_scores_f0 import run_collect
from aegis_alpha.tools.freeze_trrm_forward_candidate_f0 import parse_args as freeze_args
from aegis_alpha.tools.freeze_trrm_forward_candidate_f0 import run_freeze


def write_signal(path: Path, signal_id: str, ts: str, symbol: str = "SOLUSDT", reason: str = "candidate") -> None:
    row = {
        "timestamp": ts,
        "signal_id": signal_id,
        "symbol": symbol,
        "strategy": "AEGIS_TURBO",
        "mode": "AEGIS_TURBO_MICRO_LIVE",
        "raw_action": "SHORT",
        "gated_action": "SHORT",
        "final_action": "SHORT",
        "reason": reason,
        "turbo_score": 0.7,
        "gate_allowed": True,
        "executed": False,
    }
    path.write_text(json.dumps(row) + "\n")


def frozen_candidate(root: Path) -> tuple[Path, Path]:
    art = make_artifacts(root)
    common.EXPECTED_FEATURE_HASH = art["feature_hash"]
    signal = root / "turbo_signals_2026-07-10.jsonl"
    write_signal(signal, "SIG-1", "2026-07-10T20:10:00Z")
    payload = run_freeze(
        freeze_args(
            [
                "--e2-report-json",
                str(art["e2"]),
                "--e21-report-json",
                str(art["e21"]),
                "--fable-audit-json",
                str(art["fable"]),
                "--model-dir",
                str(art["model_dir"]),
                "--policy-dir",
                str(art["policy_dir"]),
                "--internal-predictions",
                str(art["internal"]),
                "--output-root",
                str(root / "out"),
                "--candidate-id",
                "candidate_test",
                "--freeze-time",
                "2026-07-10T20:00:00Z",
                "--feature-hash",
                art["feature_hash"],
            ]
        )
    )
    return Path(payload["candidate_dir"]), signal


def test_no_new_opportunities_status() -> None:
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        cdir, signal = frozen_candidate(root)
        payload = run_collect(
            collect_args(
                [
                    "--candidate-dir",
                    str(cdir),
                    "--source-path",
                    str(signal),
                    "--since",
                    "2026-07-10T21:00:00Z",
                    "--until",
                    "2026-07-10T22:00:00Z",
                ]
            )
        )
        assert payload["decision"] == "F0_FROZEN_NO_NEW_OPPORTUNITIES"
        assert payload["opportunities_appended"] == 0
        assert payload["FORWARD_OUTCOMES_NOT_EVALUATED"] is True


def test_collect_append_idempotent_and_no_decision() -> None:
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        cdir, signal = frozen_candidate(root)
        args = collect_args(
            [
                "--candidate-dir",
                str(cdir),
                "--source-path",
                str(signal),
                "--since",
                "2026-07-10T20:00:00Z",
                "--until",
                "2026-07-10T20:30:00Z",
            ]
        )
        first = run_collect(args)
        assert first["decision"] == "F0_COLLECTION_STARTED"
        assert first["opportunities_appended"] == 1
        assert first["no_decision_count"] == 1
        rows = [json.loads(x) for x in (cdir / "opportunity_scores.jsonl").read_text().splitlines() if x.strip()]
        assert len(rows) == 1
        row = rows[0]
        assert row["hypothetical_decision"] == "NO_DECISION"
        assert row["no_decision_reason"] == "MISSING_H12_FEATURES"
        assert row["enforcement_action"] == "NONE"
        assert row["labels_resolved"] is False
        assert row["primary_horizon"] == 12
        assert row["diagnostic_horizons"] == [6, 24]
        assert row["score_h6"] is None and row["score_h24"] is None
        second = run_collect(args)
        assert second["opportunities_appended"] == 0
        assert second["duplicates_skipped"] == 1
        rows2 = [json.loads(x) for x in (cdir / "opportunity_scores.jsonl").read_text().splitlines() if x.strip()]
        assert len(rows2) == 1


def test_source_mutation_detected() -> None:
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        cdir, signal = frozen_candidate(root)
        args = collect_args(["--candidate-dir", str(cdir), "--source-path", str(signal), "--since", "2026-07-10T20:00:00Z", "--until", "2026-07-10T20:30:00Z"])
        run_collect(args)
        write_signal(signal, "SIG-1", "2026-07-10T20:10:00Z", reason="mutated")
        payload = run_collect(args)
        assert payload["source_conflicts"]
        assert payload["opportunities_appended"] == 0


def test_dry_run_does_not_append_or_touch_yaml() -> None:
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        cdir, signal = frozen_candidate(root)
        payload = run_collect(
            collect_args(
                [
                    "--candidate-dir",
                    str(cdir),
                    "--source-path",
                    str(signal),
                    "--since",
                    "2026-07-10T20:00:00Z",
                    "--until",
                    "2026-07-10T20:30:00Z",
                    "--dry-run",
                    "true",
                ]
            )
        )
        assert payload["dry_run"] is True
        assert payload["opportunities_appended"] == 0
        assert not list(cdir.glob("*.yaml"))
        assert not (cdir / "active_manifest.json").exists()


if __name__ == "__main__":
    test_no_new_opportunities_status()
    test_collect_append_idempotent_and_no_decision()
    test_source_mutation_detected()
    test_dry_run_does_not_append_or_touch_yaml()
    print("test_collect_trrm_forward_scores_f0: OK")
