#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import aegis_alpha.tools.trrm_forward_common_f0 as common
from aegis_alpha.tests.test_freeze_trrm_forward_candidate_f0 import make_artifacts
from aegis_alpha.tools.freeze_trrm_forward_candidate_f0 import parse_args as freeze_args
from aegis_alpha.tools.freeze_trrm_forward_candidate_f0 import run_freeze
from aegis_alpha.tools.prepare_trrm_fresh_seed_f01 import parse_args, run_prepare


def previous_candidate(root: Path, art: dict[str, Path]) -> Path:
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
                str(root / "f0"),
                "--candidate-id",
                "trrm_e21_f0_old",
                "--freeze-time",
                "2026-05-01T00:00:00Z",
                "--feature-hash",
                art["feature_hash"],
            ]
        )
    )
    return Path(payload["candidate_dir"])


def prepare_base_args(root: Path, art: dict[str, Path], prev: Path, freeze_time: str) -> list[str]:
    return [
        "--previous-candidate-dir",
        str(prev),
        "--e2-report-json",
        str(art["e2"]),
        "--e21-report-json",
        str(art["e21"]),
        "--model-dir",
        str(art["model_dir"]),
        "--policy-dir",
        str(art["policy_dir"]),
        "--internal-predictions",
        str(art["internal"]),
        "--feature-source-path",
        str(art["dense"]),
        "--output-root",
        str(root / "f01"),
        "--candidate-id",
        "trrm_e21_f01_new",
        "--freeze-time",
        freeze_time,
        "--feature-hash",
        art["feature_hash"],
    ]


def test_fresh_seed_candidate_supersedes_without_modifying_previous() -> None:
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        art = make_artifacts(root)
        common.EXPECTED_FEATURE_HASH = art["feature_hash"]
        prev = previous_candidate(root, art)
        before = (prev / "candidate_manifest.json").read_bytes()
        payload = run_prepare(parse_args(prepare_base_args(root, art, prev, "2026-02-15T00:00:00Z")))
        assert payload["decision"] == "F01_FRESH_SEED_READY"
        cdir = Path(payload["candidate_dir"])
        assert cdir.name == "trrm_e21_f01_new"
        manifest = json.loads((cdir / "candidate_manifest.json").read_text())
        assert manifest["phase"] == "F0.1"
        assert manifest["supersedes_candidate_id"] == "trrm_e21_f0_old"
        assert manifest["previous_candidate_modified"] is False
        assert manifest["feature_hash"] == art["feature_hash"]
        assert manifest["target"] == "target.tail_risk_roe_030"
        assert manifest["policy_method"] == "ROLLING_GLOBAL_QUANTILE_PAST_ONLY"
        assert manifest["budget"] == 0.30
        assert manifest["rolling_window_days"] == 30
        assert manifest["engine_name"] == "E21_PER_ROW_CANONICAL"
        assert manifest["primary_horizon"] == 12
        assert manifest["diagnostic_horizons"] == [6, 24]
        assert manifest["fresh_seed_last_age_hours"] <= 24
        assert manifest["fresh_seed_days"] >= 25
        assert (prev / "candidate_manifest.json").read_bytes() == before
        seed_text = (cdir / "history_seed.jsonl").read_text().lower()
        assert "target." not in seed_text
        assert "future" not in seed_text
        assert "pnl" not in seed_text
        assert manifest["history_seed"]["labels_present"] is False


def test_stale_and_insufficient_seed_block_candidate_creation() -> None:
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        art = make_artifacts(root)
        common.EXPECTED_FEATURE_HASH = art["feature_hash"]
        prev = previous_candidate(root, art)
        stale = run_prepare(parse_args(prepare_base_args(root, art, prev, "2026-05-06T12:00:00Z")))
        assert stale["decision"] == "RECENT_SEED_STALE"
        assert not (root / "f01" / "trrm_e21_f01_new" / "candidate_manifest.json").exists()
        insufficient_args = prepare_base_args(root, art, prev, "2025-07-15T00:00:00Z")
        insufficient_args.extend(["--minimum-recent-seed-days", "25"])
        insufficient = run_prepare(parse_args(insufficient_args))
        assert insufficient["decision"] == "RECENT_SEED_INSUFFICIENT"


def test_missing_feature_source_reports_not_ready() -> None:
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        art = make_artifacts(root)
        common.EXPECTED_FEATURE_HASH = art["feature_hash"]
        prev = previous_candidate(root, art)
        args = prepare_base_args(root, art, prev, "2026-02-15T00:00:00Z")
        args[args.index("--feature-source-path") + 1] = str(root / "missing.csv")
        payload = run_prepare(parse_args(args))
        assert payload["decision"] == "RECENT_FEATURE_SOURCE_NOT_READY"


if __name__ == "__main__":
    test_fresh_seed_candidate_supersedes_without_modifying_previous()
    test_stale_and_insufficient_seed_block_candidate_creation()
    test_missing_feature_source_reports_not_ready()
    print("test_prepare_trrm_fresh_seed_f01: OK")
