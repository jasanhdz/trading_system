#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import aegis_alpha.tools.build_trrm_fresh_candidate_f02 as mod
import aegis_alpha.tools.trrm_forward_common_f0 as common
from aegis_alpha.tests.test_freeze_trrm_forward_candidate_f0 import make_artifacts
from aegis_alpha.tools.freeze_trrm_forward_candidate_f0 import parse_args as freeze_args
from aegis_alpha.tools.freeze_trrm_forward_candidate_f0 import run_freeze


def previous_candidate(root: Path, art: dict[str, Path]) -> Path:
    payload = run_freeze(
        freeze_args(
            [
                "--e2-report-json", str(art["e2"]),
                "--e21-report-json", str(art["e21"]),
                "--fable-audit-json", str(art["fable"]),
                "--model-dir", str(art["model_dir"]),
                "--policy-dir", str(art["policy_dir"]),
                "--internal-predictions", str(art["internal"]),
                "--output-root", str(root / "f0"),
                "--candidate-id", "old_f0",
                "--freeze-time", "2026-07-10T00:00:00Z",
                "--feature-hash", art["feature_hash"],
            ]
        )
    )
    return Path(payload["candidate_dir"])


def base_args(root: Path, art: dict[str, Path], prev: Path, freeze: str) -> list[str]:
    return [
        "--previous-candidate-dir", str(prev),
        "--d2-csv", str(art["dense"]),
        "--incremental-features", str(art["dense"]),
        "--e21-report-json", str(art["e21"]),
        "--internal-predictions", str(art["internal"]),
        "--model-dir", str(art["model_dir"]),
        "--output-root", str(root / "f02"),
        "--candidate-id", "trrm_e21_f02_test",
        "--freeze-time", freeze,
        "--feature-hash", art["feature_hash"],
        "--model-hash", common.combined_model_hash(art["model_dir"]),
    ]


def test_f02_candidate_created_and_stale_blocks_manifest() -> None:
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        art = make_artifacts(root)
        common.EXPECTED_FEATURE_HASH = art["feature_hash"]
        mod.EXPECTED_FEATURE_HASH = art["feature_hash"]
        prev = previous_candidate(root, art)
        payload = mod.run_build(mod.parse_args(base_args(root, art, prev, "2026-03-01T00:00:00Z")))
        assert payload["decision"] == "F02_FRESH_SEED_READY"
        cdir = Path(payload["candidate_dir"])
        manifest = json.loads((cdir / "candidate_manifest.json").read_text())
        assert manifest["phase"] == "F0.2"
        assert manifest["supersedes_candidate_id"] == "old_f0"
        assert manifest["previous_candidate_modified"] is False
        assert manifest["target"] == "target.tail_risk_roe_030"
        assert manifest["policy_method"] == "ROLLING_GLOBAL_QUANTILE_PAST_ONLY"
        assert manifest["budget"] == 0.30
        assert manifest["rolling_window_days"] == 30
        assert manifest["engine_name"] == "E21_PER_ROW_CANONICAL"
        assert manifest["primary_horizon"] == 12
        assert manifest["diagnostic_horizons"] == [6, 24]
        assert manifest["labels_enabled"] is False
        assert manifest["enforcement_enabled"] is False
        assert manifest["engine_replay"]["status"] == "OK"
        seed_text = (cdir / "history_seed.jsonl").read_text().lower()
        assert "target." not in seed_text
        assert "future" not in seed_text
        stale_args = base_args(root, art, prev, "2026-06-01T00:00:00Z")
        stale_args[stale_args.index("--candidate-id") + 1] = "trrm_e21_f02_stale"
        stale = mod.run_build(mod.parse_args(stale_args))
        assert stale["decision"] == "RECENT_SEED_STALE"
        assert not (root / "f02" / "trrm_e21_f02_stale" / "candidate_manifest.json").exists()


if __name__ == "__main__":
    test_f02_candidate_created_and_stale_blocks_manifest()
    print("test_build_trrm_fresh_candidate_f02: OK")
