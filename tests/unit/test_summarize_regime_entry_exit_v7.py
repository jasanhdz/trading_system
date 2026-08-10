from __future__ import annotations

from summarize_regime_entry_exit_v7 import compact_summary


def test_summary_preserves_fail_closed_counters() -> None:
    side = {
        "rows": 10,
        "evaluated_folds": 0,
        "passing_folds": 0,
        "router_skilled_folds": 0,
        "worst_fold_non_negative": False,
        "leave_one_symbol_out": {"passed": False},
        "validation_pass": False,
        "folds": [],
    }
    validation = {
        "experiment_id": "v7-test",
        "evidence_start": "2026-01-01T00:00:00+00:00",
        "evidence_end": "2026-01-02T00:00:00+00:00",
        "dataset_sha256": "a" * 64,
        "dataset_manifest_sha256": "b" * 64,
        "sides": {"LONG": side, "SHORT": side},
        "verdict": "RESEARCH_ONLY_NOT_PROMOTABLE",
        "exchange_calls": 0,
        "exchange_mutations": 0,
        "shadow_runtime_enabled": False,
        "model_exported": False,
    }
    manifest = {
        "trajectory_responsibility_counts": {"NO_DIRECTIONAL_EDGE": 10},
        "hindsight_best_profile_counts": {"CURRENT_TS": 10},
        "current_protection_replay_mismatches": 0,
    }
    summary = compact_summary(validation, manifest, "c" * 64)
    assert summary["gate"]["passed"] is False
    assert summary["runtime_effect"] == "NONE"
    assert summary["exchange_mutations"] == 0
