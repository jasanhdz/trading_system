from __future__ import annotations

from summarize_tail_aware_entry_v8 import summarize


def test_v8_summary_preserves_fail_closed_runtime_state() -> None:
    side = {
        "rows": 10,
        "passing_folds": 0,
        "router_skilled_folds": 0,
        "late_detector_skilled_folds": 0,
        "worst_fold_non_negative": False,
        "primary_gate": False,
        "leave_one_symbol_out": {"status": "NOT_RUN", "passed": False},
        "validation_pass": False,
        "folds": [],
    }
    validation = {
        "experiment_id": "v8-test",
        "config_sha256": "a" * 64,
        "dataset_sha256": "b" * 64,
        "evidence_start": "2026-01-01T00:00:00+00:00",
        "evidence_end": "2026-01-02T00:00:00+00:00",
        "sides": {"LONG": side, "SHORT": side},
        "validation_pass": False,
        "verdict": "RESEARCH_ONLY_NOT_PROMOTABLE",
        "shadow_runtime_enabled": False,
        "model_exported": False,
        "exchange_calls": 0,
        "exchange_mutations": 0,
    }
    summary = summarize(validation, "c" * 64)
    assert summary["gate"]["passed"] is False
    assert summary["runtime_effect"] == "NONE"
    assert summary["exchange_mutations"] == 0
