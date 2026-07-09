#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from aegis_alpha.tools.build_shadow_committee_dataset_a import (
    SCHEMA_FIELDS,
    apply_rule_baselines,
    build_rows_from_events,
    run_builder,
)


def test_no_logs_status_and_schema() -> None:
    with tempfile.TemporaryDirectory() as td:
        result = run_builder(argparse.Namespace(out_dir=td))
        assert result["status"] == "INSUFFICIENT_SHADOW_LOGS"
        assert result["schema_fields"] == SCHEMA_FIELDS
        assert Path(result["outputs"]["csv"]).exists()


def test_parse_signal_fixture() -> None:
    rows = build_rows_from_events([{"event_type": "SIGNAL", "timestamp": "2026-07-09T00:00:00Z", "symbol": "ADAUSDT", "side": "SHORT", "score": 0.8}])
    assert len(rows) == 1
    assert rows[0]["symbol"] == "ADAUSDT"


def test_baseline_blocks_two_guards() -> None:
    row = {k: "" for k in SCHEMA_FIELDS}
    row.update({"clean_entry_would_block": True, "entry_quality_action": "SHADOW_BLOCK"})
    out = apply_rule_baselines([row])[0]
    assert out["block_if_2_or_more_guards_block"] is True
    assert out["baseline_block_count"] >= 2


def test_no_training_flag() -> None:
    with tempfile.TemporaryDirectory() as td:
        result = run_builder(argparse.Namespace(out_dir=td))
        assert result["training_readiness"] == "not_ready_wait_for_forward_logs"


if __name__ == "__main__":
    test_no_logs_status_and_schema()
    test_parse_signal_fixture()
    test_baseline_blocks_two_guards()
    test_no_training_flag()
    print("test_build_shadow_committee_dataset_a: OK")
