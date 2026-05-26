#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path
from unittest.mock import patch

import numpy as np

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from aegis_alpha.tools.repair_short_v3_failure_modes import (
    best_by_symbol,
    classify_failure_mode,
    classify_repair_candidate,
    parse_symbols,
    PHASE_I_BASELINE,
    repair_configs,
    run,
    selection_mask,
    validate_research_model_dir,
    write_csv,
)


def assert_true(value: bool, message: str) -> None:
    if not value:
        raise AssertionError(message)


def lockbox_row(**updates: object) -> dict[str, object]:
    row: dict[str, object] = {
        "model_status": "trained",
        "test_samples": 400,
        "lockbox_status": "LOCKBOX_WEAK",
        "hit8_auc": 0.58,
        "hit8_top_decile_lift": 0.05,
        "quality_top_decile_lift": 0.06,
        "net_quality_lift_after_cost_proxy": 0.05,
        "quality_corr": 0.02,
        "top_decile_p90_mae": 0.006,
        "baseline_p90_mae": 0.006,
        "top_decile_mae_danger_rate": 0.10,
        "baseline_mae_danger": 0.10,
    }
    row.update(updates)
    return row


def repair_row(**updates: object) -> dict[str, object]:
    row: dict[str, object] = {
        "symbol": "ADAUSDT",
        "repair_mode": "quality_primary",
        "test_samples": 400,
        "selected_count": 40,
        "selected_hit8_lift": 0.04,
        "selected_quality_lift": 0.06,
        "selected_net_quality_lift_after_cost": 0.05,
        "selected_p90_mae": 0.006,
        "baseline_p90_mae": 0.006,
        "repair_status": "REPAIRED_CONFIRMED",
        "repair_score": 0.50,
    }
    row.update(updates)
    return row


def test_failure_classification() -> None:
    assert_true(
        classify_failure_mode(lockbox_row(hit8_auc=0.51)) == "AUC_WEAK_QUALITY_OK",
        "AUC weak quality okay",
    )
    assert_true(
        classify_failure_mode(lockbox_row(quality_corr=-0.04)) == "QUALITY_CORR_NEGATIVE_LIFTS_OK",
        "negative quality correlation",
    )
    assert_true(
        classify_failure_mode(lockbox_row(hit8_top_decile_lift=-0.01)) == "HIT8_LIFT_NEGATIVE",
        "negative hit8 lift",
    )


def test_repair_classification_and_doge_gate() -> None:
    assert_true(classify_repair_candidate(repair_row()) == "REPAIRED_CONFIRMED", "confirmed repair")
    assert_true(
        classify_repair_candidate(repair_row(selected_hit8_lift=-0.01)) == "REPAIRED_FAILED",
        "negative hit8 repair fails",
    )
    assert_true(
        classify_repair_candidate(repair_row(symbol="DOGEUSDT", selected_hit8_lift=0.02)) == "REPAIRED_WEAK",
        "DOGE needs more than three percent hit8 lift",
    )
    assert_true(
        classify_repair_candidate(repair_row(symbol="DOGEUSDT", selected_hit8_lift=0.04)) == "REPAIRED_CONFIRMED",
        "DOGE can confirm only past strict hit8 floor",
    )


def test_mode_selection_and_danger_filter() -> None:
    predictions = {
        "hit8": np.asarray([0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 1.00]),
        "quality": np.asarray([0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10]),
        "danger": np.asarray([0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.99, 1.00]),
    }
    top = selection_mask(predictions, "top_bucket_only")
    filtered = selection_mask(predictions, "top_bucket_only_danger_filtered")
    assert_true(int(top.sum()) == 1, "top bucket selects top ten percent")
    assert_true(int(filtered.sum()) < int(top.sum()), "danger filter reduces selected count")


def test_best_selection_and_serialization() -> None:
    rows = [
        repair_row(symbol="ADAUSDT", repair_status="REPAIRED_WEAK", repair_score=1.0),
        repair_row(symbol="ADAUSDT", repair_mode="quality_primary_danger_filtered", repair_score=0.30),
    ]
    selected = best_by_symbol(rows, ["ADAUSDT"])
    assert_true(selected[0]["repair_status"] == "REPAIRED_CONFIRMED", "confirmed outranks weak")
    assert_true(parse_symbols(None, False) == ["LINKUSDT", "ADAUSDT", "SOLUSDT", "BTCUSDT", "DOGEUSDT"], "default scope")
    assert_true(PHASE_I_BASELINE["DOGEUSDT"]["original_lockbox_status"] == "LOCKBOX_FAILED", "phase I status is frozen")
    assert_true(parse_symbols("ADAUSDT", True) == ["ADAUSDT", "BNBUSDT", "XRPUSDT"], "optional scope")
    assert_true(len(repair_configs("DOGEUSDT")) == 3, "DOGE repair scope remains bounded")
    assert_true(repair_configs("DOGEUSDT", True)[1]["cross_context_enabled"] is False, "context can be disabled explicitly")
    with tempfile.TemporaryDirectory() as temp:
        path = Path(temp) / "rows.csv"
        write_csv(path, selected)
        assert_true(path.exists(), "CSV serializes")
        assert_true("ADAUSDT" in json.dumps(selected), "JSON serializes")
    validate_research_model_dir(Path("/tmp/models/research/short_repair"))
    try:
        validate_research_model_dir(Path("/tmp/models/active/short_repair"))
    except ValueError:
        return
    raise AssertionError("active output path must be rejected")


def test_symbol_error_does_not_abort_and_no_models_are_written() -> None:
    with tempfile.TemporaryDirectory() as temp:
        args = argparse.Namespace(
            symbols="ADAUSDT,DOGEUSDT",
            include_bnb_xrp=False,
            disable_cross_context=False,
            out_dir=temp,
            model_dir=str(Path(temp) / "models" / "research"),
            lockbox_test_ratio=0.20,
            min_train_samples=1000,
            min_test_samples=300,
            fee_bps=8.0,
            slippage_bps=3.0,
            fast=True,
        )

        def load_market(_path: str, *, symbol_override: str) -> object:
            if symbol_override == "DOGEUSDT":
                raise RuntimeError("missing symbol")
            return object()

        original = {
            "symbol": "ADAUSDT",
            "lockbox_status": "LOCKBOX_WEAK",
            "model_status": "trained",
            "test_samples": 400,
            "hit8_auc": 0.51,
            "hit8_top_decile_lift": 0.03,
            "quality_top_decile_lift": 0.04,
            "net_quality_lift_after_cost_proxy": 0.03,
            "quality_corr": 0.02,
            "top_decile_p90_mae": 0.006,
            "baseline_p90_mae": 0.006,
            "top_decile_mae_danger_rate": 0.10,
            "baseline_mae_danger": 0.10,
        }
        with patch("aegis_alpha.tools.repair_short_v3_failure_modes.load_signal_market", side_effect=load_market):
            with patch(
                "aegis_alpha.tools.repair_short_v3_failure_modes.build_recent_dataset",
                return_value={"dataset": {}},
            ):
                with patch("aegis_alpha.tools.repair_short_v3_failure_modes.apply_feature_set", return_value={}):
                    with patch("aegis_alpha.tools.repair_short_v3_failure_modes._fit_predictions", return_value={"model_status": "trained"}):
                        with patch("aegis_alpha.tools.repair_short_v3_failure_modes.original_lockbox_metrics", return_value=original):
                            with patch("aegis_alpha.tools.repair_short_v3_failure_modes._evaluate_dataset", return_value=[repair_row()]):
                                report = run(args)
        assert_true(len(report["errors"]) == 1, "one bad symbol is recorded")
        assert_true(report["best_by_symbol"][0]["symbol"] == "ADAUSDT", "healthy symbol remains evaluated")
        assert_true(report["save_models"] is False, "models are not saved")
        assert_true(report["shadow_models_generated"] is False, "shadow artifacts are not created")
        assert_true(report["active_manifest_touched"] is False, "active manifest remains untouched")


def run_all() -> None:
    test_failure_classification()
    test_repair_classification_and_doge_gate()
    test_mode_selection_and_danger_filter()
    test_best_selection_and_serialization()
    test_symbol_error_does_not_abort_and_no_models_are_written()
    print("manual_repair_short_v3_failure_modes_tests_passed")


if __name__ == "__main__":
    run_all()
