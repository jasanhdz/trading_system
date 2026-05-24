#!/usr/bin/env python3
from __future__ import annotations

import sys
import json
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from aegis_alpha.tools.audit_turbo_training_pipeline import (
    build_feature_rows,
    build_leakage_rows,
    build_pipeline_map,
    duplicate_or_correlated_features,
    feature_quality_summary,
    feature_stats_matrix,
    load_operable_targets_context,
    likely_phase2_red_explanation,
    merge_feature_stats,
    recommendations_for_phase_b,
)


def assert_true(value: bool, message: str) -> None:
    if not value:
        raise AssertionError(message)


def assert_close(actual: float, expected: float, message: str, tol: float = 1e-9) -> None:
    if abs(actual - expected) > tol:
        raise AssertionError(f"{message}: expected={expected} actual={actual}")


def test_feature_stats_detects_nan_inf_zero_constant() -> None:
    x = np.array(
        [
            [0.0, 1.0, 2.0, 2.0],
            [0.0, 1.0, np.nan, 2.0],
            [1.0, 1.0, np.inf, 2.0],
            [0.0, 1.0, 4.0, 2.0],
        ],
        dtype=np.float32,
    )
    stats = feature_stats_matrix(x, ["a", "b", "c", "d"])
    assert_close(float(stats["a"]["zero_rate"]), 0.75, "zero rate")
    assert_close(float(stats["b"]["constant_rate"]), 1.0, "constant rate")
    assert_close(float(stats["c"]["nan_rate"]), 0.25, "nan rate")
    assert_close(float(stats["c"]["inf_rate"]), 0.25, "inf rate")


def test_duplicate_detector_flags_duplicate_column() -> None:
    x = np.array(
        [
            [1.0, 1.0, 5.0],
            [2.0, 2.0, 4.0],
            [3.0, 3.0, 3.0],
            [4.0, 4.0, 2.0],
        ],
        dtype=np.float32,
    )
    risks = duplicate_or_correlated_features(x, ["left", "dupe", "other"])
    assert_true("dupe" in risks, "duplicate feature should be flagged")
    assert_true("left" in risks["dupe"], "duplicate reason should mention source feature")


def test_merge_and_feature_rows_keep_schema() -> None:
    merged = merge_feature_stats([
        {
            "last_log_ret": {
                "nan_rate": 0.0,
                "inf_rate": 0.0,
                "zero_rate": 0.2,
                "constant_rate": 0.0,
                "duplicate_correlation_risk": "none_detected",
            }
        }
    ])
    rows = build_feature_rows(merged)
    last_log_ret = next(row for row in rows if row.feature_name == "last_log_ret")
    assert_true(last_log_ret.base_feature == "log_ret", "base feature should parse")
    assert_true(last_log_ret.symbols_observed == 1, "symbols observed should be merged")
    assert_close(float(last_log_ret.zero_rate or 0.0), 0.2, "merged zero rate")


def test_leakage_rows_include_target_misalignment() -> None:
    risks = build_leakage_rows()
    joined = " ".join(row.risk for row in risks)
    assert_true("return_target_not_trade_path" in joined, "target misalignment risk required")


def test_feature_quality_summary_marks_dead_columns() -> None:
    rows = build_feature_rows({
        "last_candle_progress": {
            "symbols_observed": 2,
            "nan_rate": 0.0,
            "inf_rate": 0.0,
            "zero_rate": 1.0,
            "constant_rate": 1.0,
            "duplicate_correlation_risk": "none_detected",
        }
    })
    summary = feature_quality_summary(rows)
    assert_true("last_candle_progress" in summary["constant_features"], "constant feature should be listed")
    assert_true(
        "candle_progress_family_all_zero_in_observed_datasets" in summary["warnings"],
        "candle_progress warning should be explicit",
    )


def test_pipeline_map_mentions_runtime_and_training() -> None:
    stages = {row["stage"] for row in build_pipeline_map()}
    assert_true("training" in stages, "pipeline map should include training")
    assert_true("runtime_signal" in stages, "pipeline map should include runtime signal")
    assert_true("inference_endpoint" in stages, "pipeline map should include inference endpoint")


def test_explanations_and_phase_b_recommendations() -> None:
    explanations = " ".join(likely_phase2_red_explanation()).lower()
    recommendations = " ".join(recommendations_for_phase_b()).lower()
    assert_true("hit-before-stop" in explanations or "hit-before-stop" in recommendations, "hit-before-stop gap should be explicit")
    assert_true("trade_quality" in recommendations, "Fase B should recommend trade_quality targets")


def test_operable_target_report_context() -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        path = Path(temp_dir) / "targets.json"
        path.write_text(json.dumps({"global_horizon_12": [{"side": "LONG"}]}), encoding="utf-8")
        context = load_operable_targets_context(str(path))
    assert_true(context["operable_v2_targets_present"], "V2 target availability should be recorded")
    assert_true(context["distribution_report_loaded"], "distribution JSON should load")
    assert_true(context["global_horizon_12"][0]["side"] == "LONG", "global summary should pass through")


def run_all() -> None:
    test_feature_stats_detects_nan_inf_zero_constant()
    test_duplicate_detector_flags_duplicate_column()
    test_merge_and_feature_rows_keep_schema()
    test_leakage_rows_include_target_misalignment()
    test_feature_quality_summary_marks_dead_columns()
    test_pipeline_map_mentions_runtime_and_training()
    test_explanations_and_phase_b_recommendations()
    test_operable_target_report_context()
    print("manual_turbo_training_pipeline_audit_tests_passed")


if __name__ == "__main__":
    run_all()
