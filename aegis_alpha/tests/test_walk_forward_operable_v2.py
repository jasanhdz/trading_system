#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from aegis_alpha.turbo.walk_forward_operable_v2 import (
    aggregate_walk_forward_results,
    run_walk_forward,
    temporal_folds,
    walkforward_model_path,
)


def assert_true(value: bool, message: str) -> None:
    if not value:
        raise AssertionError(message)


def synthetic_dataset(sample_count: int = 1600, single_class: bool = False) -> dict[str, object]:
    rng = np.random.default_rng(17)
    x = rng.normal(size=(sample_count, 5)).astype(np.float32)
    edge = x[:, 0] - x[:, 1] * 0.25
    hit = np.zeros(sample_count, dtype=np.int8) if single_class else (edge > 0.2).astype(np.int8)
    danger = np.zeros(sample_count, dtype=np.int8) if single_class else (x[:, 2] > 0.35).astype(np.int8)
    quality = np.clip(hit * 0.65 - danger * 0.35 + edge * 0.08, -1.0, 1.0).astype(np.float32)
    mae = np.where(danger > 0, 0.009, 0.002).astype(np.float32)
    return {
        "X": x,
        "feature_names": np.asarray([f"f{idx}" for idx in range(5)]),
        "short_hit8_before_minus5_12": hit,
        "short_trade_quality_12": quality,
        "short_mae_danger_12": danger,
        "short_mae_12": mae,
        "short_net_return_12": quality * 0.004,
    }


def fake_fold(hit_lift: float, quality_lift: float, top_quality: float, fold_number: int) -> dict[str, object]:
    return {
        "fold": fold_number,
        "model_status": "trained",
        "baseline_test": {"hit8_rate": 0.30, "avg_trade_quality": -0.02, "mae_danger_rate": 0.25},
        "v1_target_reference": {"corr_trade_quality": 0.40},
        "families": {
            "hit8_classifier": {
                "test_metrics": {"roc_auc": 0.62},
                "top_decile": {"hit8_lift_vs_baseline": hit_lift},
            },
            "trade_quality_regressor": {
                "test_metrics": {"spearman": 0.25},
                "top_decile": {
                    "quality_lift_vs_baseline": quality_lift,
                    "avg_trade_quality": top_quality,
                    "p90_mae_delta_vs_baseline": 0.0,
                },
            },
            "mae_danger_classifier": {
                "test_metrics": {"roc_auc": 0.60},
                "usefulness_as_filter": 0.18,
            },
        },
    }


def test_temporal_folds_preserve_order_and_no_future() -> None:
    folds = temporal_folds(
        5000,
        fold_count=4,
        min_train_samples=1000,
        min_test_samples=300,
    )
    assert_true(len(folds) == 4, "four valid folds expected")
    previous_test_end = -1
    for fold in folds:
        assert_true(int(fold["train"][-1]) < int(fold["validation"][0]), "train must precede validation")
        assert_true(int(fold["validation"][-1]) < int(fold["test"][0]), "validation must precede test")
        assert_true(int(fold["train"][0]) == 0, "default window must be expanding")
        assert_true(int(fold["test"][-1]) > previous_test_end, "folds should advance in time")
        previous_test_end = int(fold["test"][-1])


def test_sliding_window_does_not_overlap_train_and_test() -> None:
    folds = temporal_folds(
        5000,
        fold_count=3,
        expanding_window=False,
        min_train_samples=1000,
        min_test_samples=300,
    )
    assert_true(len(folds) == 3, "three valid sliding folds expected")
    assert_true(int(folds[-1]["train"][0]) > 0, "sliding train window should advance")
    assert_true(set(folds[-1]["train"]).isdisjoint(set(folds[-1]["test"])), "train and test must not overlap")


def test_aggregate_marks_promising_when_improvement_is_stable() -> None:
    folds = [fake_fold(0.08, 0.10, 0.09, index) for index in range(1, 4)]
    summary = aggregate_walk_forward_results("ADAUSDT", "short", 30, 12, folds, 3)
    assert_true(summary["recommendation"] == "WALK_FORWARD_PROMISING", "stable lifts should be promising")
    assert_true(float(summary["hit8_top_decile_lift_mean"]) > 0.0, "positive hit lift expected")


def test_aggregate_marks_bad_when_top_decile_is_worse() -> None:
    folds = [fake_fold(-0.08, -0.10, -0.12, index) for index in range(1, 4)]
    summary = aggregate_walk_forward_results("AVAXUSDT", "short", 30, 12, folds, 3)
    assert_true(summary["recommendation"] == "WALK_FORWARD_BAD", "consistently adverse folds should fail")


def test_missing_folds_are_insufficient_and_json_serializable() -> None:
    summary = aggregate_walk_forward_results("ADAUSDT", "short", 30, 12, [fake_fold(0.1, 0.1, 0.1, 1)], 4)
    assert_true(summary["recommendation"] == "INSUFFICIENT_DATA", "too few folds should be insufficient")
    assert_true("ADAUSDT" in json.dumps(summary), "summary must serialize to JSON")


def test_path_is_research_only() -> None:
    good = walkforward_model_path(Path("/tmp/models/research/turbo_v2_walkforward/ADAUSDT/run"), 1, "short", "hit8_classifier", 12, 30)
    assert_true("fold_1" in str(good), "fold path expected")
    try:
        walkforward_model_path(Path("/tmp/models/active/ADAUSDT/run"), 1, "short", "hit8_classifier", 12, 30)
    except ValueError:
        return
    raise AssertionError("active path must be rejected")


def test_single_class_fold_is_not_valid_training_evidence() -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        result = run_walk_forward(
            synthetic_dataset(single_class=True),
            symbol="ADAUSDT",
            side="short",
            lookback_days=30,
            horizon=12,
            fold_count=3,
            train_ratio=0.50,
            validation_ratio=0.15,
            test_ratio=0.15,
            expanding_window=True,
            min_train_samples=200,
            min_test_samples=100,
            run_dir=Path(temp_dir) / "research",
            save_models=False,
            fast=True,
        )
    assert_true(result["summary"]["valid_fold_count"] == 0, "single-class classifiers invalidate folds")
    assert_true(result["summary"]["recommendation"] == "INSUFFICIENT_DATA", "invalid folds should not promote")
    assert_true(
        result["folds"][0]["families"]["hit8_classifier"]["model_status"] == "insufficient_class_diversity",
        "hit8 family should explain invalidity",
    )


def run_all() -> None:
    test_temporal_folds_preserve_order_and_no_future()
    test_sliding_window_does_not_overlap_train_and_test()
    test_aggregate_marks_promising_when_improvement_is_stable()
    test_aggregate_marks_bad_when_top_decile_is_worse()
    test_missing_folds_are_insufficient_and_json_serializable()
    test_path_is_research_only()
    test_single_class_fold_is_not_valid_training_evidence()
    print("manual_walk_forward_operable_v2_tests_passed")


if __name__ == "__main__":
    run_all()
