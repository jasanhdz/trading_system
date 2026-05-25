#!/usr/bin/env python3
from __future__ import annotations

import tempfile
import sys
from pathlib import Path

import joblib
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from aegis_alpha.turbo.train_operable_edge_v2 import (
    MODEL_FAMILIES,
    classification_metrics,
    prediction_bucket_metrics,
    probability_bucket_metrics,
    regression_metrics,
    research_model_path,
    temporal_split_indices,
    train_side_models,
)
from aegis_alpha.tools.train_turbo_operable_v2_research import global_metrics


def assert_true(value: bool, message: str) -> None:
    if not value:
        raise AssertionError(message)


def assert_close(actual: float, expected: float, message: str, tol: float = 1e-6) -> None:
    if abs(actual - expected) > tol:
        raise AssertionError(f"{message}: expected={expected} actual={actual}")


def synthetic_dataset(sample_count: int = 600, single_class: bool = False) -> dict[str, object]:
    rng = np.random.default_rng(42)
    x = rng.normal(size=(sample_count, 5)).astype(np.float32)
    edge = x[:, 0] - 0.35 * x[:, 1]
    hit = np.zeros(sample_count, dtype=np.int8) if single_class else (edge > 0.35).astype(np.int8)
    danger = (x[:, 2] > 0.2).astype(np.int8)
    quality = np.clip(hit.astype(np.float32) * 0.7 - danger.astype(np.float32) * 0.45 + edge * 0.1, -1.0, 1.0)
    mae = np.where(danger > 0, 0.009, 0.002).astype(np.float32)
    return {
        "X": x,
        "feature_names": np.asarray([f"f{idx}" for idx in range(5)]),
        "long_hit8_before_minus5_12": hit,
        "long_trade_quality_12": quality,
        "long_mae_danger_12": danger,
        "long_mae_12": mae,
        "long_net_return_12": quality * 0.004,
        "short_hit8_before_minus5_12": hit,
        "short_trade_quality_12": quality,
        "short_mae_danger_12": danger,
        "short_mae_12": mae,
        "short_net_return_12": quality * 0.004,
    }


def test_temporal_split_preserves_order() -> None:
    split = temporal_split_indices(100)
    assert_true(list(split["train"]) == list(range(60)), "train should be first 60 percent")
    assert_true(list(split["validation"]) == list(range(60, 80)), "validation should follow train")
    assert_true(list(split["test"]) == list(range(80, 100)), "test should be last segment")


def test_metrics_handle_constant_probabilities() -> None:
    y = np.asarray([0, 1, 0, 1], dtype=np.int8)
    metrics = classification_metrics(y, np.full(4, 0.5))
    assert_true(metrics["brier_score"] is not None, "brier score should exist")
    assert_true(metrics["roc_auc"] is not None, "auc should exist with both classes")


def test_bucket_builders_keep_counts() -> None:
    proba = np.asarray([0.02, 0.15, 0.85, 0.95])
    binary = np.asarray([0, 0, 1, 1])
    quality = np.asarray([-0.2, -0.1, 0.4, 0.8])
    mae = np.asarray([0.01, 0.008, 0.002, 0.001])
    rows = probability_bucket_metrics(proba, binary, quality, mae, binary, binary)
    assert_true(sum(row["count"] for row in rows) == 4, "probability buckets should retain count")
    qrows = prediction_bucket_metrics(quality, quality, binary, binary, mae)
    assert_true(sum(row["count"] for row in qrows) == 4, "prediction buckets should retain count")


def test_regression_metrics_report_error() -> None:
    metrics = regression_metrics(np.asarray([0.0, 1.0, 2.0]), np.asarray([0.0, 1.0, 2.0]))
    assert_close(float(metrics["mae"] or 0.0), 0.0, "perfect MAE")
    assert_close(float(metrics["rmse"] or 0.0), 0.0, "perfect RMSE")


def test_global_metrics_are_weighted() -> None:
    values = global_metrics([
        {"sample_count": 1, "baseline_hit8_rate": 0.0},
        {"sample_count": 3, "baseline_hit8_rate": 1.0},
    ])
    assert_close(float(values["baseline_hit8_rate"] or 0.0), 0.75, "weighted global metric")


def test_research_path_rejects_active_dir() -> None:
    good = research_model_path(Path("/tmp/models/research/turbo_v2/ETHUSDT/run"), "long", "hit8_classifier", 12, 30)
    assert_true("v2_long_hit8_12_30d.joblib" in str(good), "research filename")
    try:
        research_model_path(Path("/tmp/models/active/ETHUSDT"), "long", "hit8_classifier", 12, 30)
    except ValueError:
        return
    raise AssertionError("active output path must be rejected")


def test_single_class_classifier_is_marked_insufficient() -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        result = train_side_models(
            synthetic_dataset(single_class=True),
            symbol="SUIUSDT",
            side="long",
            lookback_days=30,
            horizon=12,
            run_dir=Path(temp_dir) / "research",
            save_models=False,
            fast=True,
        )
    assert_true(
        result["families"]["hit8_classifier"]["model_status"] == "insufficient_class_diversity",
        "single-class hit classifier should not train",
    )
    assert_true(result["research_status"] == "INSUFFICIENT_DATA", "status should reflect invalid family")


def test_training_saves_research_bundles_with_metadata() -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        run_dir = Path(temp_dir) / "models" / "research" / "turbo_v2" / "ETHUSDT" / "stamp"
        result = train_side_models(
            synthetic_dataset(),
            symbol="ETHUSDT",
            side="long",
            lookback_days=30,
            horizon=12,
            run_dir=run_dir,
            save_models=True,
            fast=True,
        )
        assert_true(result["model_status"] == "trained", "synthetic set should train")
        for family in MODEL_FAMILIES:
            path = Path(result["families"][family]["model_path"])
            assert_true(path.exists(), f"model missing for {family}")
            assert_true("active" not in {part.lower() for part in path.parts}, "model should not be in active")
            bundle = joblib.load(path)
            metadata = bundle["metadata"]
            assert_true(metadata["research_only"] is True, "research metadata required")
            assert_true(metadata["not_live_promoted"] is True, "non-live metadata required")
            assert_true(bool(metadata["feature_schema_hash"]), "feature hash required")


def run_all() -> None:
    test_temporal_split_preserves_order()
    test_metrics_handle_constant_probabilities()
    test_bucket_builders_keep_counts()
    test_regression_metrics_report_error()
    test_global_metrics_are_weighted()
    test_research_path_rejects_active_dir()
    test_single_class_classifier_is_marked_insufficient()
    test_training_saves_research_bundles_with_metadata()
    print("manual_train_operable_edge_v2_tests_passed")


if __name__ == "__main__":
    run_all()
