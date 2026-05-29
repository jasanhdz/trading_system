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

from aegis_alpha.tools.research_link_short_alpha_n import (  # noqa: E402
    classify_link_n_candidate,
    configs_for_args,
    link_avoid_mask,
    link_failed_retest_mask,
    link_slow_trend_pullback_mask,
    run,
    select_best_link_n,
)


def assert_true(value: bool, message: str) -> None:
    if not value:
        raise AssertionError(message)


def row(**updates: object) -> dict[str, object]:
    result: dict[str, object] = {
        "symbol": "LINKUSDT",
        "side": "SHORT",
        "alpha_family": "slow_trend_pullback_short",
        "feature_set": "combined_v3",
        "feature_mode": "selected_family",
        "lookback_days": 30,
        "target_name": "hit3_before_minus2",
        "horizon_candles": 12,
        "decision_mode": "quality_primary",
        "model_status": "trained",
        "feature_count": 10,
        "test_samples": 600,
        "selected_count": 60,
        "selected_fraction": 0.10,
        "selected_target_lift": 0.04,
        "selected_quality_lift": 0.08,
        "selected_net_quality_lift": 0.06,
        "baseline_p90_mae": 0.006,
        "selected_p90_mae": 0.006,
        "selected_p90_mae_delta": 0.0,
        "baseline_mae_danger": 0.20,
        "selected_mae_danger_rate": 0.20,
        "selected_mae_danger_delta": 0.0,
        "target_auc": 0.56,
        "quality_corr": 0.01,
    }
    result.update(updates)
    return result


def trained_fixture(n: int = 100) -> dict[str, object]:
    return {
        "model_status": "trained",
        "test": np.arange(n, dtype=np.int64),
        "hit_prob": np.linspace(0.1, 0.9, n),
        "quality_pred": np.linspace(-0.1, 0.5, n),
        "danger_prob": np.linspace(0.9, 0.1, n),
    }


def test_classification_gates() -> None:
    assert_true(classify_link_n_candidate(row()) == "LINK_ENTRY_CONFIRMED", "positive row confirms")
    assert_true(
        classify_link_n_candidate(row(selected_target_lift=0.0)) == "LINK_ENTRY_WEAK",
        "quality-only improvement is weak, not confirmed",
    )
    assert_true(
        classify_link_n_candidate(row(selected_net_quality_lift=0.0)) == "LINK_ENTRY_FAILED",
        "non-positive net quality fails",
    )
    avoid = row(
        decision_mode="avoid_by_danger_quality",
        avoid_selected_fraction=0.12,
        avoid_selected_count=72,
        avoid_quality_delta_vs_baseline=-0.06,
        avoid_mae_danger_delta=0.10,
        avoid_p90_mae_delta=0.002,
        avoid_hit_delta=-0.01,
        avoid_usefulness_score=0.16,
    )
    assert_true(classify_link_n_candidate(avoid) == "LINK_AVOID_ONLY_CONFIRMED", "avoid-only confirms bad zone")
    best = select_best_link_n([
        {**avoid, "n_status": "LINK_AVOID_ONLY_CONFIRMED"},
        {**row(selected_net_quality_lift=0.01), "n_status": "LINK_ENTRY_CONFIRMED"},
    ])
    assert_true(best["n_status"] == "LINK_ENTRY_CONFIRMED", "entry confirmed beats avoid confirmed")


def test_masks_degrade_and_sparse() -> None:
    trained = trained_fixture(100)
    missing_dataset = {
        "X": np.ones((100, 1), dtype=np.float32),
        "feature_names": np.asarray(["local_trend_down_score"]),
    }
    mask, missing, sparse = link_slow_trend_pullback_mask(missing_dataset, trained)
    assert_true(int(mask.sum()) == 10, "slow trend degrades to top quality")
    assert_true("btc_eth_long_contradiction" in missing, "missing feature is reported")
    assert_true(isinstance(sparse, list), "sparse list returned")

    sparse_dataset = {
        "X": np.column_stack([
            np.ones(100),
            np.ones(100),
            np.zeros(100),
        ]).astype(np.float32),
        "feature_names": np.asarray([
            "short_breakdown_strength_12",
            "short_retest_failed",
            "short_reclaim_range_risk",
        ]),
    }
    retest_mask, retest_missing, retest_sparse = link_failed_retest_mask(sparse_dataset, trained)
    assert_true(int(retest_mask.sum()) == 10, "failed retest still returns a mask")
    assert_true(
        "retest_features_missing_or_sparse" in retest_missing or len(retest_sparse) > 0,
        "sparse retest features are reported",
    )

    avoid_dataset = {
        "X": np.column_stack([
            np.linspace(0.0, 1.0, 100),
            np.linspace(1.0, 0.0, 100),
            np.linspace(0.2, 0.8, 100),
        ]).astype(np.float32),
        "feature_names": np.asarray([
            "local_chop_score",
            "short_reclaim_range_risk",
            "btc_eth_long_contradiction",
        ]),
    }
    avoid_mask, _, _ = link_avoid_mask(avoid_dataset, trained, "avoid_by_chop_reclaim")
    fraction = int(avoid_mask.sum()) / 100.0
    assert_true(0.05 <= fraction <= 0.30, "avoid mask selects a 5%-30% risk bucket")


def test_run_serializes_without_artifacts_and_symbol_guard() -> None:
    bad_args = argparse.Namespace(symbol="ETHUSDT", feature_mode="selected_family", include_avoid_only=True)
    try:
        configs_for_args(bad_args)
    except ValueError:
        pass
    else:
        raise AssertionError("non-LINK symbol should be rejected")

    with tempfile.TemporaryDirectory() as temp:
        args = argparse.Namespace(
            symbol="LINKUSDT",
            out_dir=temp,
            lockbox_mode="last-block",
            lockbox_test_ratio=0.20,
            min_train_samples=10,
            min_test_samples=5,
            feature_mode="selected_family",
            fee_bps=8.0,
            slippage_bps=3.0,
            fast=True,
            strict=False,
            disable_cross_context=True,
            include_avoid_only=True,
            include_rolling_forward=False,
        )
        base = {
            "step": np.arange(50, dtype=np.int64),
            "X": np.ones((50, 3), dtype=np.float32),
            "feature_names": np.asarray(["local_trend_down_score", "btc_eth_long_contradiction", "short_room_to_fall_12"]),
        }
        targets = {
            "hit": np.asarray([0, 1] * 25, dtype=np.int8),
            "quality": np.linspace(-0.2, 0.4, 50).astype(np.float32),
            "danger": np.asarray([1, 0] * 25, dtype=np.int8),
            "mae": np.linspace(0.001, 0.010, 50).astype(np.float32),
            "mfe": np.linspace(0.010, 0.001, 50).astype(np.float32),
        }
        fake_trained = {
            "model_status": "trained",
            "train_samples": 30,
            "validation_samples": 10,
            "test_samples": 10,
            "test": np.arange(40, 50, dtype=np.int64),
            "hit_prob": np.linspace(0.1, 0.9, 10),
            "quality_pred": np.linspace(-0.1, 0.3, 10),
            "danger_prob": np.linspace(0.9, 0.1, 10),
            "target_auc": 0.60,
            "target_average_precision": 0.60,
            "quality_corr": 0.10,
            "danger_auc": 0.60,
            "danger_filter_usefulness": 0.10,
        }
        with patch("aegis_alpha.tools.research_link_short_alpha_n.load_signal_market", return_value=object()):
            with patch("aegis_alpha.tools.research_link_short_alpha_n.build_recent_dataset", return_value={"dataset": base}):
                with patch("aegis_alpha.tools.research_link_short_alpha_n.apply_feature_set", return_value=base):
                    with patch("aegis_alpha.tools.research_link_short_alpha_n.compute_alpha_target_arrays", return_value=targets):
                        with patch("aegis_alpha.tools.research_link_short_alpha_n._fit_fold_predictions", return_value=fake_trained):
                            report = run(args)
        assert_true(Path(report["paths"]["json"]).exists(), "JSON report serializes")
        assert_true(Path(report["paths"]["all_configs_csv"]).exists(), "CSV report serializes")
        assert_true("LINKUSDT" in json.dumps(report), "report is JSON-compatible")
        assert_true(report["model_artifacts_written"] is False, "no active models are written")
        assert_true(report["shadow_models_generated"] is False, "no shadow models are generated")
        assert_true(report["active_manifest_touched"] is False, "active manifest is untouched")
        assert_true(report["live_inference_changed"] is False, "live inference is untouched")


def run_all() -> None:
    test_classification_gates()
    test_masks_degrade_and_sparse()
    test_run_serializes_without_artifacts_and_symbol_guard()
    print("manual_research_link_short_alpha_n_tests_passed")


if __name__ == "__main__":
    run_all()
