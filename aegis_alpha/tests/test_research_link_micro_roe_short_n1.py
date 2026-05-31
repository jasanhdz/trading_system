#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from aegis_alpha.tools.research_link_micro_roe_short_n1 import (  # noqa: E402
    classify_link_micro_roe_candidate,
    compute_micro_roe_short_targets,
    run,
    select_best_link_micro_roe,
    select_link_micro_roe_features,
)


def assert_true(value: bool, message: str) -> None:
    if not value:
        raise AssertionError(message)


def market_fixture() -> SimpleNamespace:
    close = np.full(20, 100.0, dtype=np.float32)
    high = np.full(20, 100.1, dtype=np.float32)
    low = np.full(20, 99.9, dtype=np.float32)
    low[2] = 99.4
    high[4] = 100.5
    high[7] = 100.4
    low[7] = 99.4
    return SimpleNamespace(close=close, high=high, low=low)


def row(**updates: object) -> dict[str, object]:
    result: dict[str, object] = {
        "symbol": "LINKUSDT",
        "side": "SHORT",
        "alpha_family": "micro_scalp_short",
        "feature_set": "combined_v3",
        "feature_mode": "selected_family",
        "lookback_days": 30,
        "leverage": 20.0,
        "target_roe": 0.10,
        "stop_roe": 0.06,
        "horizon_candles": 6,
        "decision_mode": "micro_consensus",
        "model_status": "trained",
        "feature_count": 8,
        "test_samples": 600,
        "selected_count": 60,
        "selected_fraction": 0.10,
        "selected_micro_hit_lift": 0.08,
        "selected_micro_quality_lift": 0.08,
        "selected_net_micro_quality_lift": 0.058,
        "baseline_micro_stop_rate": 0.20,
        "selected_micro_stop_rate": 0.20,
        "selected_micro_stop_delta": 0.0,
        "baseline_p90_mae_roe": 0.08,
        "selected_p90_mae_roe": 0.07,
        "selected_p90_mae_delta": -0.01,
        "selected_avg_time_to_target": 3.0,
        "baseline_ambiguous_rate": 0.02,
        "selected_ambiguous_rate": 0.02,
        "cost_to_edge_ratio": 0.25,
        "micro_hit_auc": 0.56,
        "micro_quality_corr": 0.01,
    }
    result.update(updates)
    return result


def test_micro_roe_targets_short_path() -> None:
    targets = compute_micro_roe_short_targets(
        market_fixture(),
        np.asarray([1], dtype=np.int64),
        leverage=20,
        target_roe=0.10,
        stop_roe=0.06,
        horizon=6,
    )
    assert_true(float(targets["target_price_move"][0]) == np.float32(0.005), "+10% ROE / 20x is 0.5% move")
    assert_true(int(targets["micro_hit_before_stop"][0]) == 1, "short target is hit before stop")
    assert_true(int(targets["micro_stop_before_hit"][0]) == 0, "stop is not first")
    conflict = compute_micro_roe_short_targets(
        market_fixture(),
        np.asarray([6], dtype=np.int64),
        leverage=20,
        target_roe=0.10,
        stop_roe=0.06,
        horizon=2,
    )
    assert_true(int(conflict["micro_ambiguous_hit_stop"][0]) == 1, "same-candle conflict is marked ambiguous")
    assert_true(int(conflict["micro_hit_before_stop"][0]) == 0, "same-candle conflict does not count as hit")
    assert_true(int(conflict["micro_stop_before_hit"][0]) == 1, "same-candle conflict counts as stop first")


def test_classification_and_best_priority() -> None:
    assert_true(classify_link_micro_roe_candidate(row()) == "LINK_MICRO_ROE_PROMISING", "strong micro row is promising")
    assert_true(
        classify_link_micro_roe_candidate(row(cost_to_edge_ratio=0.55)) == "LINK_MICRO_ROE_WEAK",
        "costs consuming edge make candidate weak",
    )
    assert_true(
        classify_link_micro_roe_candidate(row(selected_net_micro_quality_lift=0.0)) == "LINK_MICRO_ROE_FAILED",
        "non-positive net micro quality fails",
    )
    avoid = row(
        decision_mode="avoid_by_micro_danger",
        avoid_selected_fraction=0.12,
        avoid_quality_delta_vs_baseline=-0.05,
        avoid_stop_rate_delta=0.08,
        avoid_p90_mae_delta=0.02,
        avoid_hit_rate_delta=-0.03,
        avoid_usefulness_score=0.18,
    )
    assert_true(classify_link_micro_roe_candidate(avoid) == "LINK_MICRO_ROE_AVOID_ONLY", "avoid-only confirms bad zone")
    best = select_best_link_micro_roe([
        {**avoid, "n1_status": "LINK_MICRO_ROE_AVOID_ONLY"},
        {**row(), "n1_status": "LINK_MICRO_ROE_PROMISING"},
    ])
    assert_true(best["n1_status"] == "LINK_MICRO_ROE_PROMISING", "promising entry beats avoid-only")


def test_feature_selection_proxies() -> None:
    dataset = {
        "X": np.ones((10, 4), dtype=np.float32),
        "feature_names": np.asarray(["return_15m", "upper_wick_ratio", "close_location_12", "volume_ratio_12"]),
    }
    selected = select_link_micro_roe_features(dataset, "pullback_rejection_short", "selected_family")
    assert_true(selected["X"].shape[1] >= 3, "available features and proxies are selected")
    assert_true(any(item.startswith("pullback_strength_3<-") for item in selected["proxy_features_used"]), "proxy usage is reported")
    assert_true("local_trend_down_score" in selected["missing_family_features"], "missing features are reported")


def test_run_serializes_without_artifacts_and_symbol_guard() -> None:
    with tempfile.TemporaryDirectory() as temp:
        args = argparse.Namespace(
            symbol="LINKUSDT",
            out_dir=temp,
            lockbox_mode="last-block",
            lockbox_test_ratio=0.20,
            min_train_samples=10,
            min_test_samples=5,
            feature_mode="selected_family",
            leverage=20.0,
            fee_bps=8.0,
            slippage_bps=3.0,
            fast=True,
            strict=False,
            disable_cross_context=True,
            include_avoid_only=True,
        )
        base = {
            "step": np.arange(50, dtype=np.int64),
            "X": np.ones((50, 5), dtype=np.float32),
            "feature_names": np.asarray([
                "realized_vol_24",
                "range_expansion_12",
                "volume_ratio_12",
                "close_location_12",
                "upper_wick_ratio",
            ]),
        }
        fake_targets = {
            "micro_hit_before_stop": np.asarray([0, 1] * 25, dtype=np.int8),
            "micro_stop_before_hit": np.asarray([1, 0] * 25, dtype=np.int8),
            "micro_mfe": np.linspace(0.01, 0.12, 50).astype(np.float32),
            "micro_mae": np.linspace(0.02, 0.09, 50).astype(np.float32),
            "micro_trade_quality": np.linspace(-0.2, 0.3, 50).astype(np.float32),
            "micro_time_to_target": np.ones(50, dtype=np.float32) * 3,
            "micro_time_to_stop": np.ones(50, dtype=np.float32) * 4,
            "micro_ambiguous_hit_stop": np.zeros(50, dtype=np.int8),
            "micro_net_roe_after_costs": np.linspace(-0.08, 0.08, 50).astype(np.float32),
            "cost_proxy_roe": np.asarray([0.022], dtype=np.float32),
        }
        fake_trained = {
            "model_status": "trained",
            "train_samples": 30,
            "validation_samples": 10,
            "test_samples": 10,
            "test": np.arange(40, 50, dtype=np.int64),
            "micro_hit_prob": np.linspace(0.1, 0.9, 10),
            "micro_quality_pred": np.linspace(-0.1, 0.3, 10),
            "micro_danger_prob": np.linspace(0.9, 0.1, 10),
            "micro_hit_auc": 0.60,
            "micro_hit_average_precision": 0.60,
            "micro_quality_corr": 0.10,
            "micro_danger_auc": 0.60,
            "danger_filter_usefulness": 0.10,
        }
        with patch("aegis_alpha.tools.research_link_micro_roe_short_n1.load_signal_market", return_value=market_fixture()):
            with patch("aegis_alpha.tools.research_link_micro_roe_short_n1.build_recent_dataset", return_value={"dataset": base}):
                with patch("aegis_alpha.tools.research_link_micro_roe_short_n1.apply_feature_set", return_value=base):
                    with patch("aegis_alpha.tools.research_link_micro_roe_short_n1.compute_micro_roe_short_targets", return_value=fake_targets):
                        with patch("aegis_alpha.tools.research_link_micro_roe_short_n1.fit_micro_models", return_value=fake_trained):
                            report = run(args)
        assert_true(Path(report["paths"]["json"]).exists(), "JSON report serializes")
        assert_true(Path(report["paths"]["all_configs_csv"]).exists(), "CSV report serializes")
        assert_true("LINKUSDT" in json.dumps(report), "report is JSON-compatible")
        assert_true(report["model_artifacts_written"] is False, "no model artifacts are written")
        assert_true(report["active_manifest_touched"] is False, "active manifest is untouched")
        assert_true(report["shadow_models_generated"] is False, "no shadow models are generated")
    try:
        run(argparse.Namespace(**{**vars(args), "symbol": "ETHUSDT"}))
    except ValueError:
        pass
    else:
        raise AssertionError("non-LINK symbols must be rejected")


def run_all() -> None:
    test_micro_roe_targets_short_path()
    test_classification_and_best_priority()
    test_feature_selection_proxies()
    test_run_serializes_without_artifacts_and_symbol_guard()
    print("manual_research_link_micro_roe_short_n1_tests_passed")


if __name__ == "__main__":
    run_all()
