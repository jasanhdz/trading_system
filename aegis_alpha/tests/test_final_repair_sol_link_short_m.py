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

from aegis_alpha.tools.final_repair_sol_link_short_m import (
    classify_m_final_candidate,
    configs_for_symbols,
    decision_mask,
    run,
    select_best_m_by_symbol,
    trend_confirmed_quality_mask,
)


def assert_true(value: bool, message: str) -> None:
    if not value:
        raise AssertionError(message)


def row(**updates: object) -> dict[str, object]:
    result: dict[str, object] = {
        "symbol": "SOLUSDT",
        "decision_mode": "quality_primary",
        "model_status": "trained",
        "feature_count": 10,
        "test_samples": 500,
        "selected_count": 50,
        "selected_target_lift": 0.06,
        "selected_quality_lift": 0.08,
        "selected_net_quality_lift": 0.06,
        "selected_p90_mae": 0.005,
        "baseline_p90_mae": 0.005,
        "selected_mae_danger_rate": 0.10,
        "baseline_mae_danger": 0.10,
        "target_auc": 0.56,
        "quality_corr": 0.02,
    }
    result.update(updates)
    return result


def test_classification_gates() -> None:
    assert_true(classify_m_final_candidate(row()) == "M_FINAL_CONFIRMED", "SOL positive row confirms")
    assert_true(classify_m_final_candidate(row(selected_quality_lift=0.0)) == "M_FINAL_FAILED", "zero quality fails")
    assert_true(classify_m_final_candidate(row(selected_net_quality_lift=0.0)) == "M_FINAL_FAILED", "zero net quality fails")
    assert_true(classify_m_final_candidate(row(selected_target_lift=0.04)) == "M_FINAL_WEAK", "SOL must exceed target floor")
    assert_true(classify_m_final_candidate(row(selected_net_quality_lift=0.03)) == "M_FINAL_WEAK", "SOL must exceed net floor")
    assert_true(
        classify_m_final_candidate(row(symbol="LINKUSDT", selected_quality_lift=0.05)) == "M_FINAL_WEAK",
        "LINK must exceed quality floor",
    )
    assert_true(
        classify_m_final_candidate(row(symbol="LINKUSDT", selected_quality_lift=0.07, selected_p90_mae=0.0055)) == "M_FINAL_WEAK",
        "LINK p90 cannot be worse than baseline for confirmation",
    )
    assert_true(
        classify_m_final_candidate(row(symbol="LINKUSDT", selected_quality_lift=0.07, selected_net_quality_lift=0.06)) == "M_FINAL_CONFIRMED",
        "LINK confirms when stricter gates pass",
    )
    assert_true(
        classify_m_final_candidate(row(
            decision_mode="avoid_only_candidate",
            avoid_quality_delta_vs_baseline=-0.05,
            avoid_mae_danger_delta=0.05,
            avoid_p90_mae_delta=0.001,
        )) == "M_FINAL_AVOID_ONLY",
        "avoid-only detects bad zone",
    )


def test_selection_modes_and_best_priority() -> None:
    trained = {
        "hit_prob": np.arange(100, dtype=np.float32),
        "quality_pred": np.arange(100, dtype=np.float32),
        "danger_prob": np.linspace(0.0, 1.0, 100, dtype=np.float32),
        "test": np.arange(100, dtype=np.int64),
    }
    dataset = {
        "X": np.ones((100, 2), dtype=np.float32),
        "feature_names": np.asarray(["local_trend_down_score", "noise"]),
    }
    top, _ = decision_mask("top_bucket_only", trained, dataset)
    assert_true(int(top.sum()) == 10, "top_bucket_only selects top ten percent")
    degraded, missing = trend_confirmed_quality_mask(dataset, trained["test"], trained["quality_pred"])
    assert_true(int(degraded.sum()) == 10, "trend mode degrades to quality when features missing")
    assert_true("btc_eth_long_contradiction" in missing, "missing trend features reported")
    best = select_best_m_by_symbol([
        row(symbol="LINKUSDT", m_status="M_FINAL_AVOID_ONLY", decision_mode="avoid_only_candidate", avoid_usefulness_score=10),
        row(symbol="LINKUSDT", m_status="M_FINAL_CONFIRMED", decision_mode="quality_primary", selected_net_quality_lift=0.01),
    ])
    assert_true(best[0]["m_status"] == "M_FINAL_CONFIRMED", "confirmed entry beats avoid-only")


def test_configs_and_serialization() -> None:
    configs = configs_for_symbols({"SOLUSDT", "LINKUSDT"}, "selected_family")
    assert_true(len(configs) == 8, "four configs per symbol")
    assert_true(all(config["lookback_days"] == 30 for config in configs), "no 7d/14d oversearch")
    with tempfile.TemporaryDirectory() as temp:
        args = argparse.Namespace(
            symbols="SOLUSDT,LINKUSDT",
            out_dir=temp,
            lockbox_mode="last-block",
            lockbox_test_ratio=0.20,
            min_train_samples=1000,
            min_test_samples=300,
            feature_mode="selected_family",
            fee_bps=8.0,
            slippage_bps=3.0,
            fast=True,
            strict=False,
            disable_cross_context=True,
            include_avoid_only=True,
        )
        dataset = {"step": np.arange(2000), "X": np.ones((2000, 2), dtype=np.float32)}
        selected = {
            "step": np.arange(2000),
            "X": np.ones((2000, 2), dtype=np.float32),
            "feature_names": np.asarray(["local_momentum_down_score", "local_breakdown_score"]),
            "missing_family_features": [],
        }

        def load_market(_path: str, *, symbol_override: str) -> object:
            if symbol_override == "LINKUSDT":
                raise RuntimeError("missing dataset")
            return object()

        fake_row = {
            **configs[0],
            "decision_mode": "quality_primary",
            "model_status": "trained",
            "test_samples": 400,
            "feature_count": 2,
            "m_status": "M_FINAL_CONFIRMED",
            "m_reason": "confirmed",
            "recommended_next_step": "eligible_for_metadata_only_shadow_candidate_as_final_repair_after_review",
        }
        with patch("aegis_alpha.tools.final_repair_sol_link_short_m.load_signal_market", side_effect=load_market):
            with patch("aegis_alpha.tools.final_repair_sol_link_short_m.build_recent_dataset", return_value={"dataset": dataset}):
                with patch("aegis_alpha.tools.final_repair_sol_link_short_m.apply_feature_set", return_value=selected):
                    with patch("aegis_alpha.tools.final_repair_sol_link_short_m.select_alpha_family_features", return_value=selected):
                        with patch("aegis_alpha.tools.final_repair_sol_link_short_m.compute_alpha_target_arrays", return_value={}):
                            with patch("aegis_alpha.tools.final_repair_sol_link_short_m.evaluate_config", return_value=([fake_row], [fake_row])):
                                report = run(args)
        assert_true(Path(report["paths"]["json"]).exists(), "JSON report serializes")
        assert_true(Path(report["paths"]["all_configs_csv"]).exists(), "CSV report serializes")
        assert_true(len(report["errors"]) >= 1, "symbol error is recorded without abort")
        assert_true(report["model_artifacts_written"] is False, "no active models are written")
        assert_true(report["shadow_models_generated"] is False, "no shadow models are generated")
        assert_true(report["active_manifest_touched"] is False, "active manifest is untouched")


def run_all() -> None:
    test_classification_gates()
    test_selection_modes_and_best_priority()
    test_configs_and_serialization()
    print("manual_final_repair_sol_link_short_m_tests_passed")


if __name__ == "__main__":
    run_all()
