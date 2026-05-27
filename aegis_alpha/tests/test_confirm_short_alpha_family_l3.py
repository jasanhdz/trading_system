#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import tempfile
from copy import deepcopy
from pathlib import Path
from unittest.mock import patch

import numpy as np

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from aegis_alpha.tools.confirm_short_alpha_family_l3 import (
    DEFAULT_FROZEN_ALPHA_CONFIGS,
    classify_l3_alpha_candidate,
    default_frozen_alpha_configs,
    load_frozen_alpha_configs,
    run,
    write_csv,
)
from aegis_alpha.tools.train_short_alpha_family_l2_research import selection_mask


def assert_true(value: bool, message: str) -> None:
    if not value:
        raise AssertionError(message)


def row(**updates: object) -> dict[str, object]:
    result: dict[str, object] = {
        "symbol": "BNBUSDT",
        "model_status": "trained",
        "feature_count": 12,
        "test_samples": 400,
        "selected_count": 40,
        "selected_target_lift": 0.06,
        "selected_quality_lift": 0.08,
        "selected_net_quality_lift": 0.06,
        "selected_p90_mae": 0.006,
        "baseline_p90_mae": 0.006,
        "selected_mae_danger_rate": 0.10,
        "baseline_mae_danger": 0.10,
        "target_auc": 0.58,
        "quality_corr": 0.06,
        "danger_filter_usefulness": 0.02,
    }
    result.update(updates)
    return result


def test_classification_and_symbol_specific_gates() -> None:
    assert_true(classify_l3_alpha_candidate(row()) == "L3_ALPHA_CONFIRMED", "strong row confirms")
    assert_true(
        classify_l3_alpha_candidate(row(target_auc=0.51, quality_corr=0.0)) == "L3_ALPHA_WEAK",
        "positive lifts with weak model signal remain weak",
    )
    assert_true(classify_l3_alpha_candidate(row(selected_target_lift=0.0)) == "L3_ALPHA_FAILED", "zero target lift fails")
    assert_true(classify_l3_alpha_candidate(row(selected_net_quality_lift=0.0)) == "L3_ALPHA_FAILED", "zero net quality fails")
    assert_true(classify_l3_alpha_candidate(row(test_samples=100)) == "L3_ALPHA_INSUFFICIENT_DATA", "small holdout is insufficient")
    assert_true(classify_l3_alpha_candidate(row(feature_count=0)) == "L3_ALPHA_INSUFFICIENT_DATA", "zero features are insufficient")
    assert_true(
        classify_l3_alpha_candidate(row(selected_net_quality_lift=0.02)) == "L3_ALPHA_FAILED",
        "BNB must exceed net quality floor",
    )
    assert_true(
        classify_l3_alpha_candidate(row(symbol="SOLUSDT", selected_target_lift=0.03)) == "L3_ALPHA_FAILED",
        "SOL must exceed target lift floor",
    )
    assert_true(
        classify_l3_alpha_candidate(row(symbol="XRPUSDT", selected_quality_lift=0.03)) == "L3_ALPHA_FAILED",
        "XRP must exceed quality lift floor",
    )


def test_frozen_configs_and_mode_are_not_mutated() -> None:
    before = deepcopy(DEFAULT_FROZEN_ALPHA_CONFIGS)
    configs = default_frozen_alpha_configs()
    assert_true(DEFAULT_FROZEN_ALPHA_CONFIGS == before, "defaults remain immutable")
    assert_true(len(configs) == 3, "only three L2 promising configs are frozen")
    assert_true(all(config["decision_mode"] == "hit_primary" for config in configs), "mode is frozen")
    mask = selection_mask(
        configs[0]["decision_mode"],
        np.asarray([0.1, 0.9, 0.2, 0.3]),
        np.asarray([0.8, 0.1, 0.2, 0.3]),
        np.asarray([0.1, 0.2, 0.3, 0.4]),
    )
    assert_true(bool(mask[1]), "frozen hit_primary mode is applied")
    with tempfile.TemporaryDirectory() as temp:
        path = Path(temp) / "configs.json"
        path.write_text(json.dumps({"configs": configs}), encoding="utf-8")
        selected = load_frozen_alpha_configs(str(path), "SOLUSDT,XRPUSDT")
    assert_true([item["symbol"] for item in selected] == ["SOLUSDT", "XRPUSDT"], "only frozen requested configs run")


def test_serialization_and_symbol_error_does_not_abort() -> None:
    with tempfile.TemporaryDirectory() as temp:
        csv_path = Path(temp) / "summary.csv"
        write_csv(csv_path, [{"symbol": "BNBUSDT", "side": "SHORT", "l3_status": "L3_ALPHA_CONFIRMED"}])
        assert_true(csv_path.exists(), "CSV serializes")
        args = argparse.Namespace(
            configs_json=None,
            symbols="BNBUSDT,SOLUSDT",
            out_dir=temp,
            lockbox_mode="last-block",
            lockbox_test_ratio=0.20,
            fold_count=4,
            min_train_samples=1000,
            min_test_samples=300,
            fee_bps=8.0,
            slippage_bps=3.0,
            strict=False,
            fast=True,
            disable_cross_context=False,
        )
        selected_dataset = {
            "X": np.ones((2000, 2), dtype=np.float32),
            "feature_names": np.asarray(["local_momentum_down_score", "local_breakdown_score"]),
            "missing_family_features": [],
        }

        def load_market(_path: str, *, symbol_override: str) -> object:
            if symbol_override == "SOLUSDT":
                raise RuntimeError("dataset unavailable")
            return object()

        confirmed = {
            **default_frozen_alpha_configs()[0],
            "model_status": "trained",
            "feature_count": 2,
            "test_samples": 400,
            "selected_count": 40,
            "selected_target_lift": 0.06,
            "selected_quality_lift": 0.08,
            "selected_net_quality_lift": 0.06,
            "selected_p90_mae": 0.006,
            "baseline_p90_mae": 0.006,
            "selected_mae_danger_rate": 0.1,
            "baseline_mae_danger": 0.1,
            "target_auc": 0.58,
            "quality_corr": 0.06,
            "danger_filter_usefulness": 0.0,
            "l3_status": "L3_ALPHA_CONFIRMED",
            "l3_reason": "confirmed",
            "recommended_next_step": "eligible_for_metadata_only_shadow_artifact_after_review",
        }
        with patch("aegis_alpha.tools.confirm_short_alpha_family_l3.load_signal_market", side_effect=load_market):
            with patch("aegis_alpha.tools.confirm_short_alpha_family_l3.build_recent_dataset", return_value={"dataset": {"step": np.arange(2000)}}):
                with patch("aegis_alpha.tools.confirm_short_alpha_family_l3.apply_feature_set", return_value=selected_dataset):
                    with patch("aegis_alpha.tools.confirm_short_alpha_family_l3.select_alpha_family_features", return_value=selected_dataset):
                        with patch("aegis_alpha.tools.confirm_short_alpha_family_l3.compute_alpha_target_arrays", return_value={}):
                            with patch("aegis_alpha.tools.confirm_short_alpha_family_l3.evaluate_frozen_config", return_value=(confirmed, [confirmed])):
                                report = run(args)
        assert_true(report["evaluated_config_count"] == 2, "no alternatives are evaluated")
        assert_true(len(report["errors"]) == 1, "symbol error does not abort run")
        assert_true(report["confirmed"][0]["symbol"] == "BNBUSDT", "valid frozen config remains")
        assert_true(Path(report["paths"]["json"]).exists(), "JSON serializes")
        assert_true(report["model_artifacts_written"] is False, "models are not written")
        assert_true(report["shadow_models_generated"] is False, "shadow is not generated")
        assert_true(report["active_manifest_touched"] is False, "active manifest is untouched")


def run_all() -> None:
    test_classification_and_symbol_specific_gates()
    test_frozen_configs_and_mode_are_not_mutated()
    test_serialization_and_symbol_error_does_not_abort()
    print("manual_confirm_short_alpha_family_l3_tests_passed")


if __name__ == "__main__":
    run_all()
