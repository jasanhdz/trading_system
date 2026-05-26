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

from aegis_alpha.tools.confirm_short_v3_lockbox import (
    DEFAULT_FROZEN_CONFIGS,
    build_last_block_fold,
    classify_lockbox_candidate,
    default_frozen_configs,
    load_frozen_configs,
    run,
    write_csv,
)


def assert_true(value: bool, message: str) -> None:
    if not value:
        raise AssertionError(message)


def row(
    *,
    test_samples: int = 400,
    hit8_lift: float = 0.06,
    quality_lift: float = 0.08,
    net_lift: float = 0.079,
    auc: float = 0.58,
    corr: float = 0.03,
    top_p90: float = 0.006,
    baseline_p90: float = 0.006,
) -> dict[str, object]:
    return {
        "model_status": "trained",
        "test_samples": test_samples,
        "hit8_top_decile_lift": hit8_lift,
        "quality_top_decile_lift": quality_lift,
        "net_quality_lift_after_cost_proxy": net_lift,
        "latest_fold_quality_lift": quality_lift,
        "hit8_auc": auc,
        "quality_corr": corr,
        "top_decile_p90_mae": top_p90,
        "baseline_p90_mae": baseline_p90,
        "danger_filter_usefulness": 0.02,
    }


def fake_folds() -> list[dict[str, object]]:
    return [{
        "fold": 1,
        "model_status": "trained",
        "split_samples": {"train": 1200, "validation": 300, "test": 400},
        "baseline_test": {"hit8_rate": 0.20, "avg_trade_quality": 0.01, "mae_danger_rate": 0.15, "p90_mae": 0.006},
        "families": {
            "hit8_classifier": {
                "test_metrics": {"roc_auc": 0.58, "average_precision": 0.42},
                "top_decile": {"hit8_lift_vs_baseline": 0.06, "hit8_rate": 0.26},
            },
            "trade_quality_regressor": {
                "test_metrics": {"spearman": 0.03},
                "top_decile": {
                    "quality_lift_vs_baseline": 0.08,
                    "avg_trade_quality": 0.09,
                    "mae_danger_rate": 0.10,
                    "p90_mae": 0.006,
                    "p90_mae_delta_vs_baseline": 0.0,
                },
            },
            "mae_danger_classifier": {
                "test_metrics": {"roc_auc": 0.57},
                "top_decile": {"mae_danger_rate": 0.18},
                "usefulness_as_filter": 0.02,
            },
        },
    }]


def test_classification() -> None:
    assert_true(classify_lockbox_candidate(row()) == "LOCKBOX_CONFIRMED", "strong lockbox confirmation")
    assert_true(
        classify_lockbox_candidate(row(auc=0.51)) == "LOCKBOX_WEAK",
        "partial improvement should be weak",
    )
    assert_true(
        classify_lockbox_candidate(row(hit8_lift=-0.03, quality_lift=-0.02, net_lift=-0.03)) == "LOCKBOX_FAILED",
        "negative lockbox should fail",
    )
    assert_true(
        classify_lockbox_candidate(row(hit8_lift=-0.01, quality_lift=0.04, net_lift=0.03)) == "LOCKBOX_FAILED",
        "negative hit8 lift alone should fail",
    )
    assert_true(
        classify_lockbox_candidate(row(test_samples=100)) == "LOCKBOX_INSUFFICIENT_DATA",
        "small lockbox is insufficient",
    )


def test_frozen_configs_and_holdout_are_stable() -> None:
    frozen_before = deepcopy(DEFAULT_FROZEN_CONFIGS)
    configs = default_frozen_configs()
    assert_true(DEFAULT_FROZEN_CONFIGS == frozen_before, "default tuple is not mutated")
    assert_true(len(configs) == 9, "nine frozen candidates expected")
    assert_true(configs[0]["feature_set"] == "operable_v3", "frozen feature set retained")
    fold = build_last_block_fold(3000, test_ratio=0.20, min_train_samples=1000, min_test_samples=300)
    assert_true(fold is not None, "holdout should be constructed")
    assert_true(int(fold["train"][-1]) < int(fold["validation"][0]), "training precedes validation")
    assert_true(int(fold["validation"][-1]) < int(fold["test"][0]), "lockbox remains future holdout")
    assert_true(int(fold["test"][-1]) == 2999, "test is final temporal block")


def test_json_configs_filter_without_alternatives() -> None:
    with tempfile.TemporaryDirectory() as temp:
        path = Path(temp) / "configs.json"
        path.write_text(json.dumps({"configs": default_frozen_configs()}), encoding="utf-8")
        selected = load_frozen_configs(str(path), "ADAUSDT,DOGEUSDT")
    assert_true([item["symbol"] for item in selected] == ["ADAUSDT", "DOGEUSDT"], "symbol subset only")
    assert_true(selected[0]["horizon_candles"] == 24, "frozen ADA horizon retained")


def test_serialization_and_symbol_error_do_not_abort() -> None:
    with tempfile.TemporaryDirectory() as temp:
        csv_path = Path(temp) / "summary.csv"
        write_csv(csv_path, [{"symbol": "ADAUSDT", "side": "SHORT", "lockbox_status": "LOCKBOX_CONFIRMED"}])
        assert_true(csv_path.exists(), "CSV serializes")
        args = argparse.Namespace(
            configs_json=None,
            symbols="ADAUSDT,DOGEUSDT",
            out_dir=temp,
            model_dir=str(Path(temp) / "models" / "research"),
            fold_count=4,
            lockbox_mode="last-block",
            lockbox_test_ratio=0.20,
            min_train_samples=1000,
            min_test_samples=300,
            fee_bps=8.0,
            slippage_bps=3.0,
            strict=False,
            fast=True,
        )

        def load_market(_path: str, *, symbol_override: str) -> object:
            if symbol_override == "DOGEUSDT":
                raise RuntimeError("missing dataset")
            return object()

        with patch("aegis_alpha.tools.confirm_short_v3_lockbox.load_signal_market", side_effect=load_market):
            with patch(
                "aegis_alpha.tools.confirm_short_v3_lockbox.build_recent_dataset",
                return_value={"dataset": {"X": np.zeros((2000, 2)), "feature_names": np.asarray(["a", "b"])}},
            ):
                with patch("aegis_alpha.tools.confirm_short_v3_lockbox.apply_feature_set", return_value={"X": np.zeros((2000, 2))}):
                    with patch("aegis_alpha.tools.confirm_short_v3_lockbox._evaluate_config", return_value=(fake_folds(), {})):
                        report = run(args)
        assert_true(report["evaluated_config_count"] == 2, "frozen subset evaluated only")
        assert_true(len(report["errors"]) == 1, "one symbol error captured")
        assert_true(report["confirmed"][0]["symbol"] == "ADAUSDT", "valid symbol survives")
        assert_true(report["save_models"] is False, "no models saved")
        assert_true(report["shadow_models_generated"] is False, "no shadow artifacts")
        assert_true(report["active_manifest_touched"] is False, "no active manifest")


def run_all() -> None:
    test_classification()
    test_frozen_configs_and_holdout_are_stable()
    test_json_configs_filter_without_alternatives()
    test_serialization_and_symbol_error_do_not_abort()
    print("manual_confirm_short_v3_lockbox_tests_passed")


if __name__ == "__main__":
    run_all()
