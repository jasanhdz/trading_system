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

from aegis_alpha.tools.confirm_repaired_short_v3_lockbox import (
    DEFAULT_FROZEN_REPAIRED_CONFIGS,
    classify_repair_lockbox_candidate,
    default_frozen_repaired_configs,
    load_frozen_repaired_configs,
    run,
    write_csv,
)
from aegis_alpha.tools.repair_short_v3_failure_modes import selection_mask


def assert_true(value: bool, message: str) -> None:
    if not value:
        raise AssertionError(message)


def row(**updates: object) -> dict[str, object]:
    result: dict[str, object] = {
        "symbol": "ADAUSDT",
        "model_status": "trained",
        "test_samples": 400,
        "selected_count": 40,
        "selected_hit8_lift": 0.05,
        "selected_quality_lift": 0.06,
        "selected_net_quality_lift_after_cost": 0.05,
        "selected_p90_mae": 0.006,
        "baseline_p90_mae": 0.006,
        "selected_mae_danger_rate": 0.10,
        "baseline_mae_danger": 0.10,
    }
    result.update(updates)
    return result


def test_classification_and_strict_symbol_gates() -> None:
    assert_true(
        classify_repair_lockbox_candidate(row()) == "REPAIR_LOCKBOX_CONFIRMED",
        "positive frozen repair confirms",
    )
    assert_true(
        classify_repair_lockbox_candidate(row(selected_hit8_lift=0.03, selected_quality_lift=0.0)) == "REPAIR_LOCKBOX_FAILED",
        "quality not improved fails",
    )
    assert_true(
        classify_repair_lockbox_candidate(row(selected_hit8_lift=0.0)) == "REPAIR_LOCKBOX_FAILED",
        "zero hit8 lift fails",
    )
    assert_true(
        classify_repair_lockbox_candidate(row(selected_net_quality_lift_after_cost=0.0)) == "REPAIR_LOCKBOX_FAILED",
        "zero net quality fails",
    )
    assert_true(
        classify_repair_lockbox_candidate(row(selected_count=10)) == "REPAIR_LOCKBOX_WEAK",
        "small but nonempty selection is weak",
    )
    assert_true(
        classify_repair_lockbox_candidate(row(test_samples=100)) == "REPAIR_LOCKBOX_INSUFFICIENT_DATA",
        "insufficient holdout is reported",
    )
    assert_true(
        classify_repair_lockbox_candidate(row(symbol="DOGEUSDT", selected_hit8_lift=0.03)) == "REPAIR_LOCKBOX_FAILED",
        "DOGE must exceed strict hit8 floor",
    )
    assert_true(
        classify_repair_lockbox_candidate(row(symbol="DOGEUSDT", selected_hit8_lift=0.04)) == "REPAIR_LOCKBOX_CONFIRMED",
        "DOGE can confirm beyond strict hit8 floor",
    )
    assert_true(
        classify_repair_lockbox_candidate(row(symbol="BTCUSDT", selected_hit8_lift=0.01, selected_quality_lift=0.04)) == "REPAIR_LOCKBOX_FAILED",
        "BTC fragile hit8 floor is strict",
    )
    assert_true(
        classify_repair_lockbox_candidate(row(symbol="BTCUSDT", selected_hit8_lift=0.02, selected_quality_lift=0.021)) == "REPAIR_LOCKBOX_CONFIRMED",
        "BTC can confirm only past both strict floors",
    )
    assert_true(
        classify_repair_lockbox_candidate(row(selected_hit8_lift=0.005, selected_quality_lift=0.03), strict=True) == "REPAIR_LOCKBOX_FAILED",
        "strict mode rejects marginal general hit8 lift",
    )


def test_frozen_configs_and_selection_are_not_mutated() -> None:
    before = deepcopy(DEFAULT_FROZEN_REPAIRED_CONFIGS)
    configs = default_frozen_repaired_configs()
    assert_true(DEFAULT_FROZEN_REPAIRED_CONFIGS == before, "module defaults remain frozen")
    assert_true(len(configs) == 5, "only five J.0 winners are frozen")
    assert_true(configs[0]["repair_mode"] == "top_bucket_only", "LINK mode retained")
    assert_true(configs[1]["repair_mode"] == "hit8_primary", "ADA mode retained")
    predictions = {
        "hit8": np.asarray([0.1, 0.2, 0.3, 0.9]),
        "quality": np.asarray([0.8, 0.2, 0.3, 0.4]),
        "danger": np.asarray([0.1, 0.2, 0.3, 0.4]),
    }
    hit_mask = selection_mask(predictions, configs[1]["repair_mode"])
    assert_true(bool(hit_mask[-1]), "selection uses frozen hit8 mode")
    with tempfile.TemporaryDirectory() as temp:
        path = Path(temp) / "configs.json"
        path.write_text(json.dumps({"configs": configs}), encoding="utf-8")
        selected = load_frozen_repaired_configs(str(path), "ADAUSDT,DOGEUSDT")
    assert_true([item["symbol"] for item in selected] == ["ADAUSDT", "DOGEUSDT"], "only requested frozen configs run")
    assert_true(selected[1]["warning"] == "previous_lockbox_failed", "frozen warning retained")


def test_serialization_and_symbol_error_do_not_abort() -> None:
    with tempfile.TemporaryDirectory() as temp:
        path = Path(temp) / "summary.csv"
        write_csv(path, [{"symbol": "ADAUSDT", "side": "SHORT", "repair_lockbox_status": "REPAIR_LOCKBOX_CONFIRMED"}])
        assert_true(path.exists(), "CSV serializes")
        args = argparse.Namespace(
            configs_json=None,
            symbols="ADAUSDT,DOGEUSDT",
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

        def load_market(_path: str, *, symbol_override: str) -> object:
            if symbol_override == "DOGEUSDT":
                raise RuntimeError("missing dataset")
            return object()

        confirmed = {
            **default_frozen_repaired_configs()[1],
            "lockbox_mode": "last-block",
            "repair_lockbox_status": "REPAIR_LOCKBOX_CONFIRMED",
            "repair_lockbox_reason": "fixed_mode_confirmed",
            "recommended_next_step": "eligible_for_metadata_only_shadow_artifact_after_review",
            "test_samples": 400,
        }
        with patch("aegis_alpha.tools.confirm_repaired_short_v3_lockbox.load_signal_market", side_effect=load_market):
            with patch(
                "aegis_alpha.tools.confirm_repaired_short_v3_lockbox.build_recent_dataset",
                return_value={"dataset": {"X": np.zeros((2000, 2))}},
            ):
                with patch("aegis_alpha.tools.confirm_repaired_short_v3_lockbox.apply_feature_set", return_value={"X": np.zeros((2000, 2))}):
                    with patch("aegis_alpha.tools.confirm_repaired_short_v3_lockbox._evaluate_frozen_config", return_value=(confirmed, [confirmed])):
                        report = run(args)
        assert_true(report["evaluated_config_count"] == 2, "no alternative configs are evaluated")
        assert_true(len(report["results"]) == 2, "one result per frozen symbol")
        assert_true(len(report["errors"]) == 1, "one failed symbol is retained as error")
        assert_true(report["confirmed"][0]["symbol"] == "ADAUSDT", "valid fixed repair survives")
        assert_true(report["save_models"] is False, "models are not saved")
        assert_true(report["shadow_models_generated"] is False, "shadow models are not generated")
        assert_true(report["active_manifest_touched"] is False, "active models stay untouched")


def run_all() -> None:
    test_classification_and_strict_symbol_gates()
    test_frozen_configs_and_selection_are_not_mutated()
    test_serialization_and_symbol_error_do_not_abort()
    print("manual_confirm_repaired_short_v3_lockbox_tests_passed")


if __name__ == "__main__":
    run_all()
