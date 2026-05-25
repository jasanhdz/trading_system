#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from aegis_alpha.tools.optimize_short_operable_v2_configs import (
    SIDE,
    annotate_default_comparison,
    classify_optimization_config,
    parse_symbols,
    run,
    select_best_by_symbol,
    select_best_short_config_for_symbol,
    validate_research_model_dir,
    write_csv,
)


def assert_true(value: bool, message: str) -> None:
    if not value:
        raise AssertionError(message)


def row(
    *,
    symbol: str = "AVAXUSDT",
    feature_set: str = "operable_v2",
    lookback: int = 30,
    horizon: int = 12,
    quality_lift: float = 0.12,
    hit8_lift: float = 0.06,
    latest_quality: float = 0.11,
    corr: float = 0.05,
    p90: float = 0.007,
    baseline_p90: float = 0.007,
    net_lift: float = 0.11,
    recommendation: str = "WALK_FORWARD_PROMISING",
) -> dict[str, object]:
    return {
        "symbol": symbol,
        "side": SIDE,
        "feature_set": feature_set,
        "lookback_days": lookback,
        "horizon_candles": horizon,
        "fold_count": 4,
        "valid_fold_count": 4,
        "recommendation": recommendation,
        "stability_score": 0.72,
        "decay_score": 0.01,
        "baseline_hit8_mean": 0.13,
        "baseline_quality_mean": 0.03,
        "baseline_danger_mean": 0.22,
        "v2_hit8_auc_mean": 0.59,
        "v2_hit8_auc_min": 0.55,
        "hit8_top_decile_lift_mean": hit8_lift,
        "hit8_top_decile_lift_min": 0.01,
        "latest_fold_hit8_lift": 0.05,
        "v2_quality_corr_mean": corr,
        "v2_quality_corr_min": -0.01,
        "quality_top_decile_lift_mean": quality_lift,
        "quality_top_decile_lift_min": 0.01,
        "latest_fold_quality_lift": latest_quality,
        "latest_fold_quality_p90_mae": p90,
        "latest_fold_baseline_p90_mae": baseline_p90,
        "latest_p90_mae_delta": p90 - baseline_p90,
        "v2_danger_auc_mean": 0.60,
        "danger_filter_usefulness_mean": 0.15,
        "net_quality_lift_after_cost_proxy": net_lift,
    }


def test_best_prioritizes_strong() -> None:
    strong = row(feature_set="combined")
    mixed = row(feature_set="operable_v2", corr=-0.01, quality_lift=0.40, net_lift=0.39)
    best = select_best_short_config_for_symbol([mixed, strong])
    assert_true(best["best_status"] == "STRONG_BEST", "strong must outrank mixed")
    assert_true(best["feature_set"] == "combined", "alternate feature set can be strong in optimization")


def test_best_mixed_and_risk_penalties() -> None:
    positive = row(recommendation="WALK_FORWARD_MIXED", quality_lift=0.05, latest_quality=0.03)
    negative_latest = row(recommendation="WALK_FORWARD_MIXED", quality_lift=0.08, latest_quality=-0.02)
    high_p90 = row(recommendation="WALK_FORWARD_MIXED", quality_lift=0.09, p90=0.010, baseline_p90=0.007)
    best = select_best_short_config_for_symbol([negative_latest, high_p90, positive])
    assert_true(best["best_status"] == "MIXED_BEST", "mixed best expected")
    assert_true(best["latest_fold_quality_lift"] > 0.0, "negative latest fold should be penalized")


def test_prefer_simple_equivalent_configuration() -> None:
    combined = row(feature_set="combined", quality_lift=0.121)
    operable = row(feature_set="operable_v2", quality_lift=0.12)
    best = select_best_short_config_for_symbol([combined, operable])
    assert_true(best["feature_set"] == "operable_v2", "equivalent rows prefer operable_v2")


def test_grouping_and_promoted_status() -> None:
    default_mixed = row(symbol="AVAXUSDT", recommendation="WALK_FORWARD_MIXED", corr=-0.01)
    optimized_strong = row(symbol="AVAXUSDT", feature_set="base", lookback=14)
    other_bad = row(
        symbol="SOLUSDT",
        quality_lift=-0.08,
        hit8_lift=-0.05,
        latest_quality=-0.03,
        recommendation="WALK_FORWARD_BAD",
    )
    selected = annotate_default_comparison(
        select_best_by_symbol([default_mixed, optimized_strong, other_bad], ["AVAXUSDT", "SOLUSDT"]),
        [default_mixed, optimized_strong, other_bad],
    )
    assert_true(selected[0]["best_status"] == "STRONG_BEST", "optimized strong best")
    assert_true(selected[0]["promoted_from_default"] is True, "mixed default promoted to strong")
    assert_true(selected[1]["best_status"] == "BAD_BEST", "bad grouped status")


def test_serialization_presets_and_path_guard() -> None:
    assert_true(parse_symbols(None, "controls") == ["DOGEUSDT", "ADAUSDT", "SOLUSDT"], "controls preset")
    assert_true(parse_symbols("ADAUSDT,ADAUSDT", "all") == ["ADAUSDT"], "symbol dedupe")
    payload = [select_best_short_config_for_symbol([row()])]
    assert_true("STRONG_BEST" in json.dumps(payload), "JSON serialization")
    with tempfile.TemporaryDirectory() as temp:
        csv_path = Path(temp) / "best.csv"
        write_csv(csv_path, payload)
        assert_true(csv_path.exists(), "CSV serialization")
    validate_research_model_dir(Path("/tmp/models/research/turbo_v2_optimization"))
    try:
        validate_research_model_dir(Path("/tmp/models/active/turbo_v2_optimization"))
    except ValueError:
        return
    raise AssertionError("active model path must be rejected")


def test_symbol_error_does_not_abort_and_no_models_saved() -> None:
    with tempfile.TemporaryDirectory() as temp:
        args = argparse.Namespace(
            preset="mixed",
            symbols="AVAXUSDT,BADUSDT",
            feature_sets="operable_v2",
            lookback_days="30",
            horizons="12",
            fold_count=4,
            train_ratio=0.50,
            validation_ratio=0.15,
            test_ratio=0.15,
            min_train_samples=1000,
            min_test_samples=300,
            fee_bps=8.0,
            slippage_bps=3.0,
            out_dir=temp,
            model_dir=str(Path(temp) / "models" / "research"),
            max_configs=None,
            skip_existing=False,
            seed_report=None,
            fast=True,
        )

        def load_market(_path: str, *, symbol_override: str) -> object:
            if symbol_override == "BADUSDT":
                raise RuntimeError("dataset unavailable")
            return object()

        result = {"summary": row(), "folds": []}
        with patch("aegis_alpha.tools.optimize_short_operable_v2_configs.load_signal_market", side_effect=load_market):
            with patch("aegis_alpha.tools.optimize_short_operable_v2_configs.build_recent_dataset", return_value={"dataset": {}}):
                with patch("aegis_alpha.tools.optimize_short_operable_v2_configs.apply_feature_set", return_value={}):
                    with patch("aegis_alpha.tools.optimize_short_operable_v2_configs.run_walk_forward", return_value=result):
                        report = run(args)
        assert_true(report["configuration_count"] == 1, "healthy symbol evaluates")
        assert_true(len(report["errors"]) == 1, "symbol error captured")
        assert_true(report["save_models"] is False, "models not saved")
        assert_true(report["shadow_models_generated"] is False, "shadow artifacts not generated")
        assert_true(report["active_manifest_touched"] is False, "active manifest untouched")


def test_seed_report_is_merged_without_recomputing_seeded_rows() -> None:
    with tempfile.TemporaryDirectory() as temp:
        seed_path = Path(temp) / "seed.json"
        seed_path.write_text(json.dumps({"ranking": [row(symbol="BTCUSDT")], "errors": []}), encoding="utf-8")
        args = argparse.Namespace(
            preset="mixed",
            symbols="XRPUSDT",
            feature_sets="combined",
            lookback_days="14",
            horizons="12",
            fold_count=4,
            train_ratio=0.50,
            validation_ratio=0.15,
            test_ratio=0.15,
            min_train_samples=1000,
            min_test_samples=300,
            fee_bps=8.0,
            slippage_bps=3.0,
            out_dir=temp,
            model_dir=str(Path(temp) / "models" / "research"),
            max_configs=None,
            skip_existing=False,
            seed_report=str(seed_path),
            fast=True,
        )
        new_row = row(symbol="XRPUSDT", feature_set="combined", lookback=14)
        result = {"summary": new_row, "folds": []}
        with patch("aegis_alpha.tools.optimize_short_operable_v2_configs.load_signal_market", return_value=object()):
            with patch("aegis_alpha.tools.optimize_short_operable_v2_configs.build_recent_dataset", return_value={"dataset": {}}):
                with patch("aegis_alpha.tools.optimize_short_operable_v2_configs.apply_feature_set", return_value={}):
                    with patch("aegis_alpha.tools.optimize_short_operable_v2_configs.run_walk_forward", return_value=result):
                        report = run(args)
        assert_true(report["seeded_configuration_count"] == 1, "seeded row count")
        assert_true(report["configuration_count"] == 2, "seed and newly evaluated row combined")
        assert_true({item["symbol"] for item in report["best_by_symbol"]} == {"BTCUSDT", "XRPUSDT"}, "seed symbols retained")


def run_all() -> None:
    assert_true(classify_optimization_config(row()) == "SHORT_STRONG_RESEARCH", "strong classification")
    test_best_prioritizes_strong()
    test_best_mixed_and_risk_penalties()
    test_prefer_simple_equivalent_configuration()
    test_grouping_and_promoted_status()
    test_serialization_presets_and_path_guard()
    test_symbol_error_does_not_abort_and_no_models_saved()
    test_seed_report_is_merged_without_recomputing_seeded_rows()
    print("manual_optimize_short_operable_v2_configs_tests_passed")


if __name__ == "__main__":
    run_all()
