#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path
from unittest.mock import patch

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from aegis_alpha.tools.evaluate_short_operable_v2_matrix import (
    DEFAULT_SYMBOLS,
    SIDE,
    classify_short_config,
    parse_symbols,
    rank_short_configs,
    run,
    validate_research_model_dir,
    write_csv,
)


def assert_true(value: bool, message: str) -> None:
    if not value:
        raise AssertionError(message)


def row(
    *,
    symbol: str = "ADAUSDT",
    feature_set: str = "operable_v2",
    quality_lift: float = 0.12,
    hit8_lift: float = 0.06,
    latest_quality: float = 0.11,
    corr: float = 0.05,
    net_lift: float = 0.11,
    recommendation: str = "WALK_FORWARD_PROMISING",
) -> dict[str, object]:
    return {
        "symbol": symbol,
        "side": SIDE,
        "feature_set": feature_set,
        "lookback_days": 30,
        "horizon_candles": 12,
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
        "latest_fold_quality_p90_mae": 0.007,
        "latest_fold_baseline_p90_mae": 0.007,
        "latest_p90_mae_delta": 0.0,
        "v2_danger_auc_mean": 0.60,
        "danger_filter_usefulness_mean": 0.15,
        "net_quality_lift_after_cost_proxy": net_lift,
    }


def test_classification_statuses() -> None:
    assert_true(classify_short_config(row()) == "SHORT_STRONG_RESEARCH", "strong classification")
    mixed = row(quality_lift=-0.01, hit8_lift=0.05, latest_quality=-0.01, corr=-0.03, net_lift=-0.02)
    assert_true(classify_short_config(mixed) == "SHORT_MIXED_RESEARCH", "mixed classification")
    bad = row(quality_lift=-0.08, hit8_lift=-0.04, latest_quality=-0.06, corr=-0.05, net_lift=-0.09)
    assert_true(classify_short_config(bad) == "SHORT_BAD_RESEARCH", "bad classification")


def test_ranking_prioritizes_strong_status() -> None:
    strong = row(symbol="ADAUSDT")
    mixed = row(
        symbol="AVAXUSDT",
        quality_lift=1.0,
        hit8_lift=0.5,
        latest_quality=0.8,
        corr=-0.01,
        net_lift=0.9,
    )
    ranked = rank_short_configs([mixed, strong])
    assert_true(ranked[0]["symbol"] == "ADAUSDT", "strong status must rank above mixed score")
    assert_true(ranked[0]["short_research_status"] == "SHORT_STRONG_RESEARCH", "ranking status")


def test_serialization_scope_and_path_guard() -> None:
    ranked = rank_short_configs([row()])
    assert_true("ADAUSDT" in json.dumps(ranked), "JSON should serialize")
    with tempfile.TemporaryDirectory() as temp:
        path = Path(temp) / "summary.csv"
        write_csv(path, ranked)
        assert_true(path.exists(), "CSV should write for synthetic rows")
    assert_true(SIDE == "SHORT", "scope must remain SHORT only")
    assert_true("ADAUSDT" in DEFAULT_SYMBOLS, "default universe includes ADA")
    assert_true(parse_symbols("ADAUSDT,ADAUSDT,AVAXUSDT") == ["ADAUSDT", "AVAXUSDT"], "duplicates removed")
    validate_research_model_dir(Path("/tmp/models/research/turbo_v2_short_global_matrix"))
    try:
        validate_research_model_dir(Path("/tmp/models/active/turbo_v2_short_global_matrix"))
    except ValueError:
        return
    raise AssertionError("active model path must be rejected")


def test_symbol_error_does_not_abort_run() -> None:
    with tempfile.TemporaryDirectory() as temp:
        args = argparse.Namespace(
            symbols="ADAUSDT,BADUSDT",
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
            include_reference_feature_sets=False,
            strong_only_report=False,
            max_configs=None,
            skip_existing=False,
            fast=True,
        )

        def load_market(_path: str, *, symbol_override: str) -> object:
            if symbol_override == "BADUSDT":
                raise RuntimeError("dataset unavailable")
            return object()

        result = {
            "summary": row(),
            "folds": [],
        }
        with patch("aegis_alpha.tools.evaluate_short_operable_v2_matrix.load_signal_market", side_effect=load_market):
            with patch("aegis_alpha.tools.evaluate_short_operable_v2_matrix.build_recent_dataset", return_value={"dataset": {}}):
                with patch("aegis_alpha.tools.evaluate_short_operable_v2_matrix.apply_feature_set", return_value={}):
                    with patch("aegis_alpha.tools.evaluate_short_operable_v2_matrix.run_walk_forward", return_value=result):
                        report = run(args)
        assert_true(len(report["ranking"]) == 1, "healthy symbol should still be evaluated")
        assert_true(len(report["errors"]) == 1, "failed symbol should be reported")
        assert_true(report["errors"][0]["symbol"] == "BADUSDT", "error should retain symbol")
        assert_true(report["save_models"] is False, "matrix must not save models")
        assert_true(report["active_manifest_touched"] is False, "matrix must not touch active manifest")


def run_all() -> None:
    test_classification_statuses()
    test_ranking_prioritizes_strong_status()
    test_serialization_scope_and_path_guard()
    test_symbol_error_does_not_abort_run()
    print("manual_evaluate_short_operable_v2_matrix_tests_passed")


if __name__ == "__main__":
    run_all()
