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

from aegis_alpha.tools.train_short_alpha_family_l2_research import (
    classify_l2_alpha_candidate,
    compute_alpha_target_arrays,
    run,
    select_alpha_family_features,
    select_best_l2_alpha_by_symbol,
    selection_mask,
)


def assert_true(value: bool, message: str) -> None:
    if not value:
        raise AssertionError(message)


def fixture_row(status: str, **updates: object) -> dict[str, object]:
    row: dict[str, object] = {
        "symbol": "LINKUSDT",
        "side": "SHORT",
        "alpha_family": "slow_trend_short",
        "feature_set": "combined_v3",
        "feature_mode": "selected_family",
        "lookback_days": 30,
        "horizon_candles": 12,
        "target_name": "hit5_before_minus3",
        "decision_mode": "quality_primary",
        "model_status": "trained",
        "test_samples": 600,
        "selected_count": 60,
        "selected_target_lift": 0.08,
        "selected_quality_lift": 0.12,
        "selected_net_quality_lift": 0.11,
        "baseline_p90_mae": 0.010,
        "selected_p90_mae": 0.0105,
        "selected_p90_mae_delta": 0.0005,
        "selected_mae_danger_delta": 0.02,
        "target_auc": 0.58,
        "quality_corr": 0.06,
        "l2_status": status,
    }
    row.update(updates)
    return row


def subject_market() -> SimpleNamespace:
    close = np.full(80, 100.0, dtype=np.float32)
    high = np.full(80, 100.1, dtype=np.float32)
    low = np.full(80, 99.9, dtype=np.float32)
    low[41] = 99.4
    low[42] = 99.2
    high[43] = 100.5
    timestamps = np.asarray([f"2026-05-01T00:{idx:02d}:00" for idx in range(80)])
    return SimpleNamespace(
        cfg=SimpleNamespace(risk=SimpleNamespace(total_fee=0.0004)),
        close=close,
        high=high,
        low=low,
        timestamps=timestamps,
        features=np.ones((80, 21), dtype=np.float32),
    )


def test_select_alpha_family_features_and_missing() -> None:
    dataset = {
        "X": np.asarray([[1.0, 2.0, 3.0], [3.0, 4.0, 5.0]], dtype=np.float32),
        "feature_names": np.asarray(["local_trend_down_score", "short_room_to_fall_12", "noise"]),
    }
    selected = select_alpha_family_features(dataset, "slow_trend_short")
    assert_true(selected["X"].shape == (2, 2), "family feature subset is retained")
    assert_true("ema_stack_bearish" in selected["missing_family_features"], "missing family fields are reported")
    all_features = select_alpha_family_features(dataset, "slow_trend_short", "combined_v3_all")
    assert_true(all_features["X"].shape == (2, 3), "all feature mode keeps input matrix")


def test_alternative_target_is_path_aware() -> None:
    targets = compute_alpha_target_arrays(
        subject_market(),
        np.asarray([40], dtype=np.int64),
        side="SHORT",
        target_name="hit5_before_minus3",
        horizon=12,
        cost_proxy=0.0011,
    )
    assert_true(int(targets["hit"][0]) == 1, "SHORT target hit is found before stop")
    assert_true(float(targets["time_to_target"][0]) == 1.0, "path timing is preserved")
    assert_true(np.all(np.isfinite(targets["quality"])), "target quality is finite")


def test_classification_and_best_selection() -> None:
    promising = fixture_row("ignored")
    assert_true(classify_l2_alpha_candidate(promising) == "L2_ALPHA_PROMISING", "positive candidate is promising")
    mixed = fixture_row("ignored", selected_quality_lift=-0.01, selected_net_quality_lift=0.01)
    assert_true(classify_l2_alpha_candidate(mixed) == "L2_ALPHA_BAD", "negative quality is bad")
    mixed = fixture_row("ignored", target_auc=0.51, quality_corr=0.0)
    assert_true(classify_l2_alpha_candidate(mixed) == "L2_ALPHA_MIXED", "weak model evidence is mixed")
    bad = fixture_row("ignored", selected_net_quality_lift=-0.01)
    assert_true(classify_l2_alpha_candidate(bad) == "L2_ALPHA_BAD", "negative net quality is bad")
    best = select_best_l2_alpha_by_symbol([
        fixture_row("L2_ALPHA_MIXED", selected_net_quality_lift=1.0),
        fixture_row("L2_ALPHA_PROMISING", target_name="hit3_before_minus2", selected_net_quality_lift=0.1),
    ])
    assert_true(best[0]["l2_status"] == "L2_ALPHA_PROMISING", "promising dominates mixed")


def test_selection_masks() -> None:
    hit = np.arange(100, dtype=np.float32)
    quality = np.arange(100, dtype=np.float32)
    danger = np.linspace(0.0, 1.0, 100, dtype=np.float32)
    hit_mask = selection_mask("hit_primary", hit, quality, danger)
    filtered = selection_mask("quality_primary_danger_filtered", hit, quality, danger)
    assert_true(int(hit_mask.sum()) == 10, "top bucket selects ten percent")
    assert_true(int(filtered.sum()) == 10, "filtered selection keeps requested sample size where possible")
    assert_true(np.max(np.flatnonzero(filtered)) < 90, "danger filter removes highest danger rows")


def test_run_serializes_without_artifacts_and_continues_on_error() -> None:
    with tempfile.TemporaryDirectory() as temp:
        args = argparse.Namespace(
            symbols="LINKUSDT,SOLUSDT",
            lookback_days=None,
            horizons="12",
            feature_mode="selected_family",
            lockbox_mode="last-block",
            lockbox_test_ratio=0.20,
            fold_count=4,
            min_train_samples=10,
            min_test_samples=5,
            fee_bps=8.0,
            slippage_bps=3.0,
            out_dir=temp,
            fast=True,
        )
        market = subject_market()
        base = {"step": np.arange(10, 60, dtype=np.int64)}
        combined = {
            "X": np.ones((50, 1), dtype=np.float32),
            "feature_names": np.asarray(["local_trend_down_score"]),
        }

        def load_market(_path: str, *, symbol_override: str) -> object:
            if symbol_override == "SOLUSDT":
                raise RuntimeError("dataset unavailable")
            return market

        fake_fold = {
            "model_status": "trained",
            "train_samples": 30,
            "validation_samples": 10,
            "test_samples": 10,
            "test": np.arange(40, 50, dtype=np.int64),
            "hit_prob": np.linspace(0.1, 0.9, 10),
            "quality_pred": np.linspace(-0.1, 0.3, 10),
            "danger_prob": np.linspace(0.1, 0.8, 10),
            "target_auc": 0.60,
            "target_average_precision": 0.60,
            "quality_corr": 0.10,
            "danger_auc": 0.60,
            "danger_filter_usefulness": 0.10,
        }
        with patch("aegis_alpha.tools.train_short_alpha_family_l2_research.load_signal_market", side_effect=load_market):
            with patch("aegis_alpha.tools.train_short_alpha_family_l2_research.build_recent_dataset", return_value={"dataset": base}):
                with patch("aegis_alpha.tools.train_short_alpha_family_l2_research.apply_feature_set", return_value=combined):
                    with patch("aegis_alpha.tools.train_short_alpha_family_l2_research._fit_fold_predictions", return_value=fake_fold):
                        report = run(args)
        assert_true(Path(report["paths"]["json"]).exists(), "JSON report serializes")
        assert_true(Path(report["paths"]["all_configs_csv"]).exists(), "CSV report serializes")
        assert_true(len(report["errors"]) == 1, "one symbol failure does not abort run")
        assert_true(report["model_artifacts_written"] is False, "no model artifact is written")
        assert_true(report["shadow_models_generated"] is False, "no shadow artifacts are generated")
        assert_true(report["active_manifest_touched"] is False, "active is not touched")
        assert_true("LINKUSDT" in json.dumps(report), "report remains JSON compatible")


def run_all() -> None:
    test_select_alpha_family_features_and_missing()
    test_alternative_target_is_path_aware()
    test_classification_and_best_selection()
    test_selection_masks()
    test_run_serializes_without_artifacts_and_continues_on_error()
    print("manual_train_short_alpha_family_l2_research_tests_passed")


if __name__ == "__main__":
    run_all()
