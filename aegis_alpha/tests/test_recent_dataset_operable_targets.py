#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from aegis_alpha.turbo.recent_dataset import (
    OPERABLE_TARGET_SCHEMA_VERSION,
    build_recent_dataset,
    compute_long_short_targets,
    compute_path_outcome,
    compute_trade_quality,
)


def assert_true(value: bool, message: str) -> None:
    if not value:
        raise AssertionError(message)


def assert_close(actual: float, expected: float, message: str, tol: float = 1e-6) -> None:
    if abs(actual - expected) > tol:
        raise AssertionError(f"{message}: expected={expected} actual={actual}")


def test_long_target_before_stop() -> None:
    outcome = compute_path_outcome(
        100.0,
        np.array([101.0, 108.1, 106.0]),
        np.array([99.0, 99.0, 94.0]),
        "long",
        0.08,
        0.05,
    )
    assert_true(bool(outcome["hit_before_stop"]), "LONG target before stop should win")
    assert_true(outcome["time_to_target"] == 2, "LONG target time")


def test_long_stop_before_target() -> None:
    outcome = compute_path_outcome(
        100.0,
        np.array([101.0, 108.1]),
        np.array([94.9, 99.0]),
        "long",
        0.08,
        0.05,
    )
    assert_true(not bool(outcome["hit_before_stop"]), "LONG stop first should fail")
    assert_true(outcome["time_to_stop"] == 1, "LONG stop time")
    assert_true(outcome["time_to_target"] == 2, "LONG target time is still measured after stop")


def test_short_target_before_stop() -> None:
    outcome = compute_path_outcome(
        100.0,
        np.array([101.0, 101.0, 106.0]),
        np.array([99.0, 91.9, 94.0]),
        "short",
        0.08,
        0.05,
    )
    assert_true(bool(outcome["hit_before_stop"]), "SHORT target before stop should win")
    assert_true(outcome["time_to_target"] == 2, "SHORT target time")


def test_short_stop_before_target() -> None:
    outcome = compute_path_outcome(
        100.0,
        np.array([105.1, 101.0]),
        np.array([99.0, 91.9]),
        "short",
        0.08,
        0.05,
    )
    assert_true(not bool(outcome["hit_before_stop"]), "SHORT stop first should fail")
    assert_true(outcome["time_to_stop"] == 1, "SHORT stop time")
    assert_true(outcome["time_to_target"] == 2, "SHORT target time is still measured after stop")


def test_same_candle_is_conservative_ambiguity() -> None:
    outcome = compute_path_outcome(
        100.0,
        np.array([108.1]),
        np.array([94.9]),
        "long",
        0.08,
        0.05,
    )
    assert_true(not bool(outcome["hit_before_stop"]), "ambiguous candle should count stop first")
    assert_true(bool(outcome["ambiguous_same_candle"]), "ambiguous flag should be exposed")


def test_mfe_mae_long_short() -> None:
    high = np.array([100.0, 108.0, 104.0])
    low = np.array([100.0, 98.0, 92.0])
    close = np.array([100.0, 103.0, 96.0])
    targets = compute_long_short_targets(high, low, close, 0, 2, 0.0)
    assert_close(float(targets["long_mfe"]), 0.08, "LONG MFE")
    assert_close(float(targets["long_mae"]), 0.08, "LONG MAE")
    assert_close(float(targets["short_mfe"]), 0.08, "SHORT MFE")
    assert_close(float(targets["short_mae"]), 0.08, "SHORT MAE")


def test_quality_bounded_and_finite_without_mae() -> None:
    win = compute_trade_quality(True, True, mfe=0.10, mae=0.0, fee_round_trip=0.0)
    loss = compute_trade_quality(False, False, mfe=0.0, mae=0.20, fee_round_trip=0.01)
    no_mae = compute_long_short_targets(
        np.array([100.0, 101.0]),
        np.array([100.0, 100.0]),
        np.array([100.0, 100.5]),
        0,
        1,
        0.001,
    )
    assert_true(-1.0 <= win <= 1.0, "winning quality bounded")
    assert_true(-1.0 <= loss <= 1.0, "losing quality bounded")
    assert_true(np.isfinite(float(no_mae["long_mfe_mae_ratio"])), "zero MAE ratio finite")


def _fake_market() -> SimpleNamespace:
    count = 40
    close = np.full(count, 100.0, dtype=np.float32)
    high = np.full(count, 101.0, dtype=np.float32)
    low = np.full(count, 99.0, dtype=np.float32)
    high[2] = 108.5
    steps = np.arange(1, 8, dtype=np.int64)
    timestamps = np.asarray(
        [str(np.datetime64("2026-05-01T00:00:00") + np.timedelta64(idx * 5, "m")) for idx in range(count)]
    )
    return SimpleNamespace(
        cfg=SimpleNamespace(risk=SimpleNamespace(total_fee=0.0004)),
        signal_features=np.ones((len(steps), 3), dtype=np.float32),
        high=high,
        low=low,
        close=close,
        timestamps=timestamps,
        steps=steps,
        feature_names=["f1", "f2", "f3"],
        regimes=np.full(count, "TREND", dtype="U16"),
    )


def test_build_dataset_keeps_v1_and_adds_operable_targets() -> None:
    with patch("aegis_alpha.turbo.recent_dataset.load_signal_market", return_value=_fake_market()):
        built = build_recent_dataset("ETHUSDT", 30, save=False)
    dataset = built["dataset"]
    report = built["report"]
    for existing in (
        "long_net_return_12",
        "short_net_return_12",
        "long_good_12",
        "short_good_12",
        "future_return_6",
        "future_return_12",
        "future_return_24",
        "mfe_12",
        "mae_12",
    ):
        assert_true(existing in dataset, f"missing V1 field {existing}")
    for added in (
        "long_hit8_before_minus5_12",
        "short_hit8_before_minus5_12",
        "long_mae_danger_24",
        "short_mfe_mae_ratio_12",
        "long_trade_quality_12",
        "short_time_to_minus5_24",
        "long_ambiguous_hit_stop_12",
    ):
        assert_true(added in dataset, f"missing V2 target {added}")
    assert_true(dataset["operable_targets_schema_version"] == OPERABLE_TARGET_SCHEMA_VERSION, "schema metadata")
    assert_true(report["trade_quality_formula"], "quality formula should be documented")


def run_all() -> None:
    test_long_target_before_stop()
    test_long_stop_before_target()
    test_short_target_before_stop()
    test_short_stop_before_target()
    test_same_candle_is_conservative_ambiguity()
    test_mfe_mae_long_short()
    test_quality_bounded_and_finite_without_mae()
    test_build_dataset_keeps_v1_and_adds_operable_targets()
    print("manual_recent_dataset_operable_targets_tests_passed")


if __name__ == "__main__":
    run_all()
