#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from aegis_alpha.turbo.operable_feature_builder_v3 import (
    OPERABLE_FEATURE_NAMES_V3,
    apply_feature_set,
    build_operable_feature_matrix_v3,
)


def assert_true(value: bool, message: str) -> None:
    if not value:
        raise AssertionError(message)


def market(count: int = 100) -> SimpleNamespace:
    close = np.linspace(100.0, 98.0, count, dtype=np.float32)
    open_ = np.concatenate((close[:1], close[:-1]))
    high = np.maximum(open_, close) + 0.3
    low = np.minimum(open_, close) - 0.3
    timestamps = np.asarray([f"2026-05-01T00:{idx:02d}:00" for idx in range(count)])
    features = np.ones((count, 21), dtype=np.float32)
    features[:, 3] = np.linspace(1.0, 2.0, count)
    return SimpleNamespace(high=high, low=low, close=close, timestamps=timestamps, features=features)


def build(subject: SimpleNamespace, steps: np.ndarray, contexts: dict[str, object] | None = None) -> dict[str, object]:
    return build_operable_feature_matrix_v3(
        high=subject.high,
        low=subject.low,
        close=subject.close,
        timestamps=subject.timestamps,
        steps=steps,
        base_features=subject.features,
        context_markets=contexts,
    )


def feature(result: dict[str, object], name: str) -> np.ndarray:
    return np.asarray(result["X_v3"])[:, list(OPERABLE_FEATURE_NAMES_V3).index(name)]


def test_output_is_finite_aligned_unique_and_causal() -> None:
    subject = market()
    steps = np.asarray([40, 60, 80], dtype=np.int64)
    result = build(subject, steps)
    x = np.asarray(result["X_v3"])
    assert_true(x.shape == (3, len(OPERABLE_FEATURE_NAMES_V3)), "feature shape")
    assert_true(np.all(np.isfinite(x)), "V3 features finite")
    assert_true(len(set(OPERABLE_FEATURE_NAMES_V3)) == len(OPERABLE_FEATURE_NAMES_V3), "feature names unique")
    prior = build(subject, np.asarray([50]))["X_v3"]
    subject.close[70:] *= 2.0
    subject.high[70:] *= 2.0
    subject.low[70:] *= 0.5
    later_changed = build(subject, np.asarray([50]))["X_v3"]
    assert_true(np.allclose(prior, later_changed), "future data cannot change prior feature row")


def test_breakdown_and_failed_breakdown_features_react() -> None:
    subject = market()
    break_subject = market()
    break_subject.close[60] = float(np.min(break_subject.low[48:60]) - 2.0)
    break_subject.low[60] = break_subject.close[60] - 0.2
    break_subject.high[60] = break_subject.close[60] + 0.4
    normal = build(subject, np.asarray([60]))
    broken = build(break_subject, np.asarray([60]))
    assert_true(
        feature(broken, "short_breakdown_strength_12")[0] > feature(normal, "short_breakdown_strength_12")[0],
        "breakdown strength must increase on range break",
    )
    swept = market()
    prior_low = float(np.min(swept.low[48:60]))
    swept.low[60] = prior_low - 2.0
    swept.close[60] = prior_low + 0.5
    swept.high[60] = swept.close[60] + 0.3
    sweep_result = build(swept, np.asarray([60]))
    assert_true(feature(sweep_result, "short_failed_breakdown_risk_12")[0] == 1.0, "failed breakdown risk")
    assert_true(feature(sweep_result, "short_lower_wick_sweep_risk")[0] > 0.0, "lower wick sweep detected")


def test_context_absence_and_aligned_context() -> None:
    subject = market()
    steps = np.asarray([60])
    absent = build(subject, steps)
    assert_true(absent["diagnostics"]["cross_symbol_context_available"] is False, "missing context reported")
    btc = market()
    eth = market()
    aligned = build(subject, steps, {"BTCUSDT": btc, "ETHUSDT": eth})
    assert_true(aligned["diagnostics"]["cross_symbol_context_available"] is True, "aligned context used")
    assert_true(aligned["diagnostics"]["market_breadth_available"] is False, "breadth remains disabled")


def test_feature_sets_and_schema_hash() -> None:
    subject = market()
    dataset = {
        "X": np.ones((2, 4), dtype=np.float32),
        "feature_names": np.asarray(["a", "b", "c", "d"]),
        "step": np.asarray([40, 60]),
    }
    v3 = apply_feature_set(dataset, subject, "operable_v3")
    combined = apply_feature_set(dataset, subject, "combined_v3")
    assert_true(v3["X"].shape[1] == len(OPERABLE_FEATURE_NAMES_V3), "V3-only feature shape")
    assert_true(combined["X"].shape[1] > v3["X"].shape[1], "combined includes base and V2")
    assert_true(v3["feature_schema_hash"] != combined["feature_schema_hash"], "schema hash differs by set")
    assert_true(combined["operable_v3_feature_count"] == len(OPERABLE_FEATURE_NAMES_V3), "V3 count metadata")


def run_all() -> None:
    test_output_is_finite_aligned_unique_and_causal()
    test_breakdown_and_failed_breakdown_features_react()
    test_context_absence_and_aligned_context()
    test_feature_sets_and_schema_hash()
    print("manual_operable_feature_builder_v3_tests_passed")


if __name__ == "__main__":
    run_all()
