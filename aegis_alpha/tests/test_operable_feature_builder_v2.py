#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from aegis_alpha.turbo.operable_feature_builder_v2 import (
    OPERABLE_FEATURE_NAMES,
    apply_feature_set,
    build_operable_feature_matrix_v2,
)


def assert_true(value: bool, message: str) -> None:
    if not value:
        raise AssertionError(message)


def prices(count: int = 100) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    close = np.linspace(100.0, 103.0, count, dtype=np.float32)
    open_ = np.concatenate((close[:1], close[:-1]))
    high = np.maximum(open_, close) + 0.6
    low = np.minimum(open_, close) - 0.4
    volume = np.linspace(100.0, 200.0, count, dtype=np.float32)
    return high, low, close, open_, volume


def test_feature_output_is_finite_and_aligned() -> None:
    high, low, close, open_, volume = prices()
    steps = np.asarray([40, 60, 80], dtype=np.int64)
    result = build_operable_feature_matrix_v2(
        high=high, low=low, close=close, open_=open_, volume=volume, steps=steps
    )
    x = result["X_v2"]
    assert_true(x.shape == (3, len(OPERABLE_FEATURE_NAMES)), "shape must match names and samples")
    assert_true(np.all(np.isfinite(x)), "output may not contain NaN/Inf")
    assert_true(len(set(OPERABLE_FEATURE_NAMES)) == len(OPERABLE_FEATURE_NAMES), "feature names unique")
    for name in ("close_location_12", "close_location_24", "close_location_64"):
        idx = list(OPERABLE_FEATURE_NAMES).index(name)
        assert_true(np.all((x[:, idx] >= 0.0) & (x[:, idx] <= 1.0)), f"{name} must be clipped")


def test_features_do_not_read_future() -> None:
    high, low, close, open_, volume = prices()
    base = build_operable_feature_matrix_v2(
        high=high, low=low, close=close, open_=open_, volume=volume, steps=np.asarray([50])
    )["X_v2"]
    high[70:] *= 5.0
    low[70:] *= 0.2
    close[70:] *= 4.0
    changed = build_operable_feature_matrix_v2(
        high=high, low=low, close=close, open_=open_, volume=volume, steps=np.asarray([50])
    )["X_v2"]
    assert_true(np.allclose(base, changed), "future prices must not alter a prior row")


def test_zero_volume_and_flat_prices_are_safe() -> None:
    close = np.full(80, 100.0, dtype=np.float32)
    result = build_operable_feature_matrix_v2(
        high=close, low=close, close=close, volume=np.zeros(80), steps=np.asarray([30, 60])
    )
    assert_true(np.all(np.isfinite(result["X_v2"])), "flat/zero volume path must remain finite")


def test_apply_feature_set_preserves_base_and_combines_new() -> None:
    high, low, close, _, _ = prices()
    steps = np.asarray([40, 60, 80])
    market = SimpleNamespace(
        high=high,
        low=low,
        close=close,
        features=np.ones((len(close), 21), dtype=np.float32),
    )
    dataset = {
        "X": np.ones((3, 4), dtype=np.float32),
        "feature_names": np.asarray(["a", "b", "c", "d"]),
        "step": steps,
    }
    base = apply_feature_set(dataset, market, "base")
    new = apply_feature_set(dataset, market, "operable_v2")
    combined = apply_feature_set(dataset, market, "combined")
    assert_true(base["X"].shape == (3, 4), "base feature set should remain untouched")
    assert_true(new["X"].shape[1] == len(OPERABLE_FEATURE_NAMES), "operable set count")
    assert_true(combined["X"].shape[1] == 4 + len(OPERABLE_FEATURE_NAMES), "combined feature count")
    assert_true(combined["feature_set"] == "combined", "feature set metadata")


def run_all() -> None:
    test_feature_output_is_finite_and_aligned()
    test_features_do_not_read_future()
    test_zero_volume_and_flat_prices_are_safe()
    test_apply_feature_set_preserves_base_and_combines_new()
    print("manual_operable_feature_builder_v2_tests_passed")


if __name__ == "__main__":
    run_all()
