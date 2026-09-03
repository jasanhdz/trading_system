from __future__ import annotations

import pandas as pd
import pytest

from aegis.research.entry_safety_gate_w11 import delayed_entry_price, reconstruct_path, w11_split


def _config() -> dict:
    return {"splits": {
        "train_end_exclusive": "2026-07-01T00:00:00Z",
        "validation_end_exclusive": "2026-08-01T00:00:00Z",
    }}


def test_w11_split_keeps_august_sealed() -> None:
    assert w11_split(pd.Timestamp("2026-06-30T23:59:00Z"), _config()) == "W11_TRAIN"
    assert w11_split(pd.Timestamp("2026-07-15T00:00:00Z"), _config()) == "W11_VALIDATION"
    assert w11_split(pd.Timestamp("2026-08-01T00:00:00Z"), _config()) == "W11_FINAL_HOLDOUT"


def test_reconstruct_path_is_side_aware_and_adverse_first_on_same_bar() -> None:
    times = pd.date_range("2026-01-01T00:01:00Z", periods=4, freq="min")
    candles = pd.DataFrame({
        "open_time": times,
        "open": [100.0] * 4,
        "high": [100.4, 100.1, 100.1, 100.1],
        "low": [99.7, 99.9, 99.9, 99.9],
        "close": [100.0, 100.0, 100.0, 100.0],
    })
    result = reconstruct_path(
        candles, decision_time=pd.Timestamp("2026-01-01T00:00:20Z"), entry_price=100.0,
        side="LONG", horizon_minutes=4, favorable_barrier_bps=30.0,
        adverse_barrier_bps=20.0, cost_bps=14.0,
    )
    assert result is not None
    assert result.first_barrier_hit == "ADVERSE_FIRST"
    assert result.mfe_bps == pytest.approx(40.0)
    assert result.mae_bps == pytest.approx(30.0)


def test_reconstruct_path_does_not_use_partial_predecision_bar() -> None:
    candles = pd.DataFrame({
        "open_time": pd.date_range("2026-01-01T00:00:00Z", periods=5, freq="min"),
        "open": [100.0] * 5,
        "high": [110.0, 100.1, 100.1, 100.1, 100.1],
        "low": [90.0, 99.9, 99.9, 99.9, 99.9],
        "close": [100.0] * 5,
    })
    result = reconstruct_path(
        candles, decision_time=pd.Timestamp("2026-01-01T00:00:30Z"), entry_price=100.0,
        side="LONG", horizon_minutes=4, favorable_barrier_bps=30.0,
        adverse_barrier_bps=20.0, cost_bps=14.0,
    )
    assert result is not None
    assert result.mfe_bps < 20.0
    assert result.mae_bps < 20.0


def test_delayed_entry_uses_next_minute_open() -> None:
    candles = pd.DataFrame({
        "open_time_ms": [0, 60_000, 120_000],
        "open": [100.0, 101.0, 102.0],
    })
    assert delayed_entry_price(candles, pd.Timestamp("1970-01-01T00:00:30Z")) == 101.0
