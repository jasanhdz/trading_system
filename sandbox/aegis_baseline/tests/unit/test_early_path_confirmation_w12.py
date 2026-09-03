from __future__ import annotations

import pandas as pd
import pytest

from aegis.research.early_path_confirmation_w12 import complete_bar_decision_time, early_path_features, w12_split


def _config() -> dict:
    return {"splits": {
        "train_end_exclusive": "2026-07-01T00:00:00Z",
        "validation_end_exclusive": "2026-07-27T00:00:00Z",
        "final_holdout_end_exclusive": "2026-08-01T00:00:00Z",
    }}


def test_w12_splits_exclude_w11_august_holdout() -> None:
    assert w12_split(pd.Timestamp("2026-06-01T00:00:00Z"), _config()) == "W12_TRAIN"
    assert w12_split(pd.Timestamp("2026-07-20T00:00:00Z"), _config()) == "W12_VALIDATION"
    assert w12_split(pd.Timestamp("2026-07-29T00:00:00Z"), _config()) == "W12_FINAL_HOLDOUT"
    assert w12_split(pd.Timestamp("2026-08-01T00:00:00Z"), _config()) == "EXCLUDED_W11_HOLDOUT"


def test_complete_bar_state_never_uses_partial_signal_bar() -> None:
    signal = pd.Timestamp("2026-01-01T00:00:20Z")
    assert complete_bar_decision_time(signal, 1) == pd.Timestamp("2026-01-01T00:02:00Z")
    assert complete_bar_decision_time(signal, 2) == pd.Timestamp("2026-01-01T00:03:00Z")


def test_early_path_features_are_direction_oriented() -> None:
    candles = pd.DataFrame({
        "open_time": pd.date_range("2026-01-01T00:01:00Z", periods=2, freq="min"),
        "open": [100.0, 100.1], "high": [100.2, 100.4], "low": [99.95, 100.0],
        "close": [100.1, 100.3], "volume": [10.0, 20.0], "taker_buy_volume": [7.0, 16.0],
    })
    result = early_path_features(
        candles, signal_time=pd.Timestamp("2026-01-01T00:00:20Z"),
        decision_time=pd.Timestamp("2026-01-01T00:03:00Z"), signal_price=100.0,
        side="LONG", move_threshold_bps=5.0,
    )
    assert result["early_directional_return_bps"] == pytest.approx(30.0)
    assert result["early_directional_taker_imbalance"] > 0.0
    assert result["early_path_efficiency"] > 0.9
