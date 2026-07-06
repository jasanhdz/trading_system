#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

import sys

sys.path.append(str(Path(__file__).resolve().parents[2]))

from aegis_alpha.turbo.short_quality_v4_labels import (  # noqa: E402
    ShortV4Config,
    build_operable_short_quality_v4_labels,
    classify_short_bad_entry_v4,
    classify_short_clean_entry_v4,
    classify_short_management_dependent_v4,
    classify_short_no_trade_v4,
    classify_short_premium_allowed_v4,
    compute_short_path_metrics_v4,
    summarize_short_v4_labels,
)


def test_short_mfe_uses_future_low_and_mae_uses_future_high() -> None:
    close = np.array([100.0, 99.8, 99.5, 99.7])
    high = np.array([100.0, 100.2, 100.1, 100.0])
    low = np.array([100.0, 99.4, 99.0, 99.2])
    m = compute_short_path_metrics_v4(high=high, low=low, close=close, entry_index=0, horizon=3)
    assert round(m["mfe_price_move"], 4) == 0.0100
    assert round(m["mae_price_move"], 4) == 0.0020


def test_same_candle_target_stop_counts_stop_first() -> None:
    close = np.array([100.0, 100.0, 99.0])
    high = np.array([100.0, 104.0, 100.0])
    low = np.array([100.0, 94.0, 99.0])
    m = compute_short_path_metrics_v4(high=high, low=low, close=close, entry_index=0, horizon=2)
    assert m["hit5_before_minus3"] is False
    assert m["hit5_before_minus3_stopped"] is True
    assert m["ambiguous_hit_stop"] is True


def test_clean_entry_requires_mfe_before_mae() -> None:
    cfg = ShortV4Config(horizon=6)
    close = np.array([100.0, 99.8, 99.5, 99.4, 99.3, 99.2, 99.1])
    high = np.array([100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0])
    low = np.array([100.0, 99.7, 99.3, 99.1, 99.1, 99.1, 99.1])
    m = compute_short_path_metrics_v4(high=high, low=low, close=close, entry_index=0, horizon=6, config=cfg)
    assert classify_short_clean_entry_v4(m, cfg) == 1


def test_clean_entry_fails_if_mae_early() -> None:
    cfg = ShortV4Config(horizon=6)
    close = np.array([100.0, 100.0, 99.8, 99.4, 99.1, 99.0, 98.9])
    high = np.array([100.0, 100.5, 100.1, 100.1, 100.1, 100.1, 100.1])
    low = np.array([100.0, 99.9, 99.7, 99.4, 99.1, 98.9, 98.8])
    m = compute_short_path_metrics_v4(high=high, low=low, close=close, entry_index=0, horizon=6, config=cfg)
    assert classify_short_clean_entry_v4(m, cfg) == 0
    assert classify_short_bad_entry_v4(m, cfg) == 1


def test_premium_requires_clean_and_low_danger() -> None:
    cfg = ShortV4Config(horizon=6)
    close = np.array([100.0, 99.8, 99.4, 99.0, 98.8, 98.7, 98.6])
    high = np.array([100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0])
    low = np.array([100.0, 99.6, 99.2, 98.5, 98.5, 98.5, 98.5])
    m = compute_short_path_metrics_v4(high=high, low=low, close=close, entry_index=0, horizon=6, config=cfg)
    assert classify_short_clean_entry_v4(m, cfg) == 1
    assert classify_short_premium_allowed_v4("ADAUSDT", m, cfg) == 1
    assert classify_short_premium_allowed_v4("ETHUSDT", m, cfg) == 0


def test_management_dependent_and_no_trade() -> None:
    cfg = ShortV4Config(horizon=6)
    metrics = {
        "horizon": 6,
        "mfe_roe_proxy": 0.03,
        "mae_roe_proxy": 0.05,
        "mfe_mae_ratio": 0.60,
        "time_to_mfe": 5,
        "net_quality_after_costs": -0.02,
        "initial_adverse_3_candles": 0.05,
        "mae_before_mfe": True,
        "mfe_before_mae": False,
    }
    assert classify_short_management_dependent_v4(metrics, cfg) == 1
    assert classify_short_no_trade_v4("ADAUSDT", metrics, cfg) == 1


def test_net_quality_after_costs_discounts_fees_slippage() -> None:
    cfg = ShortV4Config(horizon=3, fee_bps=4.0, slippage_bps=1.0)
    close = np.array([100.0, 99.5, 99.5, 99.5])
    high = np.array([100.0, 100.0, 100.0, 100.0])
    low = np.array([100.0, 99.5, 99.5, 99.5])
    m = compute_short_path_metrics_v4(high=high, low=low, close=close, entry_index=0, horizon=3, config=cfg)
    assert m["round_trip_cost_roe"] > 0
    assert m["net_quality_after_costs"] < m["mfe_roe_proxy"] - m["mae_roe_proxy"]


def test_build_labels_and_summary_json_serializes() -> None:
    close = np.linspace(100.0, 95.0, 40)
    high = close + 0.05
    low = close - 0.50
    rows = build_operable_short_quality_v4_labels(symbol="ADAUSDT", high=high, low=low, close=close, horizon=6)
    summary = summarize_short_v4_labels(rows)
    assert rows
    assert "clean_rate" in summary
    json.dumps({"rows": rows[:2], "summary": summary}, default=str)


if __name__ == "__main__":
    test_short_mfe_uses_future_low_and_mae_uses_future_high()
    test_same_candle_target_stop_counts_stop_first()
    test_clean_entry_requires_mfe_before_mae()
    test_clean_entry_fails_if_mae_early()
    test_premium_requires_clean_and_low_danger()
    test_management_dependent_and_no_trade()
    test_net_quality_after_costs_discounts_fees_slippage()
    test_build_labels_and_summary_json_serializes()
    print("short_quality_v4_labels tests passed")
