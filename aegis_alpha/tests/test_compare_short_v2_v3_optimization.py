#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from aegis_alpha.tools.compare_short_v2_v3_optimization import compare_best


def assert_true(value: bool, message: str) -> None:
    if not value:
        raise AssertionError(message)


def test_promotions_and_regressions() -> None:
    v2 = {"best_by_symbol": [
        {"symbol": "BTCUSDT", "best_status": "MIXED_BEST", "feature_set": "operable_v2"},
        {"symbol": "ADAUSDT", "best_status": "STRONG_BEST", "feature_set": "operable_v2"},
    ]}
    v3 = {"best_by_symbol": [
        {
            "symbol": "BTCUSDT", "best_status": "STRONG_BEST", "feature_set": "operable_v3",
            "lookback_days": 30, "horizon_candles": 12,
        },
        {
            "symbol": "ADAUSDT", "best_status": "MIXED_BEST", "feature_set": "combined_v3",
            "lookback_days": 30, "horizon_candles": 24,
        },
    ]}
    rows = {row["symbol"]: row for row in compare_best(v2, v3)}
    assert_true(rows["BTCUSDT"]["promoted_to_strong"] is True, "promotion identified")
    assert_true(rows["BTCUSDT"]["status_change"] == "IMPROVED", "improvement identified")
    assert_true(rows["ADAUSDT"]["status_change"] == "REGRESSED", "regression identified")


def run_all() -> None:
    test_promotions_and_regressions()
    print("manual_compare_short_v2_v3_optimization_tests_passed")


if __name__ == "__main__":
    run_all()
