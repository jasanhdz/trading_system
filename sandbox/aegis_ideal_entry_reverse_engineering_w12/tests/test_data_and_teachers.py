from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import pandas as pd
import pandas.testing as pdt

from aegis_ideal_entry_reverse_engineering_w12.data import (
    assign_zones,
    causal_symbol_features,
    feature_columns,
    load_config,
    teacher_labels,
)


SANDBOX = Path(__file__).resolve().parents[1]


def config() -> dict:
    value = load_config(SANDBOX / "config" / "w12_frozen.json")
    value["source"]["start_inclusive"] = "2022-04-03T00:00:00Z"
    value["source"]["end_exclusive"] = "2022-04-04T00:00:00Z"
    return value


def candles(periods: int = 800, *, trend: float = 0.01) -> pd.DataFrame:
    time = pd.date_range("2022-04-02 20:00", periods=periods, freq="1min", tz="UTC")
    close = 100.0 + np.arange(periods) * trend + np.sin(np.arange(periods) / 13.0) * 0.02
    return pd.DataFrame({
        "open_time_ms": time.astype("int64") // 1_000_000,
        "open": close - trend,
        "high": close + 0.03,
        "low": close - 0.03,
        "close": close,
        "volume": 100.0 + np.arange(periods) % 20,
        "close_time_ms": time.astype("int64") // 1_000_000 + 59_999,
        "quote_volume": (100.0 + np.arange(periods) % 20) * close,
        "trade_count": np.full(periods, 10),
        "taker_buy_volume": 50.0 + np.sin(np.arange(periods) / 7.0) * 10,
        "taker_buy_quote_volume": (50.0 + np.sin(np.arange(periods) / 7.0) * 10) * close,
    })


def test_future_mutation_cannot_change_past_features() -> None:
    source = candles()
    original = causal_symbol_features(source, "BTCUSDT", config())
    cutoff = pd.Timestamp("2022-04-03 06:00", tz="UTC")
    changed = source.copy()
    future = pd.to_datetime(changed["open_time_ms"], unit="ms", utc=True).ge(cutoff)
    changed.loc[future, ["open", "high", "low", "close", "volume", "taker_buy_volume"]] *= 4
    mutated = causal_symbol_features(changed, "BTCUSDT", config())
    columns = [column for column in feature_columns(original) if column in mutated]
    pdt.assert_frame_equal(
        original.loc[original["decision_at"].le(cutoff), columns].reset_index(drop=True),
        mutated.loc[mutated["decision_at"].le(cutoff), columns].reset_index(drop=True),
    )


def test_feature_provenance_and_forbidden_schema() -> None:
    features = causal_symbol_features(candles(), "BTCUSDT", config())
    assert features["feature_available_at"].le(features["decision_at"]).all()
    names = feature_columns(features)
    assert names and not any("future" in name or "mfe" in name for name in names)
    contaminated = features.assign(future_return=1.0)
    try:
        feature_columns(contaminated)
    except ValueError as error:
        assert "future/label" in str(error)
    else:
        raise AssertionError("future feature was accepted")


def test_teacher_long_and_short_are_symmetric_and_adverse_first() -> None:
    source = candles(trend=0.0)
    features = causal_symbol_features(source, "BTCUSDT", config()).iloc[[20]].copy()
    start = int(features.iloc[0]["source_row_index"]) + 1
    source.loc[start : start + 14, ["open", "close"]] = 100.0
    source.loc[start : start + 14, "high"] = 100.4
    source.loc[start : start + 14, "low"] = 99.7
    labels = teacher_labels(source, features, 15, config()).set_index("side")
    assert labels.loc["LONG", "barrier_outcome"] == "ADVERSE_FIRST"
    assert labels.loc["SHORT", "barrier_outcome"] == "ADVERSE_FIRST"
    assert labels.loc["LONG", "policy_gross_bps"] == -20.0
    assert labels.loc["SHORT", "policy_gross_bps"] == -20.0


def test_clean_directional_path_scores_as_ideal() -> None:
    source = candles(trend=0.0)
    features = causal_symbol_features(source, "BTCUSDT", config()).iloc[[20]].copy()
    start = int(features.iloc[0]["source_row_index"]) + 1
    path = np.linspace(100.0, 100.6, 15)
    source.loc[start : start + 14, "open"] = np.r_[100.0, path[:-1]]
    source.loc[start : start + 14, "close"] = path
    source.loc[start : start + 14, "high"] = path + 0.01
    source.loc[start : start + 14, "low"] = path - 0.01
    labels = teacher_labels(source, features, 15, config()).set_index("side")
    assert bool(labels.loc["LONG", "majority_ideal"])
    assert labels.loc["LONG", "entry_quality_score"] > labels.loc["SHORT", "entry_quality_score"]
    assert labels.loc["LONG", "policy_gross_bps"] == 30.0


def test_zone_deduplication_keeps_one_best_timestamp() -> None:
    times = pd.date_range("2022-04-03", periods=5, freq="15min", tz="UTC")
    labels = pd.DataFrame({
        "decision_at": times, "symbol": "BTCUSDT", "side": "LONG",
        "horizon_minutes": 30, "majority_ideal": [True, True, True, False, True],
        "entry_quality_score": [70.0, 90.0, 80.0, 10.0, 95.0],
    })
    result = assign_zones(labels, config())
    assert result["zone_id"].dropna().nunique() == 2
    assert result["zone_best"].sum() == 2
    assert result.loc[result["zone_best"], "entry_quality_score"].tolist() == [90.0, 95.0]
