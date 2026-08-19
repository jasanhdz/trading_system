from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from pathlib import Path

from aegis_e4.contracts import assert_feature_allowlist, feature_schema, stable_hash
from aegis_e4.features import assert_causal_availability, build_neutral_symbol_panel, orient_sides


def candles(rows: int = 30_000) -> pd.DataFrame:
    index = np.arange(rows)
    close = 100.0 + index * 0.001 + np.sin(index / 20.0)
    return pd.DataFrame({
        "open_time_ms": 1_600_000_000_000 + index * 60_000,
        "open": close - 0.01,
        "high": close + 0.05,
        "low": close - 0.05,
        "close": close,
        "volume": 100.0 + index % 7,
        "taker_buy_volume": 48.0 + index % 5,
    })


def test_feature_allowlist_rejects_future_targets() -> None:
    assert_feature_allowlist(["feature__base__return"])
    with pytest.raises(ValueError, match="FUTURE_LEAKAGE"):
        assert_feature_allowlist(["feature__future_mfe"])


def test_schema_and_hash_are_deterministic() -> None:
    left = feature_schema({"feature__b": "BASE", "feature__a": "FLOW"})
    right = feature_schema({"feature__a": "FLOW", "feature__b": "BASE"})
    assert left == right
    assert stable_hash(left) == stable_hash(right)


def test_closed_timeframes_never_exceed_decision_time_and_are_deterministic() -> None:
    source = candles()
    start = pd.to_datetime(source.open_time_ms.iloc[-20], unit="ms", utc=True).ceil("5min")
    anchors = pd.date_range(start, periods=2, freq="5min")
    left, families = build_neutral_symbol_panel(source, anchors, [5, 15, 60, 240])
    right, _ = build_neutral_symbol_panel(source, anchors, [5, 15, 60, 240])
    pd.testing.assert_frame_equal(left, right)
    left["symbol"] = "BTCUSDT"
    sided, _ = orient_sides(left, families)
    assert_causal_availability(sided)
    for column in [name for name in sided if name.startswith("available_at__")]:
        assert (pd.to_datetime(sided[column], utc=True) <= sided.decision_at).all()


def test_future_append_does_not_modify_historical_features() -> None:
    source = candles()
    anchor = pd.to_datetime(source.open_time_ms.iloc[-100], unit="ms", utc=True).floor("5min")
    base, _ = build_neutral_symbol_panel(source, pd.DatetimeIndex([anchor]), [5, 15, 60, 240])
    future = candles(100)
    future["open_time_ms"] += int(source.open_time_ms.iloc[-1] + 60_000 - future.open_time_ms.iloc[0])
    extended = pd.concat([source, future], ignore_index=True)
    replay, _ = build_neutral_symbol_panel(extended, pd.DatetimeIndex([anchor]), [5, 15, 60, 240])
    pd.testing.assert_frame_equal(base, replay)


def test_missing_required_base_fails_closed() -> None:
    frame = pd.DataFrame({
        "decision_at": [pd.Timestamp("2023-01-01T00:05:00Z")],
        "available_at__tf5m": [pd.Timestamp("2023-01-01T00:05:00Z")],
        "feature__base__x": [np.nan],
    })
    with pytest.raises(ValueError, match="REQUIRED_BASE_FEATURE_MISSING"):
        assert_causal_availability(frame)


def test_no_financial_or_production_capability_imported() -> None:
    import aegis_e4.dataset as dataset
    import aegis_e4.features as features

    text = Path(dataset.__file__).read_text() + Path(features.__file__).read_text()
    forbidden = ("createOrder", "cancelOrder", "api_secret", "position mutation", "pm2")
    assert not any(token in text for token in forbidden)
