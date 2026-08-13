import pandas as pd

from aegis.research.event_path_quality_c2a import (
    PathContract,
    _side_path_outcomes,
    build_path_dataset,
    collapse_registered_events,
    detect_registered_events,
)
from aegis.research.market_event_fast_track_m1a import FlowBucket, MinuteBar


def _frame(prices):
    return pd.DataFrame({
        "open": prices, "high": [value * 1.01 for value in prices],
        "low": [value * 0.999 for value in prices], "close": prices,
    })


def test_long_and_short_path_outcomes_are_side_aware_and_adverse_first_on_tie():
    contract = PathContract(2, 0.005, 0.005, 0.001)
    long = _side_path_outcomes(_frame([100.0, 100.0, 101.0, 101.0]), "LONG", contract)
    short = _side_path_outcomes(_frame([100.0, 100.0, 101.0, 101.0]), "SHORT", contract)
    assert long.iloc[0].barrier_outcome == "FAVORABLE_FIRST"
    assert short.iloc[0].barrier_outcome == "ADVERSE_FIRST_OR_SAME"
    tied = pd.DataFrame({
        "open": [100.0, 100.0, 100.0], "high": [100.0, 101.0, 100.0],
        "low": [100.0, 99.0, 100.0], "close": [100.0, 100.0, 100.0],
    })
    result = _side_path_outcomes(tied, "LONG", PathContract(1, .005, .005, .001))
    assert result.iloc[0].barrier_outcome == "ADVERSE_FIRST_OR_SAME"


def test_dataset_uses_next_minute_entry_and_requires_causal_history(monkeypatch):
    monkeypatch.setattr("aegis.research.event_path_quality_c2a.ZSCORE_LOOKBACK", 3)
    monkeypatch.setattr("aegis.research.event_path_quality_c2a.FLOW_WINDOWS", (1, 3))
    bars, flow = [], []
    for index in range(10):
        timestamp = 1_800_000_000_000 + index * 60_000
        price = 100.0 + index
        bars.append(MinuteBar(
            "ADAUSDT", timestamp, price, price * 1.002, price * .998,
            price * 1.001, 10.0, 1000.0, 20, 600.0,
        ))
        flow.append(FlowBucket(
            "ADAUSDT", timestamp, 600.0 + index * 20.0,
            400.0 - index * 10.0, 20 + index,
        ))
    contract = PathContract(2, .001, .001, .0005)
    result = build_path_dataset(tuple(bars), tuple(flow), (contract,))
    assert set(result.side) == {"LONG", "SHORT"}
    assert (result.h2_f010_a010_entry_price > result.open).all()
    assert result.schema_version.nunique() == 1
    assert result.history_contiguous.all()
    assert result.future_contiguous.all()


def test_registered_detectors_are_side_aware_and_cooldown_is_deterministic():
    rows = pd.DataFrame([
        {"event_timestamp_ms": 1_000_000, "symbol": "ADAUSDT", "side": "LONG",
         "side_flow_z": 3.0, "trade_count_z": 2.0, "side_flow_imbalance_3m": .2,
         "side_flow_persistence_5m": .4, "side_price_response_1m": .001},
        {"event_timestamp_ms": 1_060_000, "symbol": "ADAUSDT", "side": "LONG",
         "side_flow_z": 3.0, "trade_count_z": 2.0, "side_flow_imbalance_3m": .2,
         "side_flow_persistence_5m": .4, "side_price_response_1m": .001},
        {"event_timestamp_ms": 1_000_000, "symbol": "ADAUSDT", "side": "SHORT",
         "side_flow_z": -3.0, "trade_count_z": 2.0, "side_flow_imbalance_3m": 0.0,
         "side_flow_persistence_5m": -.4, "side_price_response_1m": .001},
    ])
    config = {"event_detectors": {
        "FLOW_IMPULSE_CONTINUATION": {
            "side_flow_z_minimum": 2.5, "side_trade_count_z_minimum": 1.0,
            "side_flow_imbalance_3m_minimum": .1,
            "side_flow_persistence_5m_minimum": .2,
            "side_price_response_1m_minimum": 0.0,
        },
        "FLOW_ABSORPTION_REVERSAL": {
            "opposing_flow_z_minimum": 2.5, "opposing_trade_count_z_minimum": 1.0,
            "side_price_response_1m_minimum": .0001,
            "side_flow_imbalance_3m_minimum": -.1,
        },
    }}
    detected = detect_registered_events(rows, config)
    assert detected.event_family.value_counts().to_dict() == {
        "FLOW_IMPULSE_CONTINUATION": 2, "FLOW_ABSORPTION_REVERSAL": 1,
    }
    collapsed = collapse_registered_events(detected, 15)
    assert len(collapsed) == 2
