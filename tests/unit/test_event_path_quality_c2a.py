import pandas as pd
import zipfile

from aegis.research.event_path_quality_c2a import (
    PathContract,
    _side_path_outcomes,
    build_path_dataset,
    collapse_registered_events,
    day_cluster_bootstrap,
    detect_registered_events,
    deterministic_matched_control,
    economic_summary,
    read_agg_trade_archives_chunked,
)
from aegis.research.market_event_fast_track_m1a import (
    FlowBucket, MinuteBar, read_agg_trade_archive,
)


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


def test_economic_metrics_and_control_are_deterministic():
    population = pd.DataFrame({
        "event_timestamp_ms": [1_000_000 + index * 60_000 for index in range(20)],
        "symbol": ["ADAUSDT"] * 20, "side": ["LONG"] * 20,
        "utility": [.01 if index % 2 else -.005 for index in range(20)],
    })
    selected = population.iloc[[1, 5, 9]].copy()
    first = deterministic_matched_control(population, selected)
    second = deterministic_matched_control(population, selected)
    assert first.event_timestamp_ms.tolist() == second.event_timestamp_ms.tolist()
    assert len(first) == len(selected)
    summary = economic_summary(selected, "utility")
    assert summary["events"] == 3
    assert summary["net_expectancy"] > 0
    interval = day_cluster_bootstrap(selected, "utility", repetitions=10)
    assert interval["expectancy_lower_95"] == interval["expectancy_upper_95"]


def test_chunked_archive_reader_is_equivalent_to_canonical_reader(tmp_path):
    path = tmp_path / "ADAUSDT-aggTrades-2026-07.zip"
    payload = (
        "agg_trade_id,price,quantity,first_trade_id,last_trade_id,transact_time,is_buyer_maker\n"
        "1,2,3,1,1,1800000000000,true\n"
        "2,2,4,2,2,1800000001000,false\n"
        "3,4,5,3,3,1800000060000,false\n"
    )
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("ADAUSDT-aggTrades-2026-07.csv", payload)
    canonical = read_agg_trade_archive(path, "ADAUSDT")
    chunked = read_agg_trade_archives_chunked((path,), "ADAUSDT", chunk_size=2)
    assert chunked == canonical
