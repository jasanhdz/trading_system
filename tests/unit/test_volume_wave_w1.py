import numpy as np
import pandas as pd
import yaml

from aegis.research.volume_wave_w1 import (
    SCHEMA_VERSION,
    aggregate_closed_bars,
    build_causal_feature_frame,
    build_wave_events,
    collapse_event_cooldown,
    deterministic_matched_controls,
    path_outcomes,
    registered_contracts,
)


def _config():
    return yaml.safe_load(open("config/experiments/aegis_volume_wave_w1.yaml"))


def _minutes(symbol="ADAUSDT", count=1800):
    rows = []
    for index in range(count):
        price = 100.0 + index * 0.001 + np.sin(index / 20.0) * 0.05
        close = price + np.sin(index / 7.0) * 0.01
        rows.append({
            "symbol": symbol,
            "open_time_ms": 1_700_000_100_000 + index * 60_000,
            "open": price,
            "high": max(price, close) + 0.02,
            "low": min(price, close) - 0.02,
            "close": close,
            "quote_volume": 1_000.0 + index % 17 * 10.0,
            "taker_buy_quote": 520.0 + index % 11,
            "taker_sell_quote": 480.0 + index % 7,
            "agg_trade_count": 20 + index % 5,
        })
    return pd.DataFrame(rows)


def test_closed_bar_aggregation_drops_incomplete_bucket():
    values = _minutes(count=11)
    result = aggregate_closed_bars(values, 5)
    assert len(result) == 2
    assert result.iloc[0].open == values.iloc[0].open
    assert result.iloc[0].close == values.iloc[4].close
    assert result.iloc[0].close_time_ms == result.iloc[0].open_time_ms + 300_000 - 1


def test_multitimeframe_and_btc_context_use_only_completed_bars():
    asset = _minutes()
    btc = _minutes("BTCUSDT")
    result = build_causal_feature_frame(asset, btc, _config())
    usable = result.dropna(subset=["context_15m_return_1", "btc_15m_return_1"])
    assert not usable.empty
    assert usable["close_time_ms"].is_monotonic_increasing
    assert np.isfinite(usable["btc_correlation"].dropna()).all()


def _event_frame():
    rows = []
    for index in range(12):
        open_price = 100.0 + index * 0.1
        close = open_price + (0.8 if index == 3 else 0.1)
        rows.append({
            "symbol": "ADAUSDT", "open_time_ms": 1_800_000_000_000 + index * 300_000,
            "close_time_ms": 1_800_000_000_000 + (index + 1) * 300_000 - 1,
            "open": open_price, "high": close + 0.1, "low": open_price - 0.1,
            "close": close, "atr": 1.0, "body": close - open_price,
            "body_ratio": 0.8, "body_atr": abs(close - open_price), "clv": 0.9,
            "volume_ratio_20": 2.0 if index == 3 else 1.0,
            "taker_imbalance": 0.4, "rsi_6": 55.0,
            "price_vs_ma_25_atr": 0.5, "ma_25_slope_atr": 0.1,
            "context_15m_return_1": 0.01, "context_15m_ma_25_slope_atr": 0.1,
            "btc_15m_return_1": 0.001, "btc_15m_atr_fraction": 0.01,
            "directional_persistence_3": 0.8, "directional_persistence_6": 0.7,
            "path_efficiency_3": 0.8, "path_efficiency_6": 0.7,
        })
    return pd.DataFrame(rows)


def test_events_enter_after_decision_and_keep_future_as_labels():
    result = build_wave_events(_event_frame(), _config())
    immediate = result.loc[result.entry_variant.eq("A_IMMEDIATE")].iloc[0]
    assert immediate.schema_version == SCHEMA_VERSION
    assert immediate.entry_timestamp_ms > immediate.decision_timestamp_ms
    assert immediate.entry_price == _event_frame().iloc[4].open
    assert immediate.ladder_SPACE_REMAINING
    assert immediate.future_close_1 == _event_frame().iloc[4].close


def test_cooldown_is_independent_by_entry_variant():
    first = build_wave_events(_event_frame(), _config())
    duplicate = first.copy()
    duplicate["event_timestamp_ms"] += 300_000
    combined = pd.concat([first, duplicate], ignore_index=True)
    collapsed = collapse_event_cooldown(combined, 3)
    assert collapsed.groupby(["side", "entry_variant"]).size().max() == 1


def test_matched_controls_are_deterministic_and_exclude_wave_volume():
    broad = build_wave_events(_event_frame(), _config(), minimum_volume_ratio=0.0)
    wave = broad.loc[broad.volume_ratio_20.ge(1.25)].copy()
    controls = broad.copy()
    controls["volume_ratio_20"] = 1.0
    combined = pd.concat([broad, controls], ignore_index=True)
    first = deterministic_matched_controls(
        combined, wave, minimum_volume_ratio=1.25
    )
    second = deterministic_matched_controls(
        combined, wave, minimum_volume_ratio=1.25
    )
    assert first.event_timestamp_ms.tolist() == second.event_timestamp_ms.tolist()
    assert first.volume_ratio_20.lt(1.25).all()
    assert first.sample_source.eq("MATCHED_PRICE_ONLY_CONTROL").all()


def test_triple_barrier_is_side_aware_and_adverse_first_on_same_bar():
    events = pd.DataFrame({
        "side": ["LONG", "SHORT"], "entry_price": [100.0, 100.0],
        "entry_atr": [1.0, 1.0],
        "future_high_1": [101.0, 101.0], "future_low_1": [99.0, 99.0],
        "future_close_1": [100.5, 99.5],
    })
    result = path_outcomes(
        events, horizon_bars=1, favorable_atr=0.5, adverse_atr=0.5,
        cost_bps=14.0,
    )
    assert result.adverse_before_or_same.tolist() == [True, True]
    assert result.favorable_before_adverse.tolist() == [False, False]
    assert (result.net_utility < 0.0).all()


def test_registered_contract_grid_is_unique_and_complete():
    contracts = registered_contracts(_config())
    assert len(contracts) == 30
    assert len({item[0] for item in contracts}) == 30
