from __future__ import annotations

import math

import pandas as pd
import yaml

from aegis.research.momentum_exhaustion_w2 import (
    _future_targets,
    build_episode_tables,
    next_complete_five_minute_open,
    select_nonoverlapping_candidates,
    simulate_exit_at_bar,
    simulate_policy,
    stable_episode_id,
)


def _config():
    return yaml.safe_load(
        open("config/experiments/aegis_momentum_exhaustion_w2.yaml")
    )


def _features(rows: int = 100) -> pd.DataFrame:
    start = int(pd.Timestamp("2024-02-01T00:00:00Z").timestamp() * 1_000)
    result = []
    for index in range(rows):
        open_price = 100.0 + 0.1 * index
        result.append({
            "symbol": "ADAUSDT",
            "open_time_ms": start + index * 300_000,
            "close_time_ms": start + (index + 1) * 300_000 - 1,
            "open": open_price,
            "high": open_price + 0.3,
            "low": open_price - 0.1,
            "close": open_price + 0.2,
            "atr": 1.0,
            "atr_fraction": 0.01,
            "body_ratio": 0.5,
            "clv": 0.75,
            "quote_volume": 1000.0 + index,
            "volume_ratio_20": 1.5,
            "volume_z_20": 1.0,
            "taker_imbalance": 0.2,
            "delta_velocity": 1.0,
            "delta_acceleration": 0.1,
            "return_1": 0.001,
            "velocity_atr_1": 0.1,
            "acceleration_atr": 0.01,
            "rsi_6": 55.0,
            "price_vs_ma_7_atr": 0.2,
            "price_vs_ma_25_atr": 0.3,
            "ma_7_slope_atr": 0.1,
            "ma_25_slope_atr": 0.05,
            "higher_high": True,
            "higher_low": True,
            "lower_high": False,
            "lower_low": False,
            "context_15m_return_1": 0.002,
            "context_15m_ma_25_slope_atr": 0.1,
            "btc_5m_return_1": 0.001,
            "btc_15m_return_1": 0.002,
            "btc_5m_taker_imbalance": 0.1,
        })
    return pd.DataFrame(result)


def test_episode_identity_is_stable_and_side_specific():
    first = stable_episode_id("ADAUSDT", "LONG", 100)
    assert first == stable_episode_id("ADAUSDT", "LONG", 100)
    assert first != stable_episode_id("ADAUSDT", "SHORT", 100)


def test_candidate_selection_enforces_non_overlap_and_purge():
    config = _config()
    start = int(pd.Timestamp("2024-02-01T00:00:00Z").timestamp() * 1_000)
    candidates = pd.DataFrame({
        "symbol": ["ADAUSDT"] * 3,
        "side": ["LONG"] * 3,
        "timestamp_ms": [start, start + 60_000, start + 480 * 60_000],
    })
    selected = select_nonoverlapping_candidates(candidates, config)
    assert selected["timestamp_ms"].tolist() == [start, start + 480 * 60_000]


def test_next_entry_is_after_signal_and_five_minute_aligned():
    signal = int(pd.Timestamp("2024-02-01T00:03:59Z").timestamp() * 1_000)
    entry = next_complete_five_minute_open(signal)
    assert entry > signal
    assert entry % 300_000 == 0


def test_episode_builder_separates_simulated_and_actual_fields():
    config = _config()
    features = _features()
    timestamp = int(pd.Timestamp("2024-02-01T00:00:59Z").timestamp() * 1_000)
    candidates = pd.DataFrame({
        "symbol": ["ADAUSDT"], "side": ["LONG"],
        "timestamp_ms": [timestamp], "partition": ["TRAIN"],
    })
    episodes, decisions = build_episode_tables(
        "ADAUSDT", features, candidates, config
    )
    assert len(episodes) == 1
    assert episodes.iloc[0].outcome_source == "SIMULATED"
    assert math.isnan(episodes.iloc[0].actual_entry)
    assert episodes.iloc[0].simulated_entry == 100.1
    assert not decisions.empty
    assert decisions["position_episode_id"].nunique() == 1
    assert decisions["gate_025"].all()
    assert "target_giveback_before_new_extreme" in decisions


def test_fixed_tp_policy_is_side_symmetric_and_cost_aware():
    base = {
        "simulated_entry": 100.0,
        "entry_atr": 1.0,
        "path_atr": [1.0, 1.0, 1.0],
    }
    long = simulate_policy({
        **base, "side": "LONG", "path_high": [100.2, 100.6, 100.7],
        "path_low": [99.9, 100.0, 100.1], "path_close": [100.1, 100.5, 100.6],
    }, policy="FIXED_TP", parameter=0.5, gate_atr=0.25, cost_bps=14)
    short = simulate_policy({
        **base, "side": "SHORT", "path_high": [100.1, 100.0, 99.9],
        "path_low": [99.8, 99.4, 99.3], "path_close": [99.9, 99.5, 99.4],
    }, policy="FIXED_TP", parameter=0.5, gate_atr=0.25, cost_bps=14)
    assert long.gross_return == short.gross_return == 0.005
    assert long.net_return == short.net_return == 0.0036


def test_common_hard_stop_has_priority_over_same_bar_take_profit():
    result = simulate_policy({
        "side": "LONG", "simulated_entry": 100.0, "entry_atr": 1.0,
        "path_high": [101.0], "path_low": [97.0], "path_close": [100.0],
        "path_atr": [1.0],
    }, policy="FIXED_TP", parameter=0.5, gate_atr=0.25, cost_bps=0)
    assert result.exit_reason == "COMMON_HARD_STOP"
    assert result.gross_return == -0.02


def test_future_giveback_measures_additional_loss_not_past_giveback():
    targets = _future_targets(
        favorable_high=pd.Series([1.0, 1.0, 1.0, 1.0]).to_numpy(),
        favorable_low=pd.Series([0.9, 0.45, 0.44, 0.43]).to_numpy(),
        closes=pd.Series([0.9, 0.5, 0.49, 0.48]).to_numpy(),
        index=1,
        peak_at_index=1.0,
        atr=1.0,
    )
    assert targets["target_giveback_025_atr_next_3"] is False
    assert targets["target_giveback_before_new_extreme"] is False


def test_profit_capture_uses_common_full_episode_opportunity():
    result = simulate_exit_at_bar({
        "side": "LONG", "simulated_entry": 100.0,
        "path_high": [100.5, 101.0], "path_low": [100.0, 100.4],
        "path_close": [100.5, 100.8],
    }, requested_exit_bar=1, cost_bps=0.0, reason="W2")
    assert result.peak_mfe == 0.01
    assert result.profit_capture_ratio == 0.5
