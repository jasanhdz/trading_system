from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from aegis.config import CANONICAL_SYMBOLS
from aegis.research.entry_intelligence_shadow import (
    EntryIntelligenceShadowConfig,
    EntryIntelligenceShadowError,
    EntryIntelligenceShadowRuntime,
    TimingState,
    load_entry_intelligence_shadow_config,
)
from aegis.research.directional_acceleration_shadow import (
    DirectionalAccelerationSettings,
)
from aegis.research.regime_v2 import RegimeV2Settings


def settings() -> RegimeV2Settings:
    return RegimeV2Settings(
        schema_version="aegis-regime-v2-research-settings-v1",
        history_window=8,
        minimum_history=2,
        low_volatility_quantile=0.2,
        high_volatility_quantile=0.8,
        trend_enter_fraction=0.002,
        trend_exit_fraction=0.001,
        trend_strength_enter=0.5,
        trend_strength_exit=0.35,
        chop_enter_fraction=0.7,
        chop_exit_fraction=0.6,
        high_expansion_ratio=1.0,
        low_expansion_ratio=0.0,
        minimum_state_bars=1,
    )


def config(tmp_path: Path) -> EntryIntelligenceShadowConfig:
    return EntryIntelligenceShadowConfig(
        observer_id="test-entry-intelligence",
        config_path=tmp_path / "config.yaml",
        config_sha256="a" * 64,
        signal_journal=tmp_path / "signals.jsonl",
        outcome_journal=tmp_path / "outcomes.jsonl",
        acceleration_outcome_journal=tmp_path / "acceleration-outcomes.jsonl",
        horizon_bars=2,
        round_trip_cost_fraction=0.001,
        maximum_wait_bars=3,
        require_global_bearish=True,
        require_local_bearish=True,
        require_trend_structure=True,
        require_short_ema_stack=True,
        require_bearish_confirmation_candle=True,
        maximum_counterfactual_candidates_per_cycle=1,
        regime_settings=settings(),
        acceleration_settings=DirectionalAccelerationSettings(
            minimum_pressure_components=5,
            minimum_acceleration_components=7,
            minimum_persistence=1.0 / 3.0,
            minimum_impulse_atr_multiple=1.5,
            minimum_relative_atr_multiple=0.5,
            minimum_volume_zscore=0.5,
            minimum_volume_ratio=1.1,
            minimum_trend_strength=1.25,
        ),
    )


def features(*, reversal: bool = False) -> dict[str, float]:
    values = {
        "market_direction_6": -0.01,
        "ret_6": -0.012,
        "ret_12": -0.015,
        "atr_12": 0.01,
        "range_mean_24": 0.01,
        "range_expansion": 0.2,
        "chop_12": 0.2,
        "trend_strength_12": 0.8,
        "overextended_down_risk_proxy": 0.0,
        "trend_stack_short": 1.0,
        "close_to_open_return": -0.002,
        "ret_3": -0.006,
        "ret_24": -0.02,
        "btc_divergence_6": -0.004,
        "persistence_6": -1.0,
        "ema_slope_6": -0.004,
        "ema_slope_24": -0.002,
        "trend_stack_long": 0.0,
        "distance_to_rolling_high_12": 0.02,
        "close_below_rolling_low_12": 1.0,
        "volume_zscore_24": 1.0,
        "volume_ratio_6_24": 1.2,
        "volume_spike_12": 0.0,
    }
    for name in (
        "failed_breakdown_proxy",
        "fake_breakdown_risk_proxy",
        "rebound_risk_proxy",
        "squeeze_risk_proxy_causal",
        "immediate_reversal_risk_proxy",
        "low_room_to_fall_risk_proxy",
        "high_wick_reclaim_risk_proxy",
        "squeeze_plus_reclaim_risk_proxy",
    ):
        values[name] = 0.0
    if reversal:
        values["low_room_to_fall_risk_proxy"] = 1.0
    return values


def batch(timestamp: datetime, *, selected: str | None, reversal: bool = False) -> dict:
    results = {}
    for index, symbol in enumerate(CANONICAL_SYMBOLS):
        chosen = symbol == selected
        score = 1.0 - index / 20
        results[symbol] = {
            "market_timestamp": timestamp.isoformat().replace("+00:00", "Z"),
            "feature_schema": "aegis-features-v2",
            "feature_vector_hash": f"feature-{timestamp.timestamp()}-{symbol}",
            "research_features": features(reversal=reversal and chosen),
            "market_bar": {"open": 100.0, "high": 101.0, "low": 98.0, "close": 99.0},
            "layer": {"model_disagreement": 0.0},
            "candidate": {
                "side": "SHORT",
                "eligible": True,
                "calibrated_score": score,
                "candidate_hash": f"candidate-{timestamp.timestamp()}-{symbol}",
            },
            "selected": chosen,
        }
    return {
        "decision_cycle_id": f"cycle-{timestamp.timestamp()}",
        "market_timestamp": timestamp.isoformat().replace("+00:00", "Z"),
        "results": results,
    }


def test_runtime_records_wait_then_causal_confirmation_and_restores(
    tmp_path: Path,
) -> None:
    runtime = EntryIntelligenceShadowRuntime(config(tmp_path))
    first = datetime(2026, 8, 1, tzinfo=timezone.utc)
    initial = runtime.observe_batch(batch(first, selected="BTCUSDT", reversal=True))
    assert (
        initial["BTCUSDT"]["entry_timing_shadow"]["state"]
        == TimingState.WAITING_FOR_RETEST.value
    )
    assert initial["BTCUSDT"]["uncertainty"]["value"] is None
    assert len(initial["BTCUSDT"]["candidate_ranking_shadow"]) == 11
    assert initial["BTCUSDT"]["exchange_authority"] is False
    assert initial["BTCUSDT"]["exchange_mutations"] == 0

    second = first + timedelta(minutes=5)
    confirmed = runtime.observe_batch(batch(second, selected=None))
    timing = confirmed["BTCUSDT"]["entry_timing_shadow"]
    assert timing["state"] == TimingState.TIMING_CONFIRMED.value
    assert timing["paper_action"] == "ENTER_NOW"
    assert timing["selection_effect"] == "NONE"
    assert confirmed["BTCUSDT"]["regime_v3_shadow"]["alignment"] == "ALIGNED_BEARISH"

    before = len(runtime._signals.rows)
    runtime.observe_batch(batch(second, selected=None))
    assert len(runtime._signals.rows) == before

    restored = EntryIntelligenceShadowRuntime(config(tmp_path))
    assert restored.health()["signal_records"] == 22
    assert restored.health()["pending_setups"] == 0


def test_clean_selected_candidate_is_observational_only(tmp_path: Path) -> None:
    runtime = EntryIntelligenceShadowRuntime(config(tmp_path))
    timestamp = datetime(2026, 8, 1, tzinfo=timezone.utc)
    overlay = runtime.observe_batch(batch(timestamp, selected="ETHUSDT"))
    timing = overlay["ETHUSDT"]["entry_timing_shadow"]
    assert timing["state"] == TimingState.TIMING_CONFIRMED.value
    assert overlay["ETHUSDT"]["counterfactuals"]["CONTROL_IMMEDIATE"] == "ENTER_NOW"
    assert overlay["ETHUSDT"]["selection_effect"] == "NONE"
    assert runtime.health()["exchange_mutations"] == 0


def test_acceleration_outcomes_mature_for_every_symbol(tmp_path: Path) -> None:
    runtime = EntryIntelligenceShadowRuntime(config(tmp_path))
    first = datetime(2026, 8, 1, tzinfo=timezone.utc)
    runtime.observe_batch(batch(first, selected="ADAUSDT"))
    runtime.observe_batch(batch(first + timedelta(minutes=5), selected=None))
    runtime.observe_batch(batch(first + timedelta(minutes=10), selected=None))

    health = runtime.health()
    assert health["directional_acceleration_outcomes"] == len(CANONICAL_SYMBOLS)
    assert runtime._acceleration_outcomes.rows[0]["selection_effect"] == "NONE"
    assert runtime._acceleration_outcomes.rows[0]["exchange_mutations"] == 0


def test_config_rejects_live_authority(tmp_path: Path) -> None:
    root = Path(__file__).parents[2]
    source = (root / "config/entry_intelligence_shadow.yaml").read_text(
        encoding="utf-8"
    )
    candidate = tmp_path / "entry_intelligence.yaml"
    candidate.write_text(
        source.replace("mode: SHADOW", "mode: LIVE", 1), encoding="utf-8"
    )
    with pytest.raises(EntryIntelligenceShadowError, match="AUTHORITY_INVALID"):
        load_entry_intelligence_shadow_config(
            candidate,
            repo_root=root,
            regime_config_path=root / "config/entry_quality_v2.yaml",
        )
