from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import yaml

from aegis.domain import Candle
from aegis.features import DeterministicFeaturePipeline
from aegis.research.hybrid_ts_protection_replay import TsProtectionConfig
from aegis.research.long_entry_v21_shadow import (
    AtrPathContract,
    LONG_V21_FEATURE_NAMES,
    aggregate_causal_candles,
    atr_normalized_long_outcome,
    multitimeframe_long_features,
    protected_long_utility,
)


def candles(count: int, *, start: float = 100.0) -> tuple[Candle, ...]:
    origin = datetime(2026, 1, 1, tzinfo=timezone.utc)
    rows = []
    for index in range(count):
        opened = origin + timedelta(minutes=5 * index)
        price = start * (1.0 + index * 0.0001)
        rows.append(
            Candle(
                open_time=opened,
                close_time=opened + timedelta(minutes=5),
                open=price,
                high=price * 1.001,
                low=price * 0.999,
                close=price * 1.0002,
                volume=100.0 + index,
                is_closed=True,
                source="TEST",
            )
        )
    return tuple(rows)


def contract() -> AtrPathContract:
    return AtrPathContract(1.0, 0.75, 0.0025, 0.002, 0.015, 0.012, 12, 0.001)


def test_multitimeframe_features_are_built_only_from_supplied_history() -> None:
    history = candles(576)
    pipeline = DeterministicFeaturePipeline()
    base = pipeline._local_features(history[-96:])
    base.update({name: 0.0 for name in pipeline.feature_names if name not in base})
    vector, context = multitimeframe_long_features(base, history, pipeline=pipeline)
    assert len(vector) == len(LONG_V21_FEATURE_NAMES)
    assert len(aggregate_causal_candles(history, 3)) == 48
    assert len(aggregate_causal_candles(history, 12)) == 48
    assert context["15m_ret_1"] != context["1h_ret_1"]


def test_atr_barriers_scale_by_symbol_and_use_next_bar_open() -> None:
    signal = candles(1)[0]
    future = (
        Candle(
            open_time=signal.close_time,
            close_time=signal.close_time + timedelta(minutes=5),
            open=101.0,
            high=101.6,
            low=100.8,
            close=101.4,
            volume=1.0,
            is_closed=True,
            source="TEST",
        ),
    )
    low_vol = atr_normalized_long_outcome(
        signal=signal, future=future, atr_fraction=0.001, contract=contract()
    )
    high_vol = atr_normalized_long_outcome(
        signal=signal, future=future, atr_fraction=0.01, contract=contract()
    )
    assert low_vol["entry_price"] == 101.0
    assert low_vol["favorable_barrier_fraction"] == 0.0025
    assert high_vol["favorable_barrier_fraction"] == 0.01
    assert high_vol["adverse_barrier_fraction"] == 0.0075


def test_protected_utility_uses_worst_intrabar_result() -> None:
    history = candles(20)
    start = history[-1].close_time
    future = tuple(
        Candle(
            open_time=start + timedelta(minutes=5 * index),
            close_time=start + timedelta(minutes=5 * (index + 1)),
            open=102.0,
            high=103.0,
            low=101.0,
            close=102.5,
            volume=1.0,
            is_closed=True,
            source="TEST",
        )
        for index in range(3)
    )
    outcome = {
        "mae_fraction": 0.005,
        "time_underwater_bars": 1,
        "atr_fraction": 0.004,
    }
    result = protected_long_utility(
        history=history,
        future=future,
        outcome=outcome,
        protection=TsProtectionConfig(),
        mae_penalty_weight=0.5,
        underwater_bar_penalty_fraction=0.000025,
        catastrophic_mae_atr_multiple=1.5,
    )
    path_values = [
        row["net_return_after_costs"] for row in result["protection_results"].values()
    ]
    assert result["protected_worst_net_return"] == min(path_values)
    assert result["utility_target"] < result["protected_worst_net_return"]
    assert result["selection_effect"] == "NONE"
    assert result["exchange_mutations"] == 0


def test_preregistration_cannot_enable_live() -> None:
    root = Path(__file__).parents[2]
    payload = yaml.safe_load(
        (root / "config/experiments/aegis_long_entry_v21_shadow.yaml").read_text()
    )
    assert payload["mode"] == "SHADOW"
    assert payload["selection_effect"] == "NONE"
    assert payload["automatic_live_promotion"] is False
    assert payload["deployment"]["live_runtime"] == "PROHIBITED"
    assert payload["deployment"]["exchange_mutations"] == 0


def test_public_delta_refresh_is_an_exact_unauthenticated_get_surface() -> None:
    root = Path(__file__).parents[2]
    source = (root / "scripts/refresh_long_v21_public_candles.py").read_text()
    assert 'HOST = "fapi.binance.com"' in source
    assert 'PATH = "/fapi/v1/klines"' in source
    assert 'urllib.request.Request(url, method="GET")' in source
    assert "HTTPRedirectHandler" in source
    assert 'authenticated_requests": 0' in source
    assert 'exchange_mutations": 0' in source
    for prohibited in (
        "X-MBX-APIKEY",
        "BINANCE_API_KEY",
        "BINANCE_API_SECRET",
        'method="POST"',
        'method="PUT"',
        'method="PATCH"',
        'method="DELETE"',
    ):
        assert prohibited not in source
