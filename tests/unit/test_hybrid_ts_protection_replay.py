from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from aegis.data import CanonicalBar
from aegis.research.hybrid_ts_protection_replay import (
    IntrabarPath,
    ProtectionExit,
    TsProtectionConfig,
    replay_ts_price_protection,
    wilder_atr,
)
from aegis.training.hybrid_directional import DirectionalSide


def _bars(prices: list[tuple[float, float, float, float]]) -> tuple[CanonicalBar, ...]:
    start = datetime(2026, 1, 1, tzinfo=timezone.utc)
    return tuple(
        CanonicalBar(
            start + timedelta(minutes=5 * index),
            open_price,
            high,
            low,
            close,
            1.0,
        )
        for index, (open_price, high, low, close) in enumerate(prices)
    )


def _history() -> tuple[CanonicalBar, ...]:
    return _bars([(100.0, 100.2, 99.8, 100.0)] * 15)


def test_wilder_atr_matches_constant_true_range() -> None:
    assert wilder_atr(_history(), 14) == pytest.approx(0.4)


def test_long_hard_stop_and_short_take_profit_are_replayed() -> None:
    long_result = replay_ts_price_protection(
        side=DirectionalSide.LONG,
        history=_history(),
        future=_bars([(100.0, 100.1, 97.0, 98.0)]),
        path=IntrabarPath.OPEN_HIGH_LOW_CLOSE,
        config=TsProtectionConfig(),
    )
    short_result = replay_ts_price_protection(
        side=DirectionalSide.SHORT,
        history=_history(),
        future=_bars([(100.0, 100.1, 96.0, 97.0)]),
        path=IntrabarPath.OPEN_HIGH_LOW_CLOSE,
        config=TsProtectionConfig(),
    )

    assert long_result.exit_reason is ProtectionExit.HARD_STOP
    assert long_result.gross_return_fraction == pytest.approx(-0.40 / 15.0)
    assert short_result.exit_reason is ProtectionExit.TAKE_PROFIT
    assert short_result.gross_return_fraction == pytest.approx(0.50 / 15.0)


def test_break_even_and_fixed_callback_protect_favorable_excursions() -> None:
    break_even = replay_ts_price_protection(
        side=DirectionalSide.LONG,
        history=_history(),
        future=_bars([(100.0, 100.7, 100.1, 100.2)]),
        path=IntrabarPath.OPEN_HIGH_LOW_CLOSE,
        config=TsProtectionConfig(trailing_activation_roe=9.0),
    )
    trailing = replay_ts_price_protection(
        side=DirectionalSide.LONG,
        history=_history(),
        future=_bars([(100.0, 101.2, 100.5, 100.7)]),
        path=IntrabarPath.OPEN_HIGH_LOW_CLOSE,
        config=TsProtectionConfig(use_atr_trailing=False),
    )

    assert break_even.exit_reason is ProtectionExit.BREAK_EVEN_STOP
    assert break_even.gross_return_fraction == pytest.approx(0.003)
    assert trailing.exit_reason is ProtectionExit.TRAILING_STOP
    assert trailing.gross_return_fraction > break_even.gross_return_fraction


def test_both_intrabar_paths_expose_unknown_event_order() -> None:
    future = _bars([(100.0, 100.7, 99.0, 100.2)])
    config = TsProtectionConfig(
        hard_stop_roe=-0.075,
        take_profit_roe=9.0,
        trailing_activation_roe=9.0,
    )
    favorable_first = replay_ts_price_protection(
        side=DirectionalSide.LONG,
        history=_history(),
        future=future,
        path=IntrabarPath.OPEN_HIGH_LOW_CLOSE,
        config=config,
    )
    adverse_first = replay_ts_price_protection(
        side=DirectionalSide.LONG,
        history=_history(),
        future=future,
        path=IntrabarPath.OPEN_LOW_HIGH_CLOSE,
        config=config,
    )

    assert favorable_first.exit_reason is ProtectionExit.BREAK_EVEN_STOP
    assert adverse_first.exit_reason is ProtectionExit.HARD_STOP
    assert favorable_first.net_return_after_costs > adverse_first.net_return_after_costs
