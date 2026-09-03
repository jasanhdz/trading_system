from datetime import timedelta

import pandas as pd

from aegis_strategy_router.domain.types import DataStatus, Side, Timeframe
from aegis_strategy_router.replay.snapshot_builder import DeterministicSnapshotBuilder


def _state(snapshot, timeframe: Timeframe):
    return next(item for item in snapshot.timeframes if item.timeframe is timeframe)


def test_future_rows_do_not_change_past_snapshot(one_minute: pd.DataFrame) -> None:
    builder = DeterministicSnapshotBuilder()
    decision = pd.to_datetime(one_minute.iloc[719]["open_time_ms"], unit="ms", utc=True) + pd.Timedelta(minutes=1)
    base = builder.build(
        symbol="BTCUSDT", decision_at=decision.to_pydatetime(), reference_price=100.0,
        one_minute=one_minute.iloc[:720], proposed_side=Side.SHORT, signal_id="s-1",
    )
    future = one_minute.iloc[720:].copy()
    future.loc[:, "close"] = 10_000.0
    future.loc[:, "high"] = 10_001.0
    injected = pd.concat([one_minute.iloc[:720], future], ignore_index=True)
    repeated = builder.build(
        symbol="BTCUSDT", decision_at=decision.to_pydatetime(), reference_price=100.0,
        one_minute=injected, proposed_side=Side.SHORT, signal_id="s-1",
    )
    assert base.canonical_bytes() == repeated.canonical_bytes()
    assert base.snapshot_id == repeated.snapshot_id


def test_open_bar_is_not_available(one_minute: pd.DataFrame) -> None:
    builder = DeterministicSnapshotBuilder()
    first = pd.to_datetime(one_minute.iloc[0]["open_time_ms"], unit="ms", utc=True)
    decision = first + pd.Timedelta(minutes=501, seconds=30)
    snapshot = builder.build(
        symbol="BTCUSDT", decision_at=decision.to_pydatetime(), reference_price=100.0,
        one_minute=one_minute, built_at=decision.to_pydatetime(),
    )
    state = _state(snapshot, Timeframe.M5)
    assert state.latest_closed_at <= decision.to_pydatetime()
    assert state.latest_closed_at == (first + pd.Timedelta(minutes=500)).to_pydatetime()


def test_missing_higher_timeframes_remain_unknown(one_minute: pd.DataFrame) -> None:
    builder = DeterministicSnapshotBuilder()
    decision = pd.to_datetime(one_minute.iloc[-1]["open_time_ms"], unit="ms", utc=True) + pd.Timedelta(minutes=1)
    snapshot = builder.build(
        symbol="BTCUSDT", decision_at=decision.to_pydatetime(), reference_price=100.0,
        one_minute=one_minute.iloc[:120],
    )
    for timeframe in (Timeframe.H4, Timeframe.D1):
        state = _state(snapshot, timeframe)
        assert state.status is DataStatus.UNKNOWN
        assert state.reason in {"NO_FULLY_CLOSED_BAR", "FEATURE_WARMUP_INCOMPLETE"}


def test_malformed_source_marks_every_timeframe_invalid() -> None:
    builder = DeterministicSnapshotBuilder()
    malformed = pd.DataFrame({"open_time_ms": [1_700_000_000_000], "close": [100.0]})
    decision = pd.Timestamp("2026-08-17T12:00:00Z")
    snapshot = builder.build(
        symbol="BTCUSDT", decision_at=decision.to_pydatetime(), reference_price=100.0,
        one_minute=malformed,
    )
    assert all(state.status is DataStatus.INVALID for state in snapshot.timeframes)


def test_future_outcome_column_injection_fails_closed(one_minute: pd.DataFrame) -> None:
    builder = DeterministicSnapshotBuilder()
    injected = one_minute.copy()
    injected["future_mfe_bps"] = 999.0
    decision = pd.to_datetime(injected.iloc[-1]["open_time_ms"], unit="ms", utc=True) + pd.Timedelta(minutes=1)
    snapshot = builder.build(
        symbol="BTCUSDT", decision_at=decision.to_pydatetime(), reference_price=100.0,
        one_minute=injected,
    )
    assert all(state.status is DataStatus.INVALID for state in snapshot.timeframes)
    assert all("forbidden outcome/future" in state.reason for state in snapshot.timeframes)
