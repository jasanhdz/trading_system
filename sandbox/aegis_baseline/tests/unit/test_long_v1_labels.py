import math
from datetime import datetime, timedelta, timezone

from aegis.domain import Candle
from aegis.training.labels import (
    LONG_LABEL_SCHEMA_VERSION,
    LongLabelConfig,
    build_long_path_label,
)


def _candle(open_time, open_price, high, low, close):
    return Candle(
        open_time,
        open_time + timedelta(minutes=5),
        open_price,
        high,
        low,
        close,
        100.0,
        True,
        "LONG_LABEL_FIXTURE",
    )


def _path(*, ambiguous=False, gap=False):
    start = datetime(2026, 7, 1, tzinfo=timezone.utc)
    signal = _candle(start, 100.0, 100.1, 99.9, 100.0)
    future = []
    for index in range(12):
        timestamp = signal.close_time + timedelta(minutes=5 * index)
        if gap and index >= 5:
            timestamp += timedelta(minutes=5)
        high = 100.5 if index == 0 else 100.4
        low = 99.6 if ambiguous and index == 0 else 100.0 if index == 0 else 99.9
        future.append(_candle(timestamp, 100.0, high, low, 100.4))
    return signal, tuple(future)


def test_long_v1_golden_path_uses_upside_mfe_and_downside_mae() -> None:
    signal, future = _path()
    label = build_long_path_label(signal, future)
    assert label.schema_version == LONG_LABEL_SCHEMA_VERSION
    assert label.valid and label.entry_convention == "SIGNAL_CLOSE"
    assert math.isclose(label.mfe_fraction, 0.005, abs_tol=1e-15)
    assert math.isclose(label.mae_fraction, 0.001, abs_tol=1e-15)
    assert math.isclose(label.terminal_long_return, 0.004, abs_tol=1e-15)
    assert label.hit_before_stop and not label.stopped_before_hit
    assert label.clean_entry and not label.bad_entry


def test_long_same_bar_target_and_stop_is_conservative() -> None:
    signal, future = _path(ambiguous=True)
    label = build_long_path_label(signal, future)
    assert label.valid and label.ambiguous_hit_stop
    assert not label.hit_before_stop and label.stopped_before_hit
    assert not label.clean_entry and label.bad_entry and label.tail_event


def test_long_gap_and_incomplete_horizon_are_quarantined() -> None:
    signal, future = _path(gap=True)
    assert build_long_path_label(signal, future).quarantine_reason == "FUTURE_GAP"
    assert (
        build_long_path_label(signal, future[:-1]).quarantine_reason
        == "INCOMPLETE_HORIZON"
    )


def test_long_next_bar_open_does_not_change_default_entry_rule() -> None:
    signal, future = _path()
    first = future[0]
    shifted = (
        _candle(first.open_time, 99.0, 100.5, 98.9, 100.4),
        *future[1:],
    )
    next_open = build_long_path_label(
        signal,
        shifted,
        LongLabelConfig(entry_rule="NEXT_BAR_OPEN"),
    )
    default = build_long_path_label(signal, shifted)
    assert next_open.entry_price == 99.0
    assert default.entry_price == 100.0
