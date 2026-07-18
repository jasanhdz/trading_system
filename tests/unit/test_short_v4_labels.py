import math
from datetime import datetime, timedelta, timezone

from aegis.domain import Candle
from aegis.training.labels import SHORT_LABEL_SCHEMA_VERSION, ShortLabelConfig, build_short_path_label


def _candle(open_time, open_price, high, low, close):
    return Candle(open_time, open_time + timedelta(minutes=5), open_price, high, low, close, 100.0, True, "LABEL_FIXTURE")


def _path(*, ambiguous=False, gap=False):
    start = datetime(2026, 7, 1, tzinfo=timezone.utc)
    signal = _candle(start, 100.0, 100.1, 99.9, 100.0)
    future = []
    for index in range(12):
        timestamp = signal.close_time + timedelta(minutes=5 * index)
        if gap and index >= 5:
            timestamp += timedelta(minutes=5)
        high = 100.4 if ambiguous and index == 0 else 100.1
        low = 99.5 if index == 0 else 99.6
        future.append(_candle(timestamp, 100.0, high, low, 99.6))
    return signal, tuple(future)


def test_short_v4_golden_path_uses_price_fraction_and_signal_close() -> None:
    signal, future = _path()
    label = build_short_path_label(signal, future)
    assert label.schema_version == SHORT_LABEL_SCHEMA_VERSION
    assert label.valid and label.entry_convention == "SIGNAL_CLOSE"
    assert math.isclose(label.mfe_fraction, 0.005, abs_tol=1e-15)
    assert math.isclose(label.mae_fraction, 0.001, abs_tol=1e-15)
    assert math.isclose(label.round_trip_cost_fraction, 0.001, abs_tol=1e-15)
    assert math.isclose(label.net_quality_after_costs, 0.003, abs_tol=1e-15)
    assert label.time_to_mfe == 1 and label.time_to_mae == 1
    assert not label.mfe_before_mae
    assert not label.clean_entry  # equal-time extrema are conservative, never favorable-first
    assert not label.bad_entry and not label.tail_event
    assert label.hit_before_stop and not label.stopped_before_hit


def test_same_bar_hit_and_stop_is_ambiguous_and_conservative() -> None:
    signal, future = _path(ambiguous=True)
    label = build_short_path_label(signal, future)
    assert label.valid and label.ambiguous_hit_stop
    assert not label.hit_before_stop and label.stopped_before_hit
    assert not label.clean_entry and label.bad_entry and label.tail_event


def test_gap_and_incomplete_horizon_are_quarantined() -> None:
    signal, future = _path(gap=True)
    assert build_short_path_label(signal, future).quarantine_reason == "FUTURE_GAP"
    assert build_short_path_label(signal, future[:-1]).quarantine_reason == "INCOMPLETE_HORIZON"


def test_label_uses_only_t_plus_one_through_t_plus_horizon() -> None:
    signal, future = _path()
    baseline = build_short_path_label(signal, future)
    outside = _candle(future[-1].close_time, 100.0, 200.0, 1.0, 150.0)
    assert build_short_path_label(signal, (*future, outside)[:12]) == baseline


def test_thresholds_are_historical_roe_values_converted_to_price_fraction() -> None:
    config = ShortLabelConfig()
    assert math.isclose(config.clean_mfe_fraction * config.historical_reference_leverage, 0.08)
    assert math.isclose(config.clean_mae_fraction * config.historical_reference_leverage, 0.055)
    assert math.isclose(config.bad_mae_fraction * config.historical_reference_leverage, 0.06)


def test_e2_label_uses_next_bar_open_without_changing_legacy_default() -> None:
    signal, future = _path()
    first = future[0]
    shifted = (_candle(first.open_time, 101.0, 101.1, 99.5, 100.0), *future[1:])
    e2 = build_short_path_label(signal, shifted, ShortLabelConfig(entry_rule="NEXT_BAR_OPEN"))
    legacy = build_short_path_label(signal, shifted)
    assert e2.entry_convention == "NEXT_BAR_OPEN" and e2.entry_price == 101.0
    assert legacy.entry_convention == "SIGNAL_CLOSE" and legacy.entry_price == 100.0
