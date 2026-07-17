from dataclasses import replace
from datetime import timedelta

import pytest

from aegis.config import load_brain_config
from aegis.domain import Candle, DomainValidationError, FeedQuality, ValidationStatus
from aegis.features import DeterministicFeaturePipeline, FrozenNormalizer, MarketSnapshotValidator, SnapshotValidationError


def _validator():
    from pathlib import Path
    return MarketSnapshotValidator(load_brain_config(Path(__file__).parents[2] / "config").universe)


def test_validation_rejects_wrong_universe_future_gap_and_short_history(snapshot_factory) -> None:
    snapshot = snapshot_factory()
    validator = _validator()
    with pytest.raises(SnapshotValidationError) as mismatch:
        validator.validate(replace(snapshot, symbol_set_hash="0" * 64), snapshot.closed_at)
    assert mismatch.value.status is ValidationStatus.NO_TRADE_UNIVERSE_MISMATCH

    future = replace(snapshot.series[0].candles[-1], close_time=snapshot.closed_at + timedelta(minutes=5))
    future_series = replace(snapshot.series[0], candles=(*snapshot.series[0].candles[:-1], future),
                            last_confirmed_close=future.close_time)
    with pytest.raises(SnapshotValidationError):
        validator.validate(replace(snapshot, series=(future_series, *snapshot.series[1:])), snapshot.closed_at)

    gapped = replace(snapshot.series[0], candles=(snapshot.series[0].candles[0], *snapshot.series[0].candles[2:]))
    with pytest.raises(SnapshotValidationError):
        validator.validate(replace(snapshot, series=(gapped, *snapshot.series[1:])), snapshot.closed_at)

    short = replace(snapshot.series[0], candles=snapshot.series[0].candles[-20:])
    with pytest.raises(SnapshotValidationError) as insufficient:
        validator.validate(replace(snapshot, series=(short, *snapshot.series[1:])), snapshot.closed_at)
    assert insufficient.value.status is ValidationStatus.NO_TRADE_DATA_INSUFFICIENT


def test_validation_rejects_invalid_ohlc_nan_and_reported_duplicates(snapshot_factory) -> None:
    snapshot = snapshot_factory()
    validator = _validator()
    bad = replace(snapshot.series[0].candles[-1], high=snapshot.series[0].candles[-1].low)
    bad_series = replace(snapshot.series[0], candles=(*snapshot.series[0].candles[:-1], bad))
    with pytest.raises(SnapshotValidationError):
        validator.validate(replace(snapshot, series=(bad_series, *snapshot.series[1:])), snapshot.closed_at)
    with pytest.raises(DomainValidationError):
        replace(snapshot.series[0].candles[-1], close=float("nan"))
    duplicate = replace(snapshot.series[0], feed_quality=FeedQuality(duplicate_bars=1))
    with pytest.raises(SnapshotValidationError):
        validator.validate(replace(snapshot, series=(duplicate, *snapshot.series[1:])), snapshot.closed_at)


def test_constant_prices_zero_volume_and_extremes_remain_finite(snapshot_factory) -> None:
    snapshot = snapshot_factory()
    series = []
    for symbol_index, item in enumerate(snapshot.series):
        price = float(symbol_index + 1)
        candles = tuple(replace(candle, open=price, high=price, low=price, close=price, volume=0.0) for candle in item.candles)
        series.append(replace(item, candles=candles))
    batch = DeterministicFeaturePipeline().transform(replace(snapshot, series=tuple(series)))
    assert all(value == value and abs(value) < float("inf") for row in batch.rows for value in row.raw_values)
    assert all(row.quality.missing_values == 0 for row in batch.rows)


def test_degenerate_normalizer_and_noncanonical_batch_fail_closed(snapshot_factory) -> None:
    with pytest.raises(ValueError):
        FrozenNormalizer(means={"ret_1": 0.0}, scales={"ret_1": 0.0}).normalize("ret_1", 1.0)
    snapshot = snapshot_factory()
    with pytest.raises(ValueError):
        DeterministicFeaturePipeline().transform(replace(snapshot, series=snapshot.series[:-1]))
