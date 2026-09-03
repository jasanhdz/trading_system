from __future__ import annotations

import math

import pytest

from aegis_range_v1.atr import RangeAtr14V1
from aegis_range_v1.data_adapter import DataValidationError, RangeDataAdapter
from aegis_range_v1.models import Candle1m
from aegis_range_v1.regime import RangeRegimeAdapter

from conftest import FakeRegimeEvaluator, make_1m, make_5m


def reference_atr(candles):
    values = [
        max(current.high - current.low, abs(current.high - previous.close), abs(current.low - previous.close))
        for previous, current in zip(candles, candles[1:])
    ]
    result = sum(values[:14]) / 14
    for value in values[14:]:
        result = (result * 13 + value) / 14
    return result


def test_atr_wilder_parity_uses_raw_binary64(origin):
    candles = [make_5m(i, origin, high=101 + i * 0.03, low=99 - i * 0.01, close=100 + i * 0.02) for i in range(160)]
    actual = RangeAtr14V1.calculate(candles)
    assert actual == reference_atr(candles)
    assert actual != round(actual, 6)


def test_strict_1m_aggregation_and_available_at(origin):
    source = [make_1m(i, origin, price=100 + i) for i in range(10)]
    result = RangeDataAdapter.aggregate_1m_to_5m(source)
    assert len(result.candles) == 2
    assert result.candles[0].open == 100
    assert result.candles[0].high == 105
    assert result.candles[0].low == 99
    assert result.candles[0].close == 104.25
    assert result.candles[0].volume == 10
    assert result.candles[0].available_at == origin.replace(minute=5)


def test_source_decimal_is_preserved_for_pivot_ids(origin):
    source = [
        RangeDataAdapter.candle_1m_from_source(
            "BTCUSDT",
            origin.replace(minute=index),
            "100.00000000",
            "101.01000000",
            "99.00000000",
            "100.25000000",
            "2.00000000",
        )
        for index in range(5)
    ]
    candle = RangeDataAdapter.aggregate_1m_to_5m(source).candles[0]
    assert candle.high_source == "101.01000000"
    assert candle.low_source == "99.00000000"


def test_gap_breaks_segment_without_fill_or_interpolation(origin):
    source = [make_1m(i, origin) for i in range(15) if i != 7]
    result = RangeDataAdapter.aggregate_1m_to_5m(source)
    assert [c.open_time.minute for c in result.candles] == [0, 10]
    assert result.candles[0].segment_id != result.candles[1].segment_id
    assert len(result.integrity_events) == 1
    assert result.integrity_events[0].block_open_time == origin.replace(minute=5)
    assert [c.segment_id for c in result.candles] == [0, 1]


@pytest.mark.parametrize(
    "mutation",
    [
        lambda candle: Candle1m(candle.symbol, candle.open_time, math.nan, candle.high, candle.low, candle.close, candle.volume),
        lambda candle: Candle1m(candle.symbol, candle.open_time, candle.open, candle.open - 1, candle.low, candle.close, candle.volume),
    ],
)
def test_invalid_ohlcv_rejected(origin, mutation):
    candle = mutation(make_1m(0, origin))
    result = RangeDataAdapter.aggregate_1m_to_5m([candle])
    assert result.candles == ()
    assert len(result.integrity_events) == 1


def test_duplicate_and_out_of_order_rejected(origin):
    candle = make_1m(0, origin)
    assert len(RangeDataAdapter.aggregate_1m_to_5m([candle, candle]).integrity_events) == 1
    assert len(RangeDataAdapter.aggregate_1m_to_5m([make_1m(1, origin), candle]).integrity_events) == 1


def test_regime_requires_exact_contiguous_160_and_market_absent(origin):
    evaluator = FakeRegimeEvaluator()
    adapter = RangeRegimeAdapter(evaluator)
    history = [make_5m(i, origin) for i in range(159)]
    with pytest.raises(ValueError, match="INSUFFICIENT_HISTORY"):
        adapter.snapshot("BTCUSDT", history)
    history.append(make_5m(159, origin))
    snapshot = adapter.snapshot("BTCUSDT", history)
    assert snapshot.atr14_raw == RangeAtr14V1.calculate(history)
    assert len(evaluator.calls[0]["candles"]) == 160
    assert "market" not in evaluator.calls[0]
    broken = history.copy()
    broken[-1] = make_5m(159, origin, segment_id=1)
    with pytest.raises(ValueError, match="INSUFFICIENT_HISTORY"):
        adapter.snapshot("BTCUSDT", broken)


def test_public_atr_boundary_rejects_non_160(origin):
    with pytest.raises(ValueError, match="INSUFFICIENT_HISTORY"):
        RangeAtr14V1.calculate([make_5m(i, origin) for i in range(159)])
    with pytest.raises(ValueError, match="INSUFFICIENT_HISTORY"):
        RangeAtr14V1.calculate([make_5m(i, origin) for i in range(161)])
