from __future__ import annotations

from .data_adapter import FIVE_MINUTES, RangeDataAdapter
from .models import Candle5m


class RangeAtr14V1:
    period = 14

    @classmethod
    def calculate(cls, candles: list[Candle5m] | tuple[Candle5m, ...]) -> float:
        if len(candles) != 160:
            raise ValueError("INSUFFICIENT_HISTORY")
        symbol = candles[0].symbol
        segment_id = candles[0].segment_id
        for candle in candles:
            RangeDataAdapter.validate_5m(candle)
            if candle.symbol != symbol or candle.segment_id != segment_id:
                raise ValueError("INSUFFICIENT_HISTORY")
        if any(current.open_time != previous.open_time + FIVE_MINUTES for previous, current in zip(candles, candles[1:])):
            raise ValueError("INSUFFICIENT_HISTORY")
        true_ranges: list[float] = []
        for index in range(1, len(candles)):
            current = candles[index]
            previous = candles[index - 1]
            true_ranges.append(
                max(
                    current.high - current.low,
                    abs(current.high - previous.close),
                    abs(current.low - previous.close),
                )
            )
        smoothed = sum(true_ranges[: cls.period]) / cls.period
        for value in true_ranges[cls.period :]:
            smoothed = (smoothed * (cls.period - 1) + value) / cls.period
        return smoothed
