from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from typing import Iterable

from .models import Candle1m, Candle5m, DataIntegrityEvent

ONE_MINUTE = timedelta(minutes=1)
FIVE_MINUTES = timedelta(minutes=5)


class DataValidationError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class AggregationResult:
    candles: tuple[Candle5m, ...]
    integrity_events: tuple[DataIntegrityEvent, ...]


class RangeDataAdapter:
    @staticmethod
    def candle_1m_from_source(
        symbol: str,
        open_time: datetime,
        open_value: str,
        high_value: str,
        low_value: str,
        close_value: str,
        volume_value: str,
    ) -> Candle1m:
        raw = (open_value, high_value, low_value, close_value, volume_value)
        parsed = tuple(Decimal(value) for value in raw)
        if not all(value.is_finite() for value in parsed):
            raise DataValidationError("source OHLCV decimals must be finite")
        open_decimal, high_decimal, low_decimal, close_decimal, volume_decimal = parsed
        if min(open_decimal, high_decimal, low_decimal, close_decimal) <= 0 or volume_decimal < 0:
            raise DataValidationError("source prices must be positive and volume nonnegative")
        if high_decimal < max(open_decimal, close_decimal, low_decimal) or low_decimal > min(open_decimal, close_decimal, high_decimal):
            raise DataValidationError("source OHLC is inconsistent")
        return Candle1m(
            symbol,
            open_time,
            *(float(value) for value in parsed),
            open_value,
            high_value,
            low_value,
            close_value,
        )

    @staticmethod
    def _validate_candle(candle: Candle1m) -> None:
        timestamp = candle.open_time
        if timestamp.tzinfo is None or timestamp.utcoffset() is None:
            raise DataValidationError("timestamp must be timezone-aware")
        utc = timestamp.astimezone(timezone.utc)
        if utc.second != 0 or utc.microsecond != 0:
            raise DataValidationError("1m timestamp must align to a UTC minute")
        values = (candle.open, candle.high, candle.low, candle.close, candle.volume)
        if not all(math.isfinite(value) for value in values):
            raise DataValidationError("OHLCV values must be finite")
        if min(candle.open, candle.high, candle.low, candle.close) <= 0 or candle.volume < 0:
            raise DataValidationError("OHLC prices must be positive and volume nonnegative")
        if candle.high < max(candle.open, candle.close, candle.low):
            raise DataValidationError("high is inconsistent with OHLC")
        if candle.low > min(candle.open, candle.close, candle.high):
            raise DataValidationError("low is inconsistent with OHLC")
        source_values = (candle.open_source, candle.high_source, candle.low_source, candle.close_source)
        numeric_values = (candle.open, candle.high, candle.low, candle.close)
        for source, numeric in zip(source_values, numeric_values, strict=True):
            if source is not None:
                decimal = Decimal(source)
                if not decimal.is_finite() or float(decimal) != numeric:
                    raise DataValidationError("source decimal does not match binary64 value")

    @staticmethod
    def validate_5m(candle: Candle5m) -> None:
        timestamp = candle.open_time
        if timestamp.tzinfo is None or timestamp.utcoffset() is None:
            raise DataValidationError("5m timestamp must be timezone-aware")
        utc = timestamp.astimezone(timezone.utc)
        if utc.second != 0 or utc.microsecond != 0 or utc.minute % 5 != 0:
            raise DataValidationError("5m timestamp must align to UTC 5m")
        if candle.available_at != timestamp + FIVE_MINUTES:
            raise DataValidationError("available_at must equal open_time + 5m")
        values = (candle.open, candle.high, candle.low, candle.close, candle.volume)
        if not all(math.isfinite(value) for value in values):
            raise DataValidationError("5m OHLCV values must be finite")
        if min(candle.open, candle.high, candle.low, candle.close) <= 0 or candle.volume < 0:
            raise DataValidationError("5m OHLC prices must be positive and volume nonnegative")
        if candle.high < max(candle.open, candle.close, candle.low) or candle.low > min(candle.open, candle.close, candle.high):
            raise DataValidationError("5m OHLC is inconsistent")
        for source, numeric in ((candle.high_source, candle.high), (candle.low_source, candle.low)):
            if source is not None:
                decimal = Decimal(source)
                if not decimal.is_finite() or float(decimal) != numeric:
                    raise DataValidationError("5m source decimal does not match binary64 value")

    @classmethod
    def aggregate_1m_to_5m(cls, candles: Iterable[Candle1m]) -> AggregationResult:
        source = list(candles)
        if not source:
            return AggregationResult((), ())
        symbol = source[0].symbol
        if any(candle.symbol != symbol for candle in source):
            raise DataValidationError("one aggregation call accepts one symbol")
        if any(candle.open_time.tzinfo is None or candle.open_time.utcoffset() is None for candle in source):
            raise DataValidationError("timestamp must be timezone-aware")

        def block_for(timestamp: datetime) -> datetime:
            utc = timestamp.astimezone(timezone.utc)
            return utc.replace(minute=(utc.minute // 5) * 5, second=0, microsecond=0)

        invalid_blocks: set[datetime] = set()
        validated: list[Candle1m] = []
        for candle in source:
            try:
                cls._validate_candle(candle)
            except DataValidationError:
                invalid_blocks.add(block_for(candle.open_time))
            else:
                validated.append(candle)
        for previous, current in zip(source, source[1:]):
            if current.open_time <= previous.open_time:
                invalid_blocks.add(block_for(previous.open_time))
                invalid_blocks.add(block_for(current.open_time))

        counts: dict[datetime, int] = {}
        for candle in validated:
            utc = candle.open_time.astimezone(timezone.utc)
            counts[utc] = counts.get(utc, 0) + 1
        for timestamp, count in counts.items():
            if count > 1:
                invalid_blocks.add(block_for(timestamp))

        indexed = {
            candle.open_time.astimezone(timezone.utc): candle
            for candle in validated
            if counts[candle.open_time.astimezone(timezone.utc)] == 1
        }
        timestamps = [candle.open_time.astimezone(timezone.utc) for candle in source]
        first = min(timestamps)
        last = max(timestamps)
        block = first.replace(minute=(first.minute // 5) * 5)
        final_block = last.replace(minute=(last.minute // 5) * 5)
        output: list[Candle5m] = []
        events: list[DataIntegrityEvent] = []
        segment_id = 0
        previous_valid_block: datetime | None = None

        while block <= final_block:
            expected = [block + offset * ONE_MINUTE for offset in range(5)]
            members = [indexed.get(timestamp) for timestamp in expected]
            if block in invalid_blocks or any(member is None for member in members):
                events.append(DataIntegrityEvent(symbol, block))
                segment_id += 1
                previous_valid_block = None
            else:
                complete = [member for member in members if member is not None]
                high_candle = max(complete, key=lambda candle: candle.high)
                low_candle = min(complete, key=lambda candle: candle.low)
                if previous_valid_block is not None and block != previous_valid_block + FIVE_MINUTES:
                    segment_id += 1
                output.append(
                    Candle5m(
                        symbol=symbol,
                        open_time=block,
                        available_at=block + FIVE_MINUTES,
                        open=complete[0].open,
                        high=max(candle.high for candle in complete),
                        low=min(candle.low for candle in complete),
                        close=complete[-1].close,
                        volume=sum(candle.volume for candle in complete),
                        segment_id=segment_id,
                        high_source=high_candle.high_source or repr(high_candle.high),
                        low_source=low_candle.low_source or repr(low_candle.low),
                    )
                )
                previous_valid_block = block
            block += FIVE_MINUTES
        return AggregationResult(tuple(output), tuple(events))

    @staticmethod
    def contiguous_window(candles: list[Candle5m], size: int = 160) -> tuple[Candle5m, ...]:
        if len(candles) < size:
            raise ValueError("INSUFFICIENT_HISTORY")
        window = candles[-size:]
        for candle in window:
            RangeDataAdapter.validate_5m(candle)
        segment_id = window[0].segment_id
        for previous, current in zip(window, window[1:]):
            if current.segment_id != segment_id or current.open_time != previous.open_time + FIVE_MINUTES:
                raise ValueError("INSUFFICIENT_HISTORY")
        return tuple(window)
