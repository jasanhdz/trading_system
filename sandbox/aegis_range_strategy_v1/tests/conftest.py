from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from aegis_range_v1.candidates import RangeCandidate
from aegis_range_v1.models import Candle1m, Candle5m


@pytest.fixture
def candidate() -> RangeCandidate:
    return RangeCandidate(0.2, 0.0125, 1.0, 0.35, 0.0, 25.0, 0.62, 0.5)


@pytest.fixture
def origin() -> datetime:
    return datetime(2030, 1, 1, tzinfo=timezone.utc)


def make_5m(index: int, origin: datetime, *, symbol: str = "BTCUSDT", open_: float = 100.0, high: float = 101.0, low: float = 99.0, close: float = 100.0, segment_id: int = 0) -> Candle5m:
    open_time = origin + timedelta(minutes=5 * index)
    return Candle5m(symbol, open_time, open_time + timedelta(minutes=5), open_, high, low, close, 10.0, segment_id)


def make_1m(index: int, origin: datetime, *, symbol: str = "BTCUSDT", price: float = 100.0) -> Candle1m:
    open_time = origin + timedelta(minutes=index)
    return Candle1m(symbol, open_time, price, price + 1.0, price - 1.0, price + 0.25, 2.0)


class FakeRegimeEvaluator:
    def __init__(self):
        self.calls = []

    def evaluate(self, **kwargs):
        self.calls.append(kwargs)
        return {
            "technicalRegime": "ACCUMULATION_RANGE",
            "transition": {"risk": "LOW"},
            "indicators": {
                "adx": 18.0,
                "atrPercentile": 0.4,
                "bollingerWidthPercentile": 0.2,
                "volumeRatio": 1.0,
                "rangeBreakout": "NONE",
                "failedBreakoutCount": 0,
                "structure": "MIXED",
            },
            "scores": {"chopRisk": 0.7},
            "marketConfirmation": {"state": "UNKNOWN"},
            "confidence": 0.999,
            "momentumEnvironment": "NONE",
        }
