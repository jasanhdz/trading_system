from datetime import datetime, timedelta, timezone

import pytest

from aegis.config import CANONICAL_SYMBOLS, CANONICAL_SYMBOL_SET_HASH
from aegis.domain import Candle, DecisionRequest, FeedQuality, MarketSnapshot, PortfolioContext, SymbolSeries


@pytest.fixture
def snapshot_factory():
    def build(*, bars: int = 60, closed_at: datetime | None = None, available_slots: int = 1) -> MarketSnapshot:
        end = closed_at or datetime(2026, 7, 17, 12, 0, tzinfo=timezone.utc)
        series = []
        for symbol_index, symbol in enumerate(CANONICAL_SYMBOLS):
            candles = []
            base = 10.0 + symbol_index * 3.0
            for index in range(bars):
                close_time = end - timedelta(minutes=5 * (bars - 1 - index))
                open_time = close_time - timedelta(minutes=5)
                drift = (symbol_index - 5) * 0.00008
                open_price = base * (1.0 + drift * index)
                close = open_price * (1.0 + drift + ((index % 5) - 2) * 0.00003)
                high = max(open_price, close) * 1.001
                low = min(open_price, close) * 0.999
                candles.append(Candle(open_time, close_time, open_price, high, low, close,
                                      1000.0 + symbol_index * 10 + index, True, "OFFLINE_FIXTURE", str(index)))
            series.append(SymbolSeries(symbol, tuple(candles), end, FeedQuality()))
        return MarketSnapshot(end, "5m", CANONICAL_SYMBOL_SET_HASH, tuple(reversed(series)),
                              PortfolioContext(available_slots=available_slots, operational_time=end))
    return build


@pytest.fixture
def decision_request(snapshot_factory):
    return DecisionRequest("request-1", "cycle-1", "aegis-decision-request-v1",
                           "aegis-clean-rebuild-v1", "aegis-scientific-config-v1", snapshot_factory())
