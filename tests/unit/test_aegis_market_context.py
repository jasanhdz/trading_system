from __future__ import annotations

from aegis.config import CANONICAL_SYMBOLS, CANONICAL_SYMBOL_SET_HASH
from aegis.market_context import MarketContextError, market_snapshot_from_context


def _payload() -> dict:
    interval = 300_000
    closed_at_ms = 1_800_000_000_000
    captured_at_ms = closed_at_ms + 5_000
    first_open_ms = closed_at_ms - 96 * interval
    universe: dict[str, dict] = {}
    for symbol_index, symbol in enumerate(CANONICAL_SYMBOLS):
        base = 100.0 + symbol_index
        candles = []
        for index in range(96):
            open_ms = first_open_ms + index * interval
            candles.append(
                {
                    "openTime": open_ms,
                    "timestamp": open_ms,
                    "open": base + index * 0.01,
                    "high": base + index * 0.01 + 1,
                    "low": base + index * 0.01 - 1,
                    "close": base + index * 0.01 + 0.2,
                    "volume": 1000 + index,
                    "buyVolume": 500 + index,
                    "closeTime": open_ms + interval - 1,
                }
            )
        universe[symbol] = {
            "source": "WEBSOCKET",
            "status": "FRESH",
            "observedAtMs": captured_at_ms - 100,
            "ageMs": 100,
            "websocketObservedAtMs": captured_at_ms - 100,
            "restFallbackCount": 0,
            "candles": candles,
        }
    return {
        "version": "AEGIS_MARKET_CONTEXT_V1",
        "symbol": "ETHUSDT",
        "capturedAtMs": captured_at_ms,
        "source": "SHARED_MARKET_DATA_RUNTIME",
        "status": "FRESH",
        "quote": {},
        "orderBook": {},
        "aggTrades": {},
        "candles5m": universe["ETHUSDT"],
        "universeCandles5m": universe,
        "liquidity": {},
    }


def test_market_context_builds_canonical_aligned_snapshot() -> None:
    snapshot = market_snapshot_from_context(_payload(), expected_symbol="ETHUSDT")
    assert snapshot.symbol_set_hash == CANONICAL_SYMBOL_SET_HASH
    assert tuple(item.symbol for item in snapshot.series) == CANONICAL_SYMBOLS
    assert all(len(item.candles) == 96 for item in snapshot.series)
    assert all(item.candles[-1].close_time == snapshot.closed_at for item in snapshot.series)
    assert all(item.candles[-1].source == "TYPESCRIPT_SHARED_WEBSOCKET_KLINES" for item in snapshot.series)


def test_market_context_rejects_missing_canonical_symbol() -> None:
    payload = _payload()
    payload["universeCandles5m"].pop("LTCUSDT")
    try:
        market_snapshot_from_context(payload, expected_symbol="ETHUSDT")
    except MarketContextError as exc:
        assert str(exc) == "AEGIS_MARKET_CONTEXT_SYMBOL_ORDER_MISMATCH"
    else:
        raise AssertionError("expected MarketContextError")


def test_market_context_rejects_rest_as_steady_state_source() -> None:
    payload = _payload()
    payload["universeCandles5m"]["ETHUSDT"]["source"] = "REST_RECOVERY"
    try:
        market_snapshot_from_context(payload, expected_symbol="ETHUSDT")
    except MarketContextError as exc:
        assert str(exc) == "AEGIS_MARKET_CONTEXT_CANDLE_SOURCE_INVALID"
    else:
        raise AssertionError("expected MarketContextError")
