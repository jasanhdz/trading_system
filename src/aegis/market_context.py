"""Strict parser for causal market snapshots transported by the TypeScript bot."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Mapping

from aegis.config import CANONICAL_SYMBOLS, CANONICAL_SYMBOL_SET_HASH
from aegis.domain import Candle, FeedQuality, MarketSnapshot, PortfolioContext, SymbolSeries

MARKET_CONTEXT_VERSION = "AEGIS_MARKET_CONTEXT_V1"
MARKET_CONTEXT_SOURCE = "SHARED_MARKET_DATA_RUNTIME"
TIMEFRAME = "5m"
INTERVAL_MS = 300_000
MIN_HISTORY_BARS = 96


class MarketContextError(ValueError):
    pass


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise MarketContextError(f"{name} must be a mapping")
    return value


def _integer(value: Any, name: str) -> int:
    if isinstance(value, bool):
        raise MarketContextError(f"{name} must be an integer")
    try:
        result = int(value)
    except (TypeError, ValueError) as exc:
        raise MarketContextError(f"{name} must be an integer") from exc
    if result < 0:
        raise MarketContextError(f"{name} cannot be negative")
    return result


def _number(value: Any, name: str, *, positive: bool = False) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise MarketContextError(f"{name} must be numeric") from exc
    if result != result or result in (float("inf"), float("-inf")):
        raise MarketContextError(f"{name} must be finite")
    if positive and result <= 0:
        raise MarketContextError(f"{name} must be positive")
    return result


def _utc_ms(value: int) -> datetime:
    return datetime.fromtimestamp(value / 1000, tz=timezone.utc)


def market_snapshot_from_context(
    payload: Mapping[str, Any],
    *,
    expected_symbol: str | None = None,
) -> MarketSnapshot:
    context = _mapping(payload, "market_context")
    if context.get("version") != MARKET_CONTEXT_VERSION:
        raise MarketContextError("AEGIS_MARKET_CONTEXT_VERSION_INVALID")
    if context.get("source") != MARKET_CONTEXT_SOURCE or context.get("status") != "FRESH":
        raise MarketContextError("AEGIS_MARKET_CONTEXT_SOURCE_INVALID")

    symbol = str(context.get("symbol", "")).strip().upper()
    if symbol not in CANONICAL_SYMBOLS:
        raise MarketContextError("AEGIS_MARKET_CONTEXT_SYMBOL_UNAUTHORIZED")
    if expected_symbol and symbol != expected_symbol.strip().upper():
        raise MarketContextError("AEGIS_MARKET_CONTEXT_SYMBOL_MISMATCH")

    captured_at_ms = _integer(context.get("capturedAtMs"), "capturedAtMs")
    universe = _mapping(context.get("universe5m"), "universe5m")
    if universe.get("timeframe") != TIMEFRAME:
        raise MarketContextError("AEGIS_MARKET_CONTEXT_TIMEFRAME_INVALID")
    if universe.get("symbolSetHash") != CANONICAL_SYMBOL_SET_HASH:
        raise MarketContextError("AEGIS_MARKET_CONTEXT_UNIVERSE_HASH_MISMATCH")

    closed_at_ms = _integer(universe.get("closedAtMs"), "universe5m.closedAtMs")
    if closed_at_ms > captured_at_ms:
        raise MarketContextError("AEGIS_MARKET_CONTEXT_FUTURE_CLOSE")
    if captured_at_ms - closed_at_ms > 15 * 60 * 1000:
        raise MarketContextError("AEGIS_MARKET_CONTEXT_SNAPSHOT_STALE")

    series_payload = universe.get("series")
    if not isinstance(series_payload, list) or len(series_payload) != len(CANONICAL_SYMBOLS):
        raise MarketContextError("AEGIS_MARKET_CONTEXT_UNIVERSE_SIZE_MISMATCH")

    by_symbol: dict[str, Mapping[str, Any]] = {}
    for raw in series_payload:
        item = _mapping(raw, "universe5m.series[]")
        item_symbol = str(item.get("symbol", "")).strip().upper()
        if item_symbol in by_symbol:
            raise MarketContextError("AEGIS_MARKET_CONTEXT_DUPLICATE_SYMBOL")
        by_symbol[item_symbol] = item
    if tuple(by_symbol.keys()) != CANONICAL_SYMBOLS:
        raise MarketContextError("AEGIS_MARKET_CONTEXT_SYMBOL_ORDER_MISMATCH")

    closed_at = _utc_ms(closed_at_ms)
    series: list[SymbolSeries] = []
    for item_symbol in CANONICAL_SYMBOLS:
        item = by_symbol[item_symbol]
        if item.get("status") != "FRESH" or item.get("source") != "WEBSOCKET":
            raise MarketContextError("AEGIS_MARKET_CONTEXT_CANDLE_SOURCE_INVALID")
        observed_at_ms = _integer(item.get("observedAtMs"), f"{item_symbol}.observedAtMs")
        websocket_at_ms = _integer(
            item.get("websocketObservedAtMs"), f"{item_symbol}.websocketObservedAtMs"
        )
        if observed_at_ms != websocket_at_ms:
            raise MarketContextError("AEGIS_MARKET_CONTEXT_CANDLE_OBSERVATION_MISMATCH")
        source_lag_ms = max(0, captured_at_ms - observed_at_ms)
        if source_lag_ms > 30_000:
            raise MarketContextError("AEGIS_MARKET_CONTEXT_CANDLE_STALE")

        rows = item.get("candles")
        if not isinstance(rows, list) or len(rows) < MIN_HISTORY_BARS:
            raise MarketContextError("AEGIS_MARKET_CONTEXT_INCOMPLETE_SERIES")
        candles: list[Candle] = []
        previous_open_ms: int | None = None
        for raw_candle in rows[-MIN_HISTORY_BARS:]:
            row = _mapping(raw_candle, f"{item_symbol}.candle")
            open_ms = _integer(row.get("openTime"), f"{item_symbol}.openTime")
            normalized_close_ms = open_ms + INTERVAL_MS
            if normalized_close_ms > closed_at_ms:
                raise MarketContextError("AEGIS_MARKET_CONTEXT_OPEN_CANDLE_PRESENT")
            if previous_open_ms is not None and open_ms - previous_open_ms != INTERVAL_MS:
                raise MarketContextError("AEGIS_MARKET_CONTEXT_CANDLE_GAP")
            previous_open_ms = open_ms
            candles.append(
                Candle(
                    _utc_ms(open_ms),
                    _utc_ms(normalized_close_ms),
                    _number(row.get("open"), f"{item_symbol}.open", positive=True),
                    _number(row.get("high"), f"{item_symbol}.high", positive=True),
                    _number(row.get("low"), f"{item_symbol}.low", positive=True),
                    _number(row.get("close"), f"{item_symbol}.close", positive=True),
                    _number(row.get("volume"), f"{item_symbol}.volume"),
                    True,
                    "TYPESCRIPT_SHARED_WEBSOCKET_KLINES",
                    str(open_ms),
                )
            )
        if candles[-1].close_time != closed_at:
            raise MarketContextError("AEGIS_MARKET_CONTEXT_UNALIGNED_SERIES")
        series.append(
            SymbolSeries(
                item_symbol,
                tuple(candles),
                closed_at,
                FeedQuality(source_lag_ms=source_lag_ms),
            )
        )

    return MarketSnapshot(
        closed_at,
        TIMEFRAME,
        CANONICAL_SYMBOL_SET_HASH,
        tuple(series),
        PortfolioContext(available_slots=1, operational_time=closed_at),
    )
