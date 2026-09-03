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
FINALIZATION_DELAY_MS = 2_000
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
    universe = _mapping(context.get("universeCandles5m"), "universeCandles5m")
    if tuple(universe.keys()) != CANONICAL_SYMBOLS:
        raise MarketContextError("AEGIS_MARKET_CONTEXT_SYMBOL_ORDER_MISMATCH")

    prepared: dict[str, tuple[Mapping[str, Any], list[Mapping[str, Any]], int]] = {}
    latest_closed_ms: list[int] = []
    for item_symbol in CANONICAL_SYMBOLS:
        item = _mapping(universe[item_symbol], f"universeCandles5m.{item_symbol}")
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

        raw_rows = item.get("candles")
        if not isinstance(raw_rows, list) or len(raw_rows) < MIN_HISTORY_BARS:
            raise MarketContextError("AEGIS_MARKET_CONTEXT_INCOMPLETE_SERIES")
        rows = [_mapping(row, f"{item_symbol}.candle") for row in raw_rows]
        eligible = [
            row
            for row in rows
            if _integer(row.get("openTime"), f"{item_symbol}.openTime")
            + INTERVAL_MS
            + FINALIZATION_DELAY_MS
            <= captured_at_ms
        ]
        if len(eligible) < MIN_HISTORY_BARS:
            raise MarketContextError("AEGIS_MARKET_CONTEXT_INCOMPLETE_CLOSED_SERIES")
        symbol_closed_at_ms = (
            _integer(eligible[-1].get("openTime"), f"{item_symbol}.openTime")
            + INTERVAL_MS
        )
        latest_closed_ms.append(symbol_closed_at_ms)
        prepared[item_symbol] = (item, eligible, source_lag_ms)

    closed_at_ms = min(latest_closed_ms)
    if closed_at_ms > captured_at_ms:
        raise MarketContextError("AEGIS_MARKET_CONTEXT_FUTURE_CLOSE")
    if captured_at_ms - closed_at_ms > 15 * 60 * 1000:
        raise MarketContextError("AEGIS_MARKET_CONTEXT_SNAPSHOT_STALE")

    closed_at = _utc_ms(closed_at_ms)
    series: list[SymbolSeries] = []
    for item_symbol in CANONICAL_SYMBOLS:
        _item, eligible, source_lag_ms = prepared[item_symbol]
        aligned = [
            row
            for row in eligible
            if _integer(row.get("openTime"), f"{item_symbol}.openTime") + INTERVAL_MS
            <= closed_at_ms
        ][-MIN_HISTORY_BARS:]
        if len(aligned) < MIN_HISTORY_BARS:
            raise MarketContextError("AEGIS_MARKET_CONTEXT_INCOMPLETE_ALIGNED_SERIES")

        candles: list[Candle] = []
        previous_open_ms: int | None = None
        for row in aligned:
            open_ms = _integer(row.get("openTime"), f"{item_symbol}.openTime")
            normalized_close_ms = open_ms + INTERVAL_MS
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
