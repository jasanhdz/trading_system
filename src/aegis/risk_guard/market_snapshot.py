"""Live market snapshot provider for E4 precomputation.

Fetches 1-minute candles from Binance Futures REST API for the frozen
E4 universe (11 symbols). Produces immutable snapshots keyed by
decision_at (5-minute aligned).

NO autonomous order placement. NO trade execution.
This module is read-only market data acquisition.

Warmup requirement:
    E4 uses timeframes [5, 15, 60, 240] minutes.
    EMA99 on tf240m needs 99 × 240 = 23,760 minutes of 1m candles.
    Binance REST limit=1500 per request → ceil(23760/1500) = 16 pages.
    11 symbols × 16 pages = 176 requests per cold start.
    Rolling cache persists across cycles to minimize fetches.
"""

from __future__ import annotations

import hashlib
import logging
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

import pandas as pd
import requests

from .feature_bridge import FROZEN_E4_UNIVERSE

logger = logging.getLogger(__name__)

BINANCE_FUTURES_REST = "https://fapi.binance.com"
KLINE_ENDPOINT = "/fapi/v1/klines"
REQUEST_TIMEOUT_S = 10
INTER_REQUEST_GAP_MS = 50

# Real warmup: EMA99 on tf240m = 99 * 240 = 23,760 minutes of 1m candles
# Add margin for other indicators (rolling windows up to 96 periods on tf240m)
WARMUP_MINUTES = 24_000  # 24,000 minutes (~16.67 days)
BINANCE_LIMIT_PER_REQUEST = 1500
PAGES_PER_SYMBOL = -(-WARMUP_MINUTES // BINANCE_LIMIT_PER_REQUEST)  # ceil division = 16


@dataclass(frozen=True)
class MarketSnapshot:
    """Immutable snapshot of 1m candles for all E4 symbols at a point in time."""
    snapshot_id: str
    decision_at: datetime
    captured_at: datetime
    candles_by_symbol: dict[str, pd.DataFrame]
    snapshot_hash: str
    source_timestamps: dict[str, datetime]
    source_feed_lag_ms: dict[str, float]


class RollingCandleCache:
    """Thread-safe rolling cache for 1m candle DataFrames.

    Persists across precompute cycles to minimize REST fetches.
    Only fetches new candles since last cycle (incremental update).
    """

    def __init__(self) -> None:
        self._cache: dict[str, pd.DataFrame] = {}
        self._last_fetch_time: dict[str, datetime] = {}
        self._lock = threading.Lock()

    def get_or_fetch(
        self,
        symbol: str,
        decision_at: datetime,
        required_minutes: int = WARMUP_MINUTES,
    ) -> pd.DataFrame:
        """Get cached candles or fetch new ones incrementally.

        If cache exists and covers enough history, only fetch recent candles.
        Otherwise, do a full paginated fetch.
        """
        with self._lock:
            cached = self._cache.get(symbol)
            last_fetch = self._last_fetch_time.get(symbol)

            if cached is not None and not cached.empty and last_fetch is not None:
                # Check if cache is sufficient
                cache_minutes = len(cached)
                cache_oldest = cached["open_time"].min()
                cache_newest = cached["open_time"].max()

                # If cache covers enough history and is recent enough, fetch incrementally
                if (cache_minutes >= required_minutes and
                    cache_newest >= decision_at - pd.Timedelta(minutes=5)):
                    return self._fetch_incremental(symbol, cached, decision_at)

            # Full fetch needed
            return self._fetch_full(symbol, decision_at, required_minutes)

    def _fetch_incremental(
        self,
        symbol: str,
        cached: pd.DataFrame,
        decision_at: datetime,
    ) -> pd.DataFrame:
        """Fetch only new candles since cache ended, merge, and trim old."""
        cache_newest_ms = int(cached["open_time_ms"].max())
        decision_ms = int(decision_at.timestamp() * 1000)

        if cache_newest_ms >= decision_ms:
            # Cache already covers decision_at, just trim old data
            cutoff_ms = decision_ms - (WARMUP_MINUTES * 60 * 1000)
            trimmed = cached[cached["open_time_ms"] >= cutoff_ms].copy()
            self._cache[symbol] = trimmed
            return trimmed

        # Fetch new candles from cache end to decision_at
        new_data_start_ms = cache_newest_ms + 60_000  # 1 minute after last candle
        minutes_to_fetch = max(0, (decision_ms - new_data_start_ms) // 60_000)

        if minutes_to_fetch > 0:
            new_df = _fetch_klines_paginated(symbol, minutes_to_fetch + 10)  # +10 margin
            if not new_df.empty:
                combined = pd.concat([cached, new_df], ignore_index=True)
                combined = combined.drop_duplicates(subset=["open_time_ms"], keep="last")
                combined = combined.sort_values("open_time_ms").reset_index(drop=True)
            else:
                combined = cached
        else:
            combined = cached

        # Trim old data beyond warmup window
        cutoff_ms = decision_ms - (WARMUP_MINUTES * 60 * 1000)
        trimmed = combined[combined["open_time_ms"] >= cutoff_ms].copy()

        with self._lock:
            self._cache[symbol] = trimmed
            self._last_fetch_time[symbol] = datetime.now(timezone.utc)

        return trimmed

    def _fetch_full(
        self,
        symbol: str,
        decision_at: datetime,
        required_minutes: int,
    ) -> pd.DataFrame:
        """Full paginated fetch for cold start."""
        df = _fetch_klines_paginated(symbol, required_minutes)

        with self._lock:
            self._cache[symbol] = df
            self._last_fetch_time[symbol] = datetime.now(timezone.utc)

        return df

    def invalidate(self, symbol: str | None = None) -> None:
        """Invalidate cache for a symbol or all symbols."""
        with self._lock:
            if symbol:
                self._cache.pop(symbol, None)
                self._last_fetch_time.pop(symbol, None)
            else:
                self._cache.clear()
                self._last_fetch_time.clear()

    def stats(self) -> dict[str, Any]:
        """Return cache statistics."""
        with self._lock:
            return {
                "symbols_cached": list(self._cache.keys()),
                "candle_counts": {s: len(df) for s, df in self._cache.items()},
                "last_fetch_times": {
                    s: t.isoformat() for s, t in self._last_fetch_time.items()
                },
            }


# Module-level singleton cache
_rolling_cache = RollingCandleCache()


def _frozen_universe() -> list[str]:
    return sorted(FROZEN_E4_UNIVERSE)


def _fetch_klines_page(
    symbol: str,
    interval: str = "1m",
    limit: int = BINANCE_LIMIT_PER_REQUEST,
    end_time_ms: int | None = None,
) -> pd.DataFrame:
    """Fetch a single page of 1m klines from Binance Futures REST.

    Args:
        symbol: Trading pair symbol
        interval: Candle interval
        limit: Max candles per request (Binance max 1500)
        end_time_ms: If provided, fetch candles ending before this timestamp

    Returns:
        DataFrame with candle data
    """
    url = f"{BINANCE_FUTURES_REST}{KLINE_ENDPOINT}"
    params: dict[str, Any] = {"symbol": symbol, "interval": interval, "limit": limit}
    if end_time_ms is not None:
        params["endTime"] = end_time_ms

    resp = requests.get(url, params=params, timeout=REQUEST_TIMEOUT_S)
    resp.raise_for_status()
    raw = resp.json()

    if not raw:
        return pd.DataFrame()

    df = pd.DataFrame(raw, columns=[
        "open_time_ms", "open", "high", "low", "close", "volume",
        "close_time_ms", "quote_volume", "trades", "taker_buy_volume",
        "taker_buy_quote_volume", "ignore",
    ])

    for col in ["open", "high", "low", "close", "volume", "taker_buy_volume",
                 "quote_volume", "taker_buy_quote_volume", "open_time_ms", "close_time_ms"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df[["open_time_ms", "open", "high", "low", "close", "volume",
             "taker_buy_volume", "quote_volume"]].copy()

    df["open_time"] = pd.to_datetime(df["open_time_ms"], unit="ms", utc=True)
    df["close_time"] = pd.to_datetime(df["close_time_ms"], unit="ms", utc=True)

    return df


def _fetch_klines_paginated(
    symbol: str,
    required_minutes: int,
    interval: str = "1m",
) -> pd.DataFrame:
    """Fetch enough 1m klines to cover required_minutes, using pagination.

    Binance REST limit is 1500 per request. For 24,000 minutes we need
    ~16 sequential requests per symbol. Uses endTime backward iteration.
    """
    pages_needed = -(-required_minutes // BINANCE_LIMIT_PER_REQUEST)  # ceil division
    all_pages: list[pd.DataFrame] = []
    end_time_ms: int | None = None

    for page_idx in range(pages_needed):
        if page_idx > 0:
            time.sleep(INTER_REQUEST_GAP_MS / 1000.0)

        df = _fetch_klines_page(symbol, interval, BINANCE_LIMIT_PER_REQUEST, end_time_ms)
        if df.empty:
            break

        all_pages.append(df)

        # Next page ends before the earliest candle in this page
        earliest_ms = int(df["open_time_ms"].min())
        end_time_ms = earliest_ms - 1  # 1ms before earliest

        # Stop if we have enough data
        total_minutes = sum(len(p) for p in all_pages)
        if total_minutes >= required_minutes:
            break

    if not all_pages:
        raise ValueError(f"EMPTY_KLINES: {symbol} returned 0 candles after {pages_needed} pages")

    result = pd.concat(all_pages, ignore_index=True)
    result = result.drop_duplicates(subset=["open_time_ms"], keep="last")
    result = result.sort_values("open_time_ms").reset_index(drop=True)

    logger.debug(
        "Fetched %d candles for %s across %d pages (required %d min)",
        len(result), symbol, len(all_pages), required_minutes,
    )

    return result


def fetch_snapshot(decision_at: datetime | None = None) -> MarketSnapshot:
    """Fetch a complete snapshot of all 11 E4 symbols.

    Uses the rolling cache to minimize REST fetches. On cold start,
    fetches WARMUP_MINUTES (~24,000) per symbol via pagination.
    On warm cycles, only fetches recent candles incrementally.

    If decision_at is None, uses the current time floored to 5 minutes.
    The snapshot is immutable once created.

    Raises ValueError if any symbol fails or universe is incomplete.
    """
    if decision_at is None:
        now = datetime.now(timezone.utc)
        decision_at = now.replace(
            minute=(now.minute // 5) * 5, second=0, microsecond=0
        )

    captured_at = datetime.now(timezone.utc)
    universe = _frozen_universe()
    candles_by_symbol: dict[str, pd.DataFrame] = {}
    source_timestamps: dict[str, datetime] = {}
    source_feed_lag_ms: dict[str, float] = {}

    for symbol in universe:
        df = _rolling_cache.get_or_fetch(symbol, decision_at, WARMUP_MINUTES)
        candles_by_symbol[symbol] = df

        if len(df) == 0:
            raise ValueError(f"EMPTY_CANDLES: {symbol}")

        last_close = df["close_time"].iloc[-1]
        if isinstance(last_close, pd.Timestamp):
            if last_close.tzinfo is None:
                last_close = last_close.tz_localize("UTC")
            else:
                last_close = last_close.tz_convert("UTC")
        else:
            last_close = pd.Timestamp(last_close, tz="UTC")
        source_timestamps[symbol] = last_close.to_pydatetime()

        lag_ms = max(0.0, (decision_at - last_close.to_pydatetime()).total_seconds() * 1000)
        source_feed_lag_ms[symbol] = lag_ms

    if len(candles_by_symbol) != len(universe):
        raise ValueError(
            f"UNIVERSE_INCOMPLETE: got {len(candles_by_symbol)}/{len(universe)} symbols"
        )

    missing = set(universe) - set(candles_by_symbol.keys())
    extra = set(candles_by_symbol.keys()) - set(universe)
    if missing or extra:
        raise ValueError(f"UNIVERSE_MISMATCH: missing={missing}, extra={extra}")

    snapshot_hash = _compute_snapshot_hash(candles_by_symbol, decision_at)
    snapshot_id = f"snap_{decision_at.strftime('%Y%m%dT%H%M%SZ')}_{snapshot_hash[:12]}"

    return MarketSnapshot(
        snapshot_id=snapshot_id,
        decision_at=decision_at,
        captured_at=captured_at,
        candles_by_symbol=candles_by_symbol,
        snapshot_hash=snapshot_hash,
        source_timestamps=source_timestamps,
        source_feed_lag_ms=source_feed_lag_ms,
    )


def get_cache_stats() -> dict[str, Any]:
    """Return rolling cache statistics for observability."""
    return _rolling_cache.stats()


def invalidate_cache(symbol: str | None = None) -> None:
    """Invalidate the rolling cache (for testing or error recovery)."""
    _rolling_cache.invalidate(symbol)


def _compute_snapshot_hash(
    candles_by_symbol: dict[str, pd.DataFrame],
    decision_at: datetime,
) -> str:
    """Compute SHA-256 hash of the snapshot for immutability verification.

    Hashes actual OHLCV values (not just row counts) to ensure the hash
    represents the actual data content. Uses last 100 candles per symbol
    for efficiency while maintaining data integrity.
    """
    parts = [f"decision_at={decision_at.isoformat()}"]
    for symbol in sorted(candles_by_symbol.keys()):
        df = candles_by_symbol[symbol]
        # Hash last 100 candles' OHLCV values for efficiency
        tail = df.tail(100)
        ohlcv_str = tail[["open", "high", "low", "close", "volume"]].to_string(index=False)
        parts.append(f"{symbol}:{len(df)}:{hashlib.sha256(ohlcv_str.encode()).hexdigest()[:16]}")
    content = "|".join(parts)
    return hashlib.sha256(content.encode()).hexdigest()
