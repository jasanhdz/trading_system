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
import os
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import requests

from .feature_bridge import (
    ANCHOR_CADENCE_MINUTES,
    ANCHOR_COUNT,
    FROZEN_E4_TIMEFROZEN,
    FROZEN_E4_UNIVERSE,
)

logger = logging.getLogger(__name__)

BINANCE_FUTURES_REST = "https://fapi.binance.com"
KLINE_ENDPOINT = "/fapi/v1/klines"
REQUEST_TIMEOUT_S = 10
INTER_REQUEST_GAP_MS = 50
REQUEST_MAX_ATTEMPTS = 4
REQUEST_BACKOFF_S = 1.0

MAX_INDICATOR_WARMUP_BARS = 99


def _derive_minimum_warmup_minutes() -> int:
    """Find the smallest window that supplies 99 tf240 bars at every anchor."""
    timeframe = max(FROZEN_E4_TIMEFROZEN)
    anchor_lookback = ANCHOR_COUNT * ANCHOR_CADENCE_MINUTES
    base = pd.Timestamp("2024-01-01T00:00:00Z")

    def complete_bars(window: int, phase: int) -> int:
        decision_at = base + pd.Timedelta(minutes=phase)
        earliest_anchor = decision_at - pd.Timedelta(minutes=anchor_lookback)
        first_open = (decision_at - pd.Timedelta(minutes=window)).ceil(
            f"{timeframe}min"
        )
        last_close = earliest_anchor.floor(f"{timeframe}min")
        return max(0, int((last_close - first_open) / pd.Timedelta(minutes=timeframe)))

    phases = range(0, timeframe, ANCHOR_CADENCE_MINUTES)
    candidate = timeframe * MAX_INDICATOR_WARMUP_BARS
    while min(complete_bars(candidate, phase) for phase in phases) < MAX_INDICATOR_WARMUP_BARS:
        candidate += 1
    return candidate


WARMUP_MINUTES = _derive_minimum_warmup_minutes()
BINANCE_LIMIT_PER_REQUEST = 1500
PAGES_PER_SYMBOL = -(-WARMUP_MINUTES // BINANCE_LIMIT_PER_REQUEST)
REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_HISTORY_SEED_ROOT = Path(os.environ.get(
    "E4_HISTORY_SEED_ROOT",
    REPO_ROOT / "data/independent_entry_quality_discovery_v1/candles_1m",
))
DEFAULT_DURABLE_CACHE_ROOT = Path(os.environ.get(
    "E4_DURABLE_CACHE_ROOT",
    REPO_ROOT / "data/e4_live_candle_cache",
))


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

    def __init__(
        self,
        seed_root: Path | None = None,
        durable_root: Path | None = None,
        require_history: bool = False,
    ) -> None:
        self._cache: dict[str, pd.DataFrame] = {}
        self._last_fetch_time: dict[str, datetime] = {}
        self._lock = threading.Lock()
        self._seed_root = seed_root
        self._durable_root = durable_root
        self._require_history = require_history

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
        decision_at = _normalize_decision_at(decision_at)
        with self._lock:
            cached_copy = self._cache.get(symbol)

        if cached_copy is None:
            cached_copy = self._load_durable_history(symbol)

        if cached_copy is None or cached_copy.empty:
            if self._require_history:
                raise ValueError(f"E4_HISTORY_SEED_UNAVAILABLE:{symbol}")
            updated = self._fetch_full(symbol, decision_at, required_minutes)
        else:
            causal_cached = _filter_through_decision(cached_copy, decision_at)
            trimmed_cached = _trim_causal(causal_cached, decision_at, required_minutes)
            if _has_required_coverage(trimmed_cached, decision_at, required_minutes):
                updated = causal_cached
            elif not causal_cached.empty:
                updated = self._fetch_incremental(
                    symbol, causal_cached, decision_at, required_minutes
                )
            else:
                updated = self._fetch_full(symbol, decision_at, required_minutes)

        updated = _canonical_runtime_candles(updated)
        _assert_causal(updated, decision_at)
        with self._lock:
            self._cache[symbol] = updated
            self._last_fetch_time[symbol] = datetime.now(timezone.utc)
        return updated

    def _fetch_incremental(
        self,
        symbol: str,
        cached: pd.DataFrame,
        decision_at: datetime,
        required_minutes: int,
    ) -> pd.DataFrame:
        """Fetch only new candles since cache ended, merge, and trim old."""
        decision_ms = int(decision_at.timestamp() * 1000)
        start_time_ms = int(cached["open_time_ms"].max()) + 60_000
        if start_time_ms >= decision_ms:
            return _filter_through_decision(cached, decision_at)

        new_df = _fetch_klines_forward(
            symbol=symbol,
            start_time_ms=start_time_ms,
            end_time_ms=decision_ms - 1,
        )
        self._persist_delta(symbol, new_df)
        combined = pd.concat([cached, new_df], ignore_index=True)
        combined = combined.drop_duplicates(subset=["open_time_ms"], keep="last")
        combined = combined.sort_values("open_time_ms").reset_index(drop=True)
        causal = _filter_through_decision(combined, decision_at)
        recent = _trim_causal(causal, decision_at, required_minutes)
        if not _has_required_coverage(recent, decision_at, required_minutes):
            return self._fetch_full(symbol, decision_at, required_minutes)
        return causal

    def _fetch_full(
        self,
        symbol: str,
        decision_at: datetime,
        required_minutes: int,
    ) -> pd.DataFrame:
        """Full paginated fetch for cold start."""
        decision_ms = int(decision_at.timestamp() * 1000)
        df = _fetch_klines_paginated(
            symbol,
            required_minutes,
            end_time_ms=decision_ms - 1,
        )
        causal = _trim_causal(df, decision_at, required_minutes)
        if not _has_required_coverage(causal, decision_at, required_minutes):
            raise ValueError(
                f"INSUFFICIENT_WARMUP:{symbol}:{len(causal)}/{required_minutes}"
            )
        return causal

    def _load_durable_history(self, symbol: str) -> pd.DataFrame | None:
        parts: list[pd.DataFrame] = []
        if self._seed_root is not None:
            seed = self._seed_root / f"{symbol}_1m.parquet"
            if seed.exists():
                parts.append(pd.read_parquet(seed))
        if self._durable_root is not None:
            symbol_root = self._durable_root / symbol
            if symbol_root.exists():
                parts.extend(pd.read_parquet(path) for path in sorted(symbol_root.glob("*.parquet")))
        if not parts:
            return None
        from .bootstrap import merge_candles

        disjoint = all(
            not part["open_time_ms"].duplicated().any()
            and (
                index == 0
                or parts[index - 1]["open_time_ms"].max()
                < part["open_time_ms"].min()
            )
            for index, part in enumerate(parts)
        )
        if disjoint:
            checked, _ = merge_candles(
                pd.DataFrame(), pd.concat(parts, ignore_index=True), f"cache-load:{symbol}"
            )
        else:
            checked = pd.DataFrame()
            for index, part in enumerate(parts):
                checked, _ = merge_candles(checked, part, f"cache-load:{symbol}:{index}")
        checked["open_time"] = pd.to_datetime(
            checked["open_time_ms"], unit="ms", utc=True
        )
        checked["close_time"] = checked["open_time"] + pd.Timedelta(minutes=1)
        return checked

    def _persist_delta(self, symbol: str, candles: pd.DataFrame) -> None:
        if self._durable_root is None or candles.empty:
            return
        working = candles.copy()
        working["_month"] = pd.to_datetime(
            working["open_time_ms"], unit="ms", utc=True
        ).dt.strftime("%Y-%m")
        symbol_root = self._durable_root / symbol
        # Bootstrap owns the strict overlap and crash-safe publication contract.
        from .bootstrap import atomic_write_parquet, merge_candles, process_lock

        with process_lock(self._durable_root):
            symbol_root.mkdir(parents=True, exist_ok=True)
            for month, chunk in working.groupby("_month", sort=True):
                path = symbol_root / f"{month}.parquet"
                chunk = chunk.drop(columns="_month")
                existing = pd.read_parquet(path) if path.exists() else pd.DataFrame()
                merge_candles(existing, chunk, f"live-cache:{symbol}:{month}")
                merged = pd.concat([existing, chunk], ignore_index=True)
                merged = (
                    merged.drop_duplicates(subset=["open_time_ms"], keep="first")
                    .sort_values("open_time_ms")
                    .reset_index(drop=True)
                )
                atomic_write_parquet(path, merged)

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
_rolling_cache = RollingCandleCache(
    seed_root=DEFAULT_HISTORY_SEED_ROOT,
    durable_root=DEFAULT_DURABLE_CACHE_ROOT,
    require_history=True,
)


def _frozen_universe() -> list[str]:
    return sorted(FROZEN_E4_UNIVERSE)


def _fetch_klines_page(
    symbol: str,
    interval: str = "1m",
    limit: int = BINANCE_LIMIT_PER_REQUEST,
    start_time_ms: int | None = None,
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
    if start_time_ms is not None:
        params["startTime"] = start_time_ms
    if end_time_ms is not None:
        params["endTime"] = end_time_ms

    resp: requests.Response | None = None
    for attempt in range(REQUEST_MAX_ATTEMPTS):
        try:
            resp = requests.get(url, params=params, timeout=REQUEST_TIMEOUT_S)
        except (requests.Timeout, requests.ConnectionError):
            if attempt + 1 == REQUEST_MAX_ATTEMPTS:
                raise
            delay = REQUEST_BACKOFF_S * (2**attempt)
            logger.warning(
                "Binance kline request failed for %s; retrying in %.1fs (%d/%d)",
                symbol, delay, attempt + 1, REQUEST_MAX_ATTEMPTS,
            )
            time.sleep(delay)
            continue

        retryable = resp.status_code in (418, 429) or resp.status_code >= 500
        if not retryable or attempt + 1 == REQUEST_MAX_ATTEMPTS:
            resp.raise_for_status()
            break
        retry_after = resp.headers.get("Retry-After")
        try:
            delay = (
                float(retry_after)
                if retry_after is not None
                else REQUEST_BACKOFF_S * (2**attempt)
            )
        except ValueError:
            delay = REQUEST_BACKOFF_S * (2**attempt)
        logger.warning(
            "Binance kline request returned %d for %s; retrying in %.1fs (%d/%d)",
            resp.status_code, symbol, delay, attempt + 1, REQUEST_MAX_ATTEMPTS,
        )
        time.sleep(max(0.0, delay))

    if resp is None:  # pragma: no cover - loop always returns or raises
        raise RuntimeError(f"BINANCE_REQUEST_EXHAUSTED:{symbol}")
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

    df["open_time"] = pd.to_datetime(df["open_time_ms"], unit="ms", utc=True)
    df["close_time"] = pd.to_datetime(df["close_time_ms"], unit="ms", utc=True)
    df = df[["open_time_ms", "open", "high", "low", "close", "volume",
             "taker_buy_volume", "quote_volume", "open_time", "close_time"]].copy()

    return df


def _fetch_klines_paginated(
    symbol: str,
    required_minutes: int,
    interval: str = "1m",
    end_time_ms: int | None = None,
) -> pd.DataFrame:
    """Fetch enough 1m klines to cover required_minutes, using pagination.

    Binance REST limit is 1500 per request. For 24,000 minutes we need
    ~16 sequential requests per symbol. Uses endTime backward iteration.
    """
    pages_needed = -(-required_minutes // BINANCE_LIMIT_PER_REQUEST)  # ceil division
    all_pages: list[pd.DataFrame] = []
    page_end_time_ms = end_time_ms

    for page_idx in range(pages_needed):
        if page_idx > 0:
            time.sleep(INTER_REQUEST_GAP_MS / 1000.0)

        df = _fetch_klines_page(
            symbol,
            interval,
            BINANCE_LIMIT_PER_REQUEST,
            end_time_ms=page_end_time_ms,
        )
        if df.empty:
            break

        all_pages.append(df)

        # Next page ends before the earliest candle in this page
        earliest_ms = int(df["open_time_ms"].min())
        page_end_time_ms = earliest_ms - 1

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


def _fetch_klines_forward(
    symbol: str,
    start_time_ms: int,
    end_time_ms: int,
    interval: str = "1m",
) -> pd.DataFrame:
    """Fetch the exact missing interval without reading beyond decision_at."""
    pages: list[pd.DataFrame] = []
    cursor = start_time_ms
    while cursor <= end_time_ms:
        if pages:
            time.sleep(INTER_REQUEST_GAP_MS / 1000.0)
        page = _fetch_klines_page(
            symbol,
            interval,
            BINANCE_LIMIT_PER_REQUEST,
            start_time_ms=cursor,
            end_time_ms=end_time_ms,
        )
        if page.empty:
            break
        pages.append(page)
        next_cursor = int(page["open_time_ms"].max()) + 60_000
        if next_cursor <= cursor:
            raise ValueError(f"NON_ADVANCING_PAGINATION:{symbol}:{cursor}")
        cursor = next_cursor
        if len(page) < BINANCE_LIMIT_PER_REQUEST:
            break
    if not pages:
        return pd.DataFrame()
    return (
        pd.concat(pages, ignore_index=True)
        .drop_duplicates(subset=["open_time_ms"], keep="last")
        .sort_values("open_time_ms")
        .reset_index(drop=True)
    )


def _normalize_decision_at(decision_at: datetime) -> datetime:
    if decision_at.tzinfo is None:
        return decision_at.replace(tzinfo=timezone.utc)
    return decision_at.astimezone(timezone.utc)


def _canonical_runtime_candles(candles: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "open_time_ms", "open", "high", "low", "close", "volume",
        "taker_buy_volume",
    ]
    result = candles.loc[:, columns].copy()
    result["open_time"] = pd.to_datetime(
        result["open_time_ms"], unit="ms", utc=True
    )
    result["close_time"] = result["open_time"] + pd.Timedelta(minutes=1)
    return result


def _trim_causal(
    candles: pd.DataFrame,
    decision_at: datetime,
    required_minutes: int,
) -> pd.DataFrame:
    decision_at = _normalize_decision_at(decision_at)
    cutoff = pd.Timestamp(decision_at) - pd.Timedelta(minutes=required_minutes)
    close_time = pd.to_datetime(candles["close_time"], utc=True)
    open_time = pd.to_datetime(candles["open_time"], utc=True)
    mask = close_time.le(pd.Timestamp(decision_at)) & open_time.ge(cutoff)
    return candles.loc[mask].sort_values("open_time_ms").reset_index(drop=True).copy()


def _filter_through_decision(
    candles: pd.DataFrame,
    decision_at: datetime,
) -> pd.DataFrame:
    close_time = pd.to_datetime(candles["close_time"], utc=True)
    return (
        candles.loc[close_time.le(pd.Timestamp(_normalize_decision_at(decision_at)))]
        .sort_values("open_time_ms")
        .reset_index(drop=True)
        .copy()
    )


def _has_required_coverage(
    candles: pd.DataFrame,
    decision_at: datetime,
    required_minutes: int,
) -> bool:
    if len(candles) < required_minutes:
        return False
    expected_last_open_ms = int(decision_at.timestamp() * 1000) - 60_000
    return int(candles["open_time_ms"].iloc[-1]) == expected_last_open_ms


def _assert_causal(candles: pd.DataFrame, decision_at: datetime) -> None:
    if candles.empty:
        raise ValueError("EMPTY_CAUSAL_CANDLES")
    latest_close = pd.to_datetime(candles["close_time"], utc=True).max()
    if latest_close > pd.Timestamp(_normalize_decision_at(decision_at)):
        raise ValueError(
            f"NON_CAUSAL_SNAPSHOT: latest_close={latest_close} decision_at={decision_at}"
        )


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

    decision_at = _normalize_decision_at(decision_at)
    captured_at = datetime.now(timezone.utc)
    universe = _frozen_universe()
    candles_by_symbol: dict[str, pd.DataFrame] = {}
    source_timestamps: dict[str, datetime] = {}
    source_feed_lag_ms: dict[str, float] = {}

    for symbol in universe:
        df = _rolling_cache.get_or_fetch(symbol, decision_at, WARMUP_MINUTES)
        _assert_causal(df, decision_at)
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

        lag_ms = (decision_at - last_close.to_pydatetime()).total_seconds() * 1000
        if lag_ms < 0:
            raise ValueError(f"NON_CAUSAL_SNAPSHOT:{symbol}:lag_ms={lag_ms}")
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
    """Hash every causal feature-driving candle field in deterministic order."""
    decision_at = _normalize_decision_at(decision_at)
    digest = hashlib.sha256(f"decision_at={decision_at.isoformat()}\n".encode())
    columns = [
        "open_time_ms", "open", "high", "low", "close", "volume",
        "taker_buy_volume",
    ]
    for symbol in sorted(candles_by_symbol.keys()):
        df = candles_by_symbol[symbol]
        if not df["open_time_ms"].is_monotonic_increasing:
            df = df.sort_values("open_time_ms")
        _assert_causal(df, decision_at)
        digest.update(f"symbol={symbol}\n".encode())
        for column in columns:
            dtype = "<i8" if column == "open_time_ms" else "<f8"
            values = np.ascontiguousarray(
                pd.to_numeric(df[column], errors="raise").to_numpy(dtype=dtype)
            )
            digest.update(f"{column}:{dtype}:{len(values)}\n".encode())
            digest.update(values.tobytes())
    return digest.hexdigest()
