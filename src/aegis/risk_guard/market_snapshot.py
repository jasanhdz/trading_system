"""Live market snapshot provider for E4 precomputation.

Fetches 1-minute candles from Binance Futures REST API for the frozen
E4 universe (11 symbols). Produces immutable snapshots keyed by
decision_at (5-minute aligned).

NO autonomous order placement. NO trade execution.
This module is read-only market data acquisition.
"""

from __future__ import annotations

import hashlib
import logging
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
CANDLE_HISTORY_BARS = 512
REQUEST_TIMEOUT_S = 10
INTER_REQUEST_GAP_MS = 50


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


def _frozen_universe() -> list[str]:
    return sorted(FROZEN_E4_UNIVERSE)


def _fetch_klines(
    symbol: str,
    interval: str = "1m",
    limit: int = CANDLE_HISTORY_BARS,
) -> pd.DataFrame:
    """Fetch 1m klines from Binance Futures REST."""
    url = f"{BINANCE_FUTURES_REST}{KLINE_ENDPOINT}"
    params = {"symbol": symbol, "interval": interval, "limit": limit}

    resp = requests.get(url, params=params, timeout=REQUEST_TIMEOUT_S)
    resp.raise_for_status()
    raw = resp.json()

    if not raw:
        raise ValueError(f"EMPTY_KLINES: {symbol} returned 0 candles")

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


def fetch_snapshot(decision_at: datetime | None = None) -> MarketSnapshot:
    """Fetch a complete snapshot of all 11 E4 symbols.

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

    for i, symbol in enumerate(universe):
        if i > 0:
            time.sleep(INTER_REQUEST_GAP_MS / 1000.0)

        df = _fetch_klines(symbol)
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


def _compute_snapshot_hash(
    candles_by_symbol: dict[str, pd.DataFrame],
    decision_at: datetime,
) -> str:
    """Compute SHA-256 hash of the snapshot for immutability verification."""
    parts = [f"decision_at={decision_at.isoformat()}"]
    for symbol in sorted(candles_by_symbol.keys()):
        df = candles_by_symbol[symbol]
        parts.append(f"{symbol}:rows={len(df)}:last_close={df['close_time'].iloc[-1]}")
    content = "|".join(parts)
    return hashlib.sha256(content.encode()).hexdigest()
