#!/usr/bin/env python3
"""Build a resumable public-only USD-M microstructure evidence database."""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Mapping

from aegis.config import CANONICAL_SYMBOLS


HOST = "fapi.binance.com"
BASE_URL = f"https://{HOST}"
KLINES_PATH = "/fapi/v1/klines"
FUNDING_PATH = "/fapi/v1/fundingRate"
OPEN_INTEREST_PATH = "/futures/data/openInterestHist"
TAKER_RATIO_PATH = "/futures/data/takerlongshortRatio"
DEPTH_PATH = "/fapi/v1/depth"
INTERVAL_MS = 5 * 60 * 1000


class RedirectProhibited(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):  # type: ignore[no-untyped-def]
        raise RuntimeError("AEGIS_LONG_V3_PUBLIC_REDIRECT_PROHIBITED")


def _initialize(connection: sqlite3.Connection) -> None:
    connection.executescript(
        """
        PRAGMA journal_mode=WAL;
        PRAGMA synchronous=NORMAL;
        CREATE TABLE IF NOT EXISTS kline_microstructure (
          symbol TEXT NOT NULL,
          open_time_ms INTEGER NOT NULL,
          quote_volume REAL NOT NULL,
          trade_count INTEGER NOT NULL,
          taker_buy_base REAL NOT NULL,
          taker_buy_quote REAL NOT NULL,
          PRIMARY KEY(symbol, open_time_ms)
        );
        CREATE TABLE IF NOT EXISTS funding_history (
          symbol TEXT NOT NULL,
          funding_time_ms INTEGER NOT NULL,
          funding_rate REAL NOT NULL,
          mark_price REAL,
          PRIMARY KEY(symbol, funding_time_ms)
        );
        CREATE TABLE IF NOT EXISTS open_interest_recent (
          symbol TEXT NOT NULL,
          timestamp_ms INTEGER NOT NULL,
          open_interest REAL NOT NULL,
          open_interest_value REAL NOT NULL,
          PRIMARY KEY(symbol, timestamp_ms)
        );
        CREATE TABLE IF NOT EXISTS taker_ratio_recent (
          symbol TEXT NOT NULL,
          timestamp_ms INTEGER NOT NULL,
          buy_sell_ratio REAL NOT NULL,
          buy_volume REAL NOT NULL,
          sell_volume REAL NOT NULL,
          PRIMARY KEY(symbol, timestamp_ms)
        );
        CREATE TABLE IF NOT EXISTS depth_snapshots (
          symbol TEXT NOT NULL,
          transaction_time_ms INTEGER NOT NULL,
          bid_notional_20 REAL NOT NULL,
          ask_notional_20 REAL NOT NULL,
          imbalance_20 REAL NOT NULL,
          PRIMARY KEY(symbol, transaction_time_ms)
        );
        CREATE TABLE IF NOT EXISTS collection_manifest (
          key TEXT PRIMARY KEY,
          value TEXT NOT NULL
        );
        """
    )


class PublicGetClient:
    def __init__(self, delay_seconds: float) -> None:
        self.opener = urllib.request.build_opener(RedirectProhibited())
        self.delay_seconds = delay_seconds
        self.requests = 0

    def get(self, path: str, params: Mapping[str, Any]) -> Any:
        if path not in {
            KLINES_PATH,
            FUNDING_PATH,
            OPEN_INTEREST_PATH,
            TAKER_RATIO_PATH,
            DEPTH_PATH,
        }:
            raise RuntimeError("AEGIS_LONG_V3_PUBLIC_ENDPOINT_NOT_ALLOWLISTED")
        query = urllib.parse.urlencode(params)
        request = urllib.request.Request(f"{BASE_URL}{path}?{query}", method="GET")
        for attempt in range(5):
            try:
                with self.opener.open(request, timeout=20.0) as response:
                    target = urllib.parse.urlparse(response.geturl())
                    if target.hostname != HOST or target.scheme != "https":
                        raise RuntimeError("AEGIS_LONG_V3_PUBLIC_WRONG_HOST")
                    payload = json.loads(response.read().decode("utf-8"))
                self.requests += 1
                time.sleep(self.delay_seconds)
                return payload
            except urllib.error.HTTPError as exc:
                if exc.code not in {418, 429, 500, 502, 503, 504} or attempt == 4:
                    raise
                retry_after = float(exc.headers.get("Retry-After", "2"))
                time.sleep(max(retry_after, 2.0 * (attempt + 1)))
        raise RuntimeError("AEGIS_LONG_V3_PUBLIC_GET_FAILED")


def _max_time(
    connection: sqlite3.Connection, table: str, field: str, symbol: str
) -> int | None:
    value = connection.execute(
        f"SELECT MAX({field}) FROM {table} WHERE symbol=?", (symbol,)
    ).fetchone()[0]
    return int(value) if value is not None else None


def _collect_klines(
    connection: sqlite3.Connection,
    client: PublicGetClient,
    symbol: str,
    start_ms: int,
    end_ms: int,
) -> int:
    stored = _max_time(connection, "kline_microstructure", "open_time_ms", symbol)
    cursor = max(start_ms, (stored + INTERVAL_MS) if stored is not None else start_ms)
    count = 0
    while cursor < end_ms:
        payload = client.get(
            KLINES_PATH,
            {
                "symbol": symbol,
                "interval": "5m",
                "startTime": cursor,
                "endTime": end_ms,
                "limit": 1500,
            },
        )
        if not isinstance(payload, list) or not payload:
            break
        rows = []
        for item in payload:
            if not isinstance(item, list) or len(item) < 11:
                raise RuntimeError("AEGIS_LONG_V3_KLINE_RESPONSE_INVALID")
            if int(item[6]) >= end_ms:
                continue
            rows.append(
                (
                    symbol,
                    int(item[0]),
                    float(item[7]),
                    int(item[8]),
                    float(item[9]),
                    float(item[10]),
                )
            )
        connection.executemany(
            "INSERT OR REPLACE INTO kline_microstructure VALUES(?,?,?,?,?,?)", rows
        )
        connection.commit()
        count += len(rows)
        next_cursor = int(payload[-1][0]) + INTERVAL_MS
        if next_cursor <= cursor:
            raise RuntimeError("AEGIS_LONG_V3_KLINE_CURSOR_STALLED")
        cursor = next_cursor
        if len(payload) < 1500:
            break
    return count


def _collect_funding(
    connection: sqlite3.Connection,
    client: PublicGetClient,
    symbol: str,
    start_ms: int,
    end_ms: int,
) -> int:
    stored = _max_time(connection, "funding_history", "funding_time_ms", symbol)
    cursor = max(start_ms, (stored + 1) if stored is not None else start_ms)
    count = 0
    while cursor < end_ms:
        payload = client.get(
            FUNDING_PATH,
            {"symbol": symbol, "startTime": cursor, "endTime": end_ms, "limit": 1000},
        )
        if not isinstance(payload, list) or not payload:
            break
        rows = [
            (
                symbol,
                int(item["fundingTime"]),
                float(item["fundingRate"]),
                float(item["markPrice"]) if item.get("markPrice") is not None else None,
            )
            for item in payload
        ]
        connection.executemany(
            "INSERT OR REPLACE INTO funding_history VALUES(?,?,?,?)", rows
        )
        connection.commit()
        count += len(rows)
        next_cursor = int(payload[-1]["fundingTime"]) + 1
        if next_cursor <= cursor:
            raise RuntimeError("AEGIS_LONG_V3_FUNDING_CURSOR_STALLED")
        cursor = next_cursor
        if len(payload) < 1000:
            break
    return count


def _collect_recent_series(
    connection: sqlite3.Connection,
    client: PublicGetClient,
    symbol: str,
    path: str,
    start_ms: int,
    end_ms: int,
) -> int:
    table = "open_interest_recent" if path == OPEN_INTEREST_PATH else "taker_ratio_recent"
    # These endpoints cap one response at 500 and may return the latest page even
    # when a historical cursor is supplied. They are prospective context only,
    # so capture one explicitly bounded snapshot rather than claiming backfill.
    del start_ms, end_ms
    payload = client.get(path, {"symbol": symbol, "period": "5m", "limit": 500})
    if not isinstance(payload, list):
        raise RuntimeError("AEGIS_LONG_V3_RECENT_RESPONSE_INVALID")
    if path == OPEN_INTEREST_PATH:
        rows = [
            (
                symbol,
                int(item["timestamp"]),
                float(item["sumOpenInterest"]),
                float(item["sumOpenInterestValue"]),
            )
            for item in payload
        ]
    else:
        rows = [
            (
                symbol,
                int(item["timestamp"]),
                float(item["buySellRatio"]),
                float(item["buyVol"]),
                float(item["sellVol"]),
            )
            for item in payload
        ]
    placeholders = "?,?,?,?" if path == OPEN_INTEREST_PATH else "?,?,?,?,?"
    connection.executemany(
        f"INSERT OR REPLACE INTO {table} VALUES({placeholders})", rows
    )
    connection.commit()
    return len(rows)


def _collect_depth(
    connection: sqlite3.Connection, client: PublicGetClient, symbol: str
) -> int:
    payload = client.get(DEPTH_PATH, {"symbol": symbol, "limit": 20})
    if not isinstance(payload, Mapping):
        raise RuntimeError("AEGIS_LONG_V3_DEPTH_RESPONSE_INVALID")
    bids = sum(float(price) * float(quantity) for price, quantity in payload["bids"])
    asks = sum(float(price) * float(quantity) for price, quantity in payload["asks"])
    total = bids + asks
    imbalance = (bids - asks) / total if total > 0.0 else 0.0
    timestamp = int(payload.get("T") or payload.get("E") or time.time() * 1000)
    connection.execute(
        "INSERT OR REPLACE INTO depth_snapshots VALUES(?,?,?,?,?)",
        (symbol, timestamp, bids, asks, imbalance),
    )
    connection.commit()
    return 1


def refresh(output: Path, lookback_days: int, delay_seconds: float) -> Mapping[str, Any]:
    now = datetime.now(timezone.utc)
    end_ms = int(now.timestamp() * 1000)
    start_ms = int((now - timedelta(days=lookback_days)).timestamp() * 1000)
    recent_ms = int((now - timedelta(days=30)).timestamp() * 1000)
    output.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(output)
    os.chmod(output, 0o600)
    client = PublicGetClient(delay_seconds)
    counts: dict[str, int] = {}
    try:
        _initialize(connection)
        for symbol in CANONICAL_SYMBOLS:
            counts[f"{symbol}:klines"] = _collect_klines(
                connection, client, symbol, start_ms, end_ms
            )
            counts[f"{symbol}:funding"] = _collect_funding(
                connection, client, symbol, start_ms, end_ms
            )
            counts[f"{symbol}:open_interest_recent"] = _collect_recent_series(
                connection, client, symbol, OPEN_INTEREST_PATH, recent_ms, end_ms
            )
            counts[f"{symbol}:taker_ratio_recent"] = _collect_recent_series(
                connection, client, symbol, TAKER_RATIO_PATH, recent_ms, end_ms
            )
            counts[f"{symbol}:depth"] = _collect_depth(connection, client, symbol)
        manifest = {
            "schema_id": "aegis-long-v3-public-microstructure-v1",
            "generated_at": now.isoformat(),
            "host": HOST,
            "method": "GET",
            "authenticated_requests": 0,
            "public_get_requests": client.requests,
            "exchange_mutations": 0,
            "lookback_days": lookback_days,
            "counts": counts,
        }
        connection.execute(
            "INSERT OR REPLACE INTO collection_manifest VALUES(?,?)",
            ("latest", json.dumps(manifest, sort_keys=True)),
        )
        connection.commit()
    finally:
        connection.close()
    os.chmod(output, 0o600)
    return {**manifest, "output": str(output)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/long_entry_v3_shadow/public_microstructure.db"),
    )
    parser.add_argument("--lookback-days", type=int, default=540)
    parser.add_argument("--delay-seconds", type=float, default=0.35)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    output = args.output if args.output.is_absolute() else root / args.output
    print(json.dumps(refresh(output, args.lookback_days, args.delay_seconds)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
