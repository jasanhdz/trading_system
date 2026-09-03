#!/usr/bin/env python3
"""Refresh an isolated LONG v2.1 candle delta using public GET requests only."""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
import time
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from aegis.config import CANONICAL_SYMBOLS

HOST = "fapi.binance.com"
BASE_URL = f"https://{HOST}"
PATH = "/fapi/v1/klines"
INTERVAL_MS = 5 * 60 * 1000


class RedirectProhibited(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):  # type: ignore[no-untyped-def]
        raise RuntimeError("AEGIS_LONG_V21_PUBLIC_CANDLE_REDIRECT_PROHIBITED")


def _db_symbol(symbol: str) -> str:
    return f"{symbol[:-4]}/USDT"


def _milliseconds(value: str) -> int:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return int(parsed.timestamp() * 1000)


def refresh(
    database: Path, output: Path, *, now_ms: int | None = None
) -> dict[str, Any]:
    current_ms = now_ms or int(datetime.now(timezone.utc).timestamp() * 1000)
    connection = sqlite3.connect(f"file:{database.resolve()}?mode=ro", uri=True)
    try:
        starts = {
            symbol: _milliseconds(
                str(
                    connection.execute(
                        "SELECT MAX(timestamp) FROM ohlcv_data "
                        "WHERE symbol=? AND timeframe='5m'",
                        (_db_symbol(symbol),),
                    ).fetchone()[0]
                )
            )
            + INTERVAL_MS
            for symbol in CANONICAL_SYMBOLS
        }
    finally:
        connection.close()

    opener = urllib.request.build_opener(RedirectProhibited())
    rows: list[dict[str, Any]] = []
    requests = 0
    for symbol in CANONICAL_SYMBOLS:
        cursor = starts[symbol]
        while cursor < current_ms:
            query = urllib.parse.urlencode(
                {
                    "symbol": symbol,
                    "interval": "5m",
                    "startTime": cursor,
                    "endTime": current_ms,
                    "limit": 1000,
                }
            )
            url = f"{BASE_URL}{PATH}?{query}"
            request = urllib.request.Request(url, method="GET")
            with opener.open(request, timeout=15.0) as response:
                if urllib.parse.urlparse(response.geturl()).hostname != HOST:
                    raise RuntimeError("AEGIS_LONG_V21_PUBLIC_CANDLE_WRONG_HOST")
                payload = json.loads(response.read().decode("utf-8"))
            requests += 1
            if not isinstance(payload, list):
                raise RuntimeError("AEGIS_LONG_V21_PUBLIC_CANDLE_RESPONSE_INVALID")
            if not payload:
                break
            accepted = 0
            for item in payload:
                if not isinstance(item, list) or len(item) < 7:
                    raise RuntimeError("AEGIS_LONG_V21_PUBLIC_CANDLE_RESPONSE_INVALID")
                open_ms = int(item[0])
                close_ms = int(item[6])
                if close_ms >= current_ms:
                    continue
                rows.append(
                    {
                        "symbol": symbol,
                        "timeframe": "5m",
                        "open_time_ms": open_ms,
                        "open": float(item[1]),
                        "high": float(item[2]),
                        "low": float(item[3]),
                        "close": float(item[4]),
                        "volume": float(item[5]),
                        "source": "BINANCE_USDM_PUBLIC_KLINES_GET",
                    }
                )
                accepted += 1
            next_cursor = int(payload[-1][0]) + INTERVAL_MS
            if next_cursor <= cursor:
                raise RuntimeError("AEGIS_LONG_V21_PUBLIC_CANDLE_CURSOR_STALLED")
            cursor = next_cursor
            if len(payload) < 1000:
                break
            time.sleep(0.05)

    identities = [(row["symbol"], row["open_time_ms"]) for row in rows]
    if len(identities) != len(set(identities)):
        raise RuntimeError("AEGIS_LONG_V21_PUBLIC_CANDLE_DUPLICATE")
    rows.sort(key=lambda row: (int(row["open_time_ms"]), str(row["symbol"])))
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")
    os.chmod(temporary, 0o600)
    temporary.replace(output)
    return {
        "output": str(output),
        "rows": len(rows),
        "symbols": len({str(row["symbol"]) for row in rows}),
        "public_get_requests": requests,
        "authenticated_requests": 0,
        "exchange_mutations": 0,
        "latest_open_time_ms": max(
            (int(row["open_time_ms"]) for row in rows), default=None
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--database", type=Path, default=Path("data/binance_candles.db")
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/long_entry_v21_shadow/public_candles_delta.jsonl"),
    )
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    database = args.database if args.database.is_absolute() else root / args.database
    output = args.output if args.output.is_absolute() else root / args.output
    print(json.dumps(refresh(database, output), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
