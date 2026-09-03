#!/usr/bin/env python3
"""Download public Binance USD-M 1m klines for the read-only Live audit."""

from __future__ import annotations

import argparse
import json
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

import pandas as pd


COLUMNS = [
    "open_time_ms", "open", "high", "low", "close", "volume",
    "close_time_ms", "quote_volume", "trade_count", "taker_buy_volume",
    "taker_buy_quote_volume", "ignore",
]
DEFAULT_SYMBOLS = "ADAUSDT,AVAXUSDT,BNBUSDT,BTCUSDT,DOGEUSDT,ETHUSDT,LINKUSDT,LTCUSDT,SOLUSDT,SUIUSDT,XRPUSDT"


def fetch_page(symbol: str, start_ms: int, end_ms: int) -> list[list]:
    query = urllib.parse.urlencode({
        "symbol": symbol, "interval": "1m", "startTime": start_ms,
        "endTime": end_ms, "limit": 1500,
    })
    url = f"https://fapi.binance.com/fapi/v1/klines?{query}"
    for attempt in range(8):
        try:
            with urllib.request.urlopen(url, timeout=30) as response:
                payload = json.load(response)
            if not isinstance(payload, list):
                raise RuntimeError(f"unexpected Binance response for {symbol}: {payload}")
            return payload
        except (urllib.error.URLError, TimeoutError) as error:
            if attempt == 7:
                raise
            time.sleep(min(2 ** attempt, 30))
    raise RuntimeError("unreachable")


def download(symbol: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    cursor = int(start.timestamp() * 1000)
    end_ms = int(end.timestamp() * 1000) - 1
    pages: list[pd.DataFrame] = []
    while cursor <= end_ms:
        payload = fetch_page(symbol, cursor, end_ms)
        if not payload:
            break
        page = pd.DataFrame(payload, columns=COLUMNS)
        pages.append(page)
        next_cursor = int(page.iloc[-1]["open_time_ms"]) + 60_000
        if next_cursor <= cursor:
            raise RuntimeError(f"non-advancing Binance cursor for {symbol}")
        cursor = next_cursor
        time.sleep(0.08)
    if not pages:
        return pd.DataFrame(columns=COLUMNS)
    result = pd.concat(pages, ignore_index=True).drop_duplicates("open_time_ms").sort_values("open_time_ms")
    numeric = [column for column in COLUMNS if column != "ignore"]
    result[numeric] = result[numeric].apply(pd.to_numeric, errors="raise")
    return result.drop(columns="ignore")


def audit(frame: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> dict:
    times = pd.to_datetime(frame["open_time_ms"], unit="ms", utc=True)
    gaps = int(times.diff().dropna().ne(pd.Timedelta(minutes=1)).sum())
    return {
        "rows": int(len(frame)), "start": times.min().isoformat(), "end": times.max().isoformat(),
        "requested_start": start.isoformat(), "requested_end": end.isoformat(),
        "duplicate_open_times": int(frame["open_time_ms"].duplicated().sum()), "gaps": gaps,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbols", default=DEFAULT_SYMBOLS)
    parser.add_argument("--start", default="2026-05-01T00:00:00Z")
    parser.add_argument("--end", default="2026-08-16T00:00:00Z")
    parser.add_argument("--out-dir", type=Path, default=Path("data/live_entry_quality_audit_20260815/candles_1m"))
    args = parser.parse_args()
    start, end = pd.Timestamp(args.start), pd.Timestamp(args.end)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    manifest = {}
    for symbol in [item.strip().upper() for item in args.symbols.split(",") if item.strip()]:
        path = args.out_dir / f"{symbol}_1m.parquet"
        if path.exists():
            frame = pd.read_parquet(path)
            existing = audit(frame, start, end)
            if pd.Timestamp(existing["start"]) <= start and pd.Timestamp(existing["end"]) >= end - pd.Timedelta(minutes=1):
                manifest[symbol] = existing
                print(json.dumps({"symbol": symbol, "status": "REUSED", **existing}), flush=True)
                continue
        frame = download(symbol, start, end)
        if frame.empty:
            raise RuntimeError(f"no public 1m candles returned for {symbol}")
        frame.to_parquet(path, index=False)
        manifest[symbol] = audit(frame, start, end)
        print(json.dumps({"symbol": symbol, "status": "DOWNLOADED", **manifest[symbol]}), flush=True)
    (args.out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
