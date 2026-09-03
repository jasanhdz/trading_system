#!/usr/bin/env python3
"""Collect allowlisted public Binance USD-M microstructure streams for C2."""

from __future__ import annotations

import argparse
import json
import os
import time
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import websocket

from aegis.config import CANONICAL_SYMBOLS
from aegis.research.microstructure_events_c2 import (
    C2Archive,
    parse_aggregate_trade,
    parse_depth,
    parse_liquidation,
    parse_open_interest,
)


HOST = "fstream.binance.com"
BASE_URL = f"wss://{HOST}/stream?streams="
STREAM_KINDS = ("aggTrade", "forceOrder", "depth20@100ms")
OPEN_INTEREST_URL = "https://fapi.binance.com/futures/data/openInterestHist"


def stream_names() -> tuple[str, ...]:
    return tuple(
        f"{symbol.lower()}@{kind}"
        for symbol in CANONICAL_SYMBOLS
        for kind in STREAM_KINDS
    )


def normalize_message(message: str) -> tuple[str, Any]:
    payload = json.loads(message)
    if not isinstance(payload, Mapping) or not isinstance(payload.get("data"), Mapping):
        raise ValueError("AEGIS_C2_STREAM_ENVELOPE_INVALID")
    stream = str(payload.get("stream"))
    if stream not in stream_names():
        raise ValueError("AEGIS_C2_STREAM_NOT_ALLOWLISTED")
    data = payload["data"]
    if stream.endswith("@aggTrade"):
        return "agg_trade", parse_aggregate_trade(data)
    if stream.endswith("@forceOrder"):
        return "liquidation", parse_liquidation(data)
    if stream.endswith("@depth20@100ms"):
        return "depth", parse_depth(data)
    raise ValueError("AEGIS_C2_STREAM_NOT_ALLOWLISTED")


def collect_open_interest(archive: C2Archive) -> tuple[int, int]:
    accepted = duplicates = 0
    for symbol in CANONICAL_SYMBOLS:
        query = urllib.parse.urlencode({"symbol": symbol, "period": "5m", "limit": 2})
        request = urllib.request.Request(f"{OPEN_INTEREST_URL}?{query}", method="GET")
        with urllib.request.urlopen(request, timeout=15) as response:
            if urllib.parse.urlparse(response.geturl()).hostname != "fapi.binance.com":
                raise ValueError("AEGIS_C2_OPEN_INTEREST_REDIRECT_PROHIBITED")
            payload = json.loads(response.read().decode("utf-8"))
        if not isinstance(payload, list):
            raise ValueError("AEGIS_C2_OPEN_INTEREST_RESPONSE_INVALID")
        for item in payload:
            if archive.insert(parse_open_interest(item, symbol)):
                accepted += 1
            else:
                duplicates += 1
    return accepted, duplicates


def collect(output: Path, duration_seconds: int, *, poll_open_interest: bool = True) -> Mapping[str, Any]:
    if duration_seconds < 0:
        raise ValueError("AEGIS_C2_DURATION_INVALID")
    archive = C2Archive(output)
    os.chmod(output, 0o600)
    if duration_seconds == 0:
        archive.close()
        return {"state": "INITIALIZED_ONLY", "output": str(output), "public_streams": 0}
    url = BASE_URL + "/".join(stream_names())
    connection = websocket.create_connection(url, timeout=10, origin=f"https://{HOST}")
    accepted = duplicates = rejected = 0
    first_timestamp = last_timestamp = None
    deadline = time.monotonic() + duration_seconds
    next_oi_poll = 0.0
    try:
        while time.monotonic() < deadline:
            if poll_open_interest and time.monotonic() >= next_oi_poll:
                oi_accepted, oi_duplicates = collect_open_interest(archive)
                accepted += oi_accepted
                duplicates += oi_duplicates
                next_oi_poll = time.monotonic() + 300.0
            try:
                source, row = normalize_message(connection.recv())
                timestamp = int(
                    getattr(row, "trade_time_ms", 0)
                    or getattr(row, "order_trade_time_ms", 0)
                    or getattr(row, "transaction_time_ms", 0)
                )
                if archive.insert(row):
                    accepted += 1
                else:
                    duplicates += 1
                first_timestamp = timestamp if first_timestamp is None else min(first_timestamp, timestamp)
                last_timestamp = timestamp if last_timestamp is None else max(last_timestamp, timestamp)
            except (ValueError, KeyError, TypeError):
                rejected += 1
        archive.append_manifest({
            "created_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "source": "BINANCE_USDM_PUBLIC_COMBINED_STREAM",
            "first_timestamp_ms": first_timestamp,
            "last_timestamp_ms": last_timestamp,
            "accepted_rows": accepted,
            "duplicate_rows": duplicates,
            "rejected_rows": rejected,
        })
    finally:
        connection.close()
        archive.close()
    return {
        "state": "COLLECTION_COMPLETE", "output": str(output),
        "accepted_rows": accepted, "duplicate_rows": duplicates,
        "rejected_rows": rejected, "authenticated_requests": 0,
        "exchange_mutations": 0,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("data/market_microstructure_events_c2/c2_archive.db"))
    parser.add_argument("--duration-seconds", type=int, default=0)
    parser.add_argument("--no-open-interest", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    output = args.output if args.output.is_absolute() else root / args.output
    print(json.dumps(collect(output, args.duration_seconds, poll_open_interest=not args.no_open_interest), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
