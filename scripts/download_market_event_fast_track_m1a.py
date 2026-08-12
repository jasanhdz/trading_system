#!/usr/bin/env python3
"""Download checksum-verified public Binance archives for M1A."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from aegis.config import CANONICAL_SYMBOLS
from aegis.research.binance_public_archive import (
    ArchiveRequest,
    BinancePublicArchiveClient,
    append_manifest,
)


def _months(start: str, end: str) -> tuple[str, ...]:
    start_year, start_month = map(int, start.split("-"))
    end_year, end_month = map(int, end.split("-"))
    cursor = start_year * 12 + start_month - 1
    finish = end_year * 12 + end_month - 1
    if cursor > finish:
        raise ValueError("start month must not follow end month")
    values = []
    while cursor <= finish:
        values.append(f"{cursor // 12:04d}-{cursor % 12 + 1:02d}")
        cursor += 1
    return tuple(values)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start-month", required=True)
    parser.add_argument("--end-month", required=True)
    parser.add_argument("--symbols", nargs="+", default=list(CANONICAL_SYMBOLS))
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["futures-klines", "futures-aggTrades", "spot-klines", "spot-aggTrades"],
        choices=[
            "futures-klines",
            "futures-aggTrades",
            "futures-fundingRate",
            "futures-markPriceKlines",
            "spot-klines",
            "spot-aggTrades",
        ],
    )
    parser.add_argument("--interval", default="1m")
    parser.add_argument(
        "--output-root", type=Path, default=Path("data/market_event_fast_track_m1a/raw")
    )
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    output = args.output_root if args.output_root.is_absolute() else root / args.output_root
    mapping = {
        "futures-klines": ("futures/um", "klines", args.interval),
        "futures-aggTrades": ("futures/um", "aggTrades", None),
        "futures-fundingRate": ("futures/um", "fundingRate", None),
        "futures-markPriceKlines": ("futures/um", "markPriceKlines", args.interval),
        "spot-klines": ("spot", "klines", args.interval),
        "spot-aggTrades": ("spot", "aggTrades", None),
    }
    client = BinancePublicArchiveClient()
    evidence = []
    for month in _months(args.start_month, args.end_month):
        for symbol in args.symbols:
            for dataset in args.datasets:
                market, data_type, interval = mapping[dataset]
                evidence.append(
                    client.download(
                        ArchiveRequest(market, data_type, symbol, month, interval), output
                    )
                )
    manifest = output.parent / "archive_manifest.jsonl"
    append_manifest(manifest, evidence)
    print(
        json.dumps(
            {
                "archives": len(evidence),
                "downloaded": sum(item.downloaded for item in evidence),
                "bytes": sum(item.byte_size for item in evidence),
                "manifest": str(manifest),
                "authenticated_requests": 0,
                "exchange_mutations": 0,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
