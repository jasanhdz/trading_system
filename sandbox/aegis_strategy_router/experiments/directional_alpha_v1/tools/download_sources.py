#!/usr/bin/env python3
"""Download only frozen checksum-verified public Binance kline archives."""

from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


EXPERIMENT = Path(__file__).resolve().parents[1]
REPOSITORY = EXPERIMENT.parents[3]
sys.path.insert(0, str(REPOSITORY / "src"))

from aegis.research.binance_public_archive import (  # noqa: E402
    ArchiveRequest, BinancePublicArchiveClient, append_manifest, month_range,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=EXPERIMENT / "config/preregistration_v1.json")
    parser.add_argument("--output", type=Path, default=REPOSITORY / "data/directional_alpha_v1/raw")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    requests = [
        ArchiveRequest("futures/um", "klines", symbol, month, "1m")
        for symbol in config["symbols"]
        for month in month_range("2022-01", "2023-03")
    ]
    evidence = []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        pending = {
            executor.submit(BinancePublicArchiveClient(timeout_seconds=120).download, request, args.output): request
            for request in requests
        }
        for future in as_completed(pending):
            evidence.append(future.result())
    evidence.sort(key=lambda item: (item.request.symbol, item.request.month))
    append_manifest(args.output.parent / "archive_manifest.jsonl", evidence)
    print(json.dumps({"archives": len(evidence), "downloaded": sum(item.downloaded for item in evidence)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
