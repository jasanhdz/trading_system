#!/usr/bin/env python3
"""Build the label-free independent general-market Phase 2 dataset."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import sys
from datetime import datetime
from pathlib import Path

SANDBOX = Path(__file__).resolve().parents[1]
REPOSITORY = SANDBOX.parents[1]
sys.path.insert(0, str(SANDBOX / "src"))
sys.path.insert(0, str(REPOSITORY / "src"))

from aegis_strategy_router.domain.serialization import utc_datetime
from aegis_strategy_router.adapters.shared_market_data import SharedNeutralMinuteCandleSource
from aegis_strategy_router.replay.general_market_pipeline import (
    GeneralMarketCandidatePipeline,
    merge_general_market_results,
    persist_general_market_result,
)


def parse_timestamp(value: str) -> datetime:
    return utc_datetime(datetime.fromisoformat(value.replace("Z", "+00:00")))


def run_symbol_partition(arguments: tuple[str, datetime, datetime, tuple[Path, ...]]):
    symbol, start_at, end_at, roots = arguments
    return GeneralMarketCandidatePipeline().run(
        symbols=(symbol,),
        start_at=start_at,
        end_at=end_at,
        candle_source=SharedNeutralMinuteCandleSource(roots),
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candle-root", type=Path, action="append", required=True)
    parser.add_argument("--symbols", required=True, help="Comma-separated frozen research universe")
    parser.add_argument("--start", type=parse_timestamp, required=True)
    parser.add_argument("--end", type=parse_timestamp, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()
    symbols = tuple(sorted({value.strip().upper() for value in args.symbols.split(",") if value.strip()}))
    if args.workers < 1:
        parser.error("--workers must be positive")
    if args.workers == 1 or len(symbols) == 1:
        result = GeneralMarketCandidatePipeline().run(
            symbols=symbols,
            start_at=args.start,
            end_at=args.end,
            candle_source=SharedNeutralMinuteCandleSource(args.candle_root),
        )
    else:
        work = tuple(
            (symbol, args.start, args.end, tuple(args.candle_root))
            for symbol in symbols
        )
        with ProcessPoolExecutor(max_workers=min(args.workers, len(symbols))) as executor:
            result = merge_general_market_results(executor.map(run_symbol_partition, work))
    persist_general_market_result(result, args.output)
    print(json.dumps(result.manifest(), indent=2, sort_keys=True))
    return 0 if result.snapshots else 2


if __name__ == "__main__":
    raise SystemExit(main())
