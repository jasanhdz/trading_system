#!/usr/bin/env python3
"""Build the preregistered C2A causal feature and future-outcome dataset."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import pandas as pd
import yaml

from aegis.config import CANONICAL_SYMBOLS
from aegis.research.event_path_quality_c2a import (
    build_path_dataset,
    contracts_from_preregistration,
    read_agg_trade_archives_chunked,
)
from aegis.research.market_event_fast_track_m1a import (
    read_kline_archive,
)
from aegis.utils import sha256_file


def _paths(root: Path, symbol: str, months: tuple[str, ...], kind: str) -> tuple[Path, ...]:
    if kind == "klines":
        base = root / "futures/um/monthly/klines" / symbol / "1m"
        paths = tuple(base / f"{symbol}-1m-{month}.zip" for month in months)
    else:
        base = root / "futures/um/monthly/aggTrades" / symbol
        paths = tuple(base / f"{symbol}-aggTrades-{month}.zip" for month in months)
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"AEGIS_C2A_ARCHIVES_MISSING:{','.join(missing)}")
    return paths


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--months", nargs="+", required=True)
    parser.add_argument("--symbols", nargs="+", default=list(CANONICAL_SYMBOLS))
    parser.add_argument(
        "--archive-root", type=Path,
        default=Path("data/market_event_fast_track_m1a/raw"),
    )
    parser.add_argument(
        "--config", type=Path,
        default=Path("config/experiments/aegis_event_path_quality_c2a.yaml"),
    )
    parser.add_argument(
        "--output-root", type=Path,
        default=Path("data/event_path_quality_c2a/dataset_01"),
    )
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    archive_root = args.archive_root if args.archive_root.is_absolute() else root / args.archive_root
    config_path = args.config if args.config.is_absolute() else root / args.config
    output = args.output_root if args.output_root.is_absolute() else root / args.output_root
    symbols = tuple(args.symbols)
    if not symbols or any(symbol not in CANONICAL_SYMBOLS for symbol in symbols):
        raise ValueError("AEGIS_C2A_SYMBOL_SET_INVALID")
    months = tuple(args.months)
    config = yaml.safe_load(config_path.read_text())
    contracts = contracts_from_preregistration(config)
    output.mkdir(parents=True, exist_ok=True)
    os.chmod(output, 0o700)
    source_evidence = {}
    counts = {}
    for symbol in symbols:
        kline_paths = _paths(archive_root, symbol, months, "klines")
        trade_paths = _paths(archive_root, symbol, months, "aggTrades")
        bars = tuple(
            row for path in kline_paths for row in read_kline_archive(path, symbol)
        )
        flow = read_agg_trade_archives_chunked(trade_paths, symbol)
        dataset = build_path_dataset(bars, flow, contracts)
        destination = output / f"{symbol}.parquet"
        dataset.to_parquet(destination, compression="zstd", index=False)
        os.chmod(destination, 0o600)
        counts[symbol] = len(dataset)
        source_evidence[symbol] = {
            "klines": [
                {"path": str(path.resolve()), "sha256": sha256_file(path)}
                for path in kline_paths
            ],
            "aggregate_trades": [
                {"path": str(path.resolve()), "sha256": sha256_file(path)}
                for path in trade_paths
            ],
            "rows": len(dataset), "dataset_sha256": sha256_file(destination),
            "minimum_event_timestamp_ms": int(dataset.event_timestamp_ms.min()),
            "maximum_event_timestamp_ms": int(dataset.event_timestamp_ms.max()),
        }
        print(json.dumps({"symbol": symbol, "rows": len(dataset)}), flush=True)
    manifest = {
        "schema_version": "aegis-event-path-quality-c2a-manifest-v1",
        "config": str(config_path.resolve()),
        "config_sha256": sha256_file(config_path),
        "months": months, "symbols": symbols,
        "contracts": [contract.identity for contract in contracts],
        "source_evidence": source_evidence,
        "total_rows": sum(counts.values()),
        "authenticated_requests": 0, "exchange_mutations": 0,
        "C2A_DATASET_READY": len(counts) == len(symbols) and all(counts.values()),
        "C2A_READY_FOR_MODELING": False,
        "C2A_READY_FOR_SHADOW": False,
        "C2A_READY_FOR_LIVE": False,
    }
    manifest_path = output / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    os.chmod(manifest_path, 0o600)
    print(json.dumps({"manifest": str(manifest_path), "total_rows": manifest["total_rows"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
