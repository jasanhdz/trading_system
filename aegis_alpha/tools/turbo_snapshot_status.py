#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from aegis_alpha.turbo.config import DEFAULT_TURBO_CONFIG  # noqa: E402
from aegis_alpha.turbo.snapshot_utils import load_turbo_snapshot_status, normalize_turbo_symbol, turbo_snapshot_path, turbo_symbol_data_dir  # noqa: E402


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", default=DEFAULT_TURBO_CONFIG.symbol)
    parser.add_argument("--symbols", help="Comma-separated symbols, e.g. ETHUSDT,BTCUSDT")
    parser.add_argument("--all-configured", action="store_true")
    parser.add_argument("--lookback-days", type=int, default=7)
    parser.add_argument("--json-out")
    args = parser.parse_args()

    if args.all_configured:
        symbol_values = {normalize_turbo_symbol(DEFAULT_TURBO_CONFIG.symbol)}
        root = turbo_symbol_data_dir(DEFAULT_TURBO_CONFIG.symbol).parent
        if root.exists():
            symbol_values.update(child.name for child in root.iterdir() if child.is_dir())
        symbols = sorted(symbol_values)
    elif args.symbols:
        symbols = [normalize_turbo_symbol(value) for value in args.symbols.split(",") if value.strip()]
    else:
        symbols = [normalize_turbo_symbol(args.symbol)]

    by_symbol: dict[str, Any] = {}
    for symbol in symbols:
        path = turbo_snapshot_path(int(args.lookback_days), symbol)
        status = load_turbo_snapshot_status(path, include_sample_count=True)
        by_symbol[symbol] = {
            "symbol": symbol,
            "path": str(path),
            **status,
            "queried_at": datetime.now(timezone.utc).isoformat(),
        }
    payload = by_symbol[symbols[0]] if len(symbols) == 1 else {
        "lookback_days": int(args.lookback_days),
        "queried_at": datetime.now(timezone.utc).isoformat(),
        "symbols": by_symbol,
    }
    if args.json_out:
        _write_json(Path(args.json_out), payload)
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
