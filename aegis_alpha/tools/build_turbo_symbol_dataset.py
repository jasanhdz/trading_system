#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from aegis_alpha.turbo.config import DEFAULT_TURBO_CONFIG  # noqa: E402
from aegis_alpha.turbo.recent_dataset import build_recent_dataset  # noqa: E402
from aegis_alpha.turbo.snapshot_utils import normalize_turbo_symbol  # noqa: E402


def _parse_windows(raw: str) -> list[int]:
    values: list[int] = []
    for part in raw.split(","):
        text = part.strip().lower().removesuffix("d")
        if text:
            values.append(int(text))
    return values or list(DEFAULT_TURBO_CONFIG.lookback_days)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", required=True)
    parser.add_argument("--windows", default="7d,14d,30d")
    args = parser.parse_args()

    symbol = normalize_turbo_symbol(args.symbol)
    reports = [build_recent_dataset(symbol, days, save=True)["report"] for days in _parse_windows(args.windows)]
    print(json.dumps({"symbol": symbol, "reports": reports}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
