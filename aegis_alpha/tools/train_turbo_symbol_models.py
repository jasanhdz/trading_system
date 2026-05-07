#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from aegis_alpha.turbo.config import DEFAULT_TURBO_CONFIG  # noqa: E402
from aegis_alpha.turbo.evaluate_recent_models import evaluate_recent_models  # noqa: E402
from aegis_alpha.turbo.snapshot_utils import normalize_turbo_symbol  # noqa: E402
from aegis_alpha.turbo.train_recent_edge import train_recent_edge_models  # noqa: E402


def _parse_windows(raw: str) -> tuple[int, ...]:
    values: list[int] = []
    for part in raw.split(","):
        text = part.strip().lower().removesuffix("d")
        if text:
            values.append(int(text))
    return tuple(values or list(DEFAULT_TURBO_CONFIG.lookback_days))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", required=True)
    parser.add_argument("--windows", default="7d,14d,30d")
    args = parser.parse_args()

    symbol = normalize_turbo_symbol(args.symbol)
    windows = _parse_windows(args.windows)
    train_report = train_recent_edge_models(symbol, windows)
    eval_report = evaluate_recent_models(symbol)
    print(json.dumps({"symbol": symbol, "windows": windows, "train_report": train_report, "eval_report": eval_report}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
