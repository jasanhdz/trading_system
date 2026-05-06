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
from aegis_alpha.turbo.snapshot_utils import load_turbo_snapshot_status, turbo_snapshot_path  # noqa: E402


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", default=DEFAULT_TURBO_CONFIG.symbol)
    parser.add_argument("--lookback-days", type=int, default=7)
    parser.add_argument("--json-out")
    args = parser.parse_args()

    path = turbo_snapshot_path(int(args.lookback_days))
    status = load_turbo_snapshot_status(path, include_sample_count=True)
    payload = {
        "symbol": args.symbol,
        "path": str(path),
        **status,
        "queried_at": datetime.now(timezone.utc).isoformat(),
    }
    if args.json_out:
        _write_json(Path(args.json_out), payload)
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
