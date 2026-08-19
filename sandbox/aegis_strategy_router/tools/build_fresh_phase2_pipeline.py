#!/usr/bin/env python3
"""Build label-free Phase 1 snapshots and Phase 2 audits from local public data."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pyarrow.parquet as pq

SANDBOX = Path(__file__).resolve().parents[1]
REPOSITORY = SANDBOX.parents[1]
sys.path.insert(0, str(SANDBOX / "src"))
sys.path.insert(0, str(REPOSITORY / "src"))

from aegis_strategy_router.domain.types import Side
from aegis_strategy_router.replay.fresh_pipeline import (
    FreshSignal,
    FreshSnapshotCandidatePipeline,
    ParquetMinuteCandleSource,
    persist_pipeline_result,
)


FREEZE_US = 1_787_000_371_000_000


def load_signals(root: Path) -> tuple[FreshSignal, ...]:
    eligible = set()
    for path in sorted((root / "quality").rglob("*.parquet")):
        for row in pq.read_table(path, columns=["signal_id", "signal_timestamp_us", "W13_ELIGIBLE"]).to_pylist():
            if row["W13_ELIGIBLE"] and row["signal_timestamp_us"] >= FREEZE_US:
                eligible.add(str(row["signal_id"]))
    signals = {}
    columns = ["signal_id", "signal_timestamp_us", "symbol", "side", "reference_mid"]
    for path in sorted((root / "signal").rglob("*.parquet")):
        for row in pq.read_table(path, columns=columns).to_pylist():
            signal_id = str(row["signal_id"])
            if signal_id not in eligible:
                continue
            signals[signal_id] = FreshSignal(
                signal_id=signal_id,
                timestamp=datetime.fromtimestamp(row["signal_timestamp_us"] / 1e6, tz=timezone.utc),
                symbol=str(row["symbol"]),
                side=Side(str(row["side"])),
                reference_price=float(row["reference_mid"]),
            )
    return tuple(signals[key] for key in sorted(signals))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--signal-root", type=Path, required=True)
    parser.add_argument("--candle-root", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    signals = load_signals(args.signal_root)
    result = FreshSnapshotCandidatePipeline().run(signals, ParquetMinuteCandleSource(args.candle_root))
    persist_pipeline_result(result, args.output)
    print(json.dumps(result.manifest(), indent=2, sort_keys=True))
    return 0 if result.snapshots else 2


if __name__ == "__main__":
    raise SystemExit(main())
