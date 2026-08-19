#!/usr/bin/env python3
"""Read-only audit of post-freeze W13-P public collection coverage.

This intentionally reads only timestamps, stream identity, symbols, sides, and
quality flags. It does not load or derive any outcome.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import pyarrow.parquet as pq


FREEZE_US = 1_787_000_371_000_000


def iso(timestamp_us: int | None) -> str | None:
    if timestamp_us is None:
        return None
    return datetime.fromtimestamp(timestamp_us / 1_000_000, tz=timezone.utc).isoformat()


def rows(root: Path, kind: str, columns: tuple[str, ...]):
    for path in sorted((root / kind).rglob("*.parquet")):
        schema = set(pq.read_schema(path).names)
        selected = [column for column in columns if column in schema]
        if not selected:
            continue
        yield from pq.read_table(path, columns=selected).to_pylist()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--pipeline-manifest", type=Path)
    args = parser.parse_args()

    events = [
        row for row in rows(
            args.root, "event", ("exchange_event_timestamp_us", "symbol", "event_type")
        )
        if row.get("exchange_event_timestamp_us", -1) >= FREEZE_US
    ]
    signals = [
        row for row in rows(args.root, "signal", ("signal_timestamp_us", "symbol", "side"))
        if row.get("signal_timestamp_us", -1) >= FREEZE_US
    ]
    quality = [
        row for row in rows(
            args.root, "quality", ("signal_timestamp_us", "symbol", "side", "W13_ELIGIBLE", "max_gap_ms")
        )
        if row.get("signal_timestamp_us", -1) >= FREEZE_US
    ]
    event_times = sorted(row["exchange_event_timestamp_us"] for row in events)
    signal_times = sorted(row["signal_timestamp_us"] for row in signals)
    pipeline = None
    if args.pipeline_manifest:
        pipeline = json.loads(args.pipeline_manifest.read_text(encoding="utf-8"))
        if pipeline.get("outcomes_loaded") is not False or pipeline.get("edge_validation_performed") is not False:
            raise RuntimeError("PIPELINE_MANIFEST_IS_NOT_LABEL_FREE")
    snapshot_count = int(pipeline.get("snapshots", 0)) if pipeline else 0
    candidate_counts = dict(pipeline.get("candidate_counts", {})) if pipeline else {}
    detailed_candidate_counts = dict(
        pipeline.get("candidate_counts_by_strategy_symbol_side_status", {})
    ) if pipeline else {}
    event_rates = dict(pipeline.get("eligible_event_rate_by_strategy_symbol_side", {})) if pipeline else {}
    has_frozen_gap = any("BLOCKED_FROZEN_DECISION_GAP" in key for key in candidate_counts)
    report = {
        "schema": "aegis-strategy-router-phase2-fresh-coverage-v1",
        "freeze_checkpoint": "dcd445cb293d661cc6c184a75cd39df054447ab1",
        "freeze_timestamp": "2026-08-17T20:59:31+00:00",
        "first_event_at": iso(event_times[0] if event_times else None),
        "last_event_at": iso(event_times[-1] if event_times else None),
        "first_signal_at": iso(signal_times[0] if signal_times else None),
        "last_signal_at": iso(signal_times[-1] if signal_times else None),
        "event_rows": len(events),
        "signals": len(signals),
        "eligible_quality_records": sum(bool(row.get("W13_ELIGIBLE")) for row in quality),
        "events_by_stream": dict(sorted(Counter(row.get("event_type", "UNKNOWN") for row in events).items())),
        "events_by_symbol": dict(sorted(Counter(row.get("symbol", "UNKNOWN") for row in events).items())),
        "signals_by_symbol_side": {
            f"{symbol}:{side}": count
            for (symbol, side), count in sorted(Counter((row.get("symbol"), row.get("side")) for row in signals).items())
        },
        "phase1_snapshot_count": snapshot_count,
        "generator_evaluation_counts": candidate_counts,
        "generator_evaluation_counts_by_strategy_symbol_side_status": detailed_candidate_counts,
        "generator_evaluations_per_snapshot": {
            key: value / snapshot_count for key, value in sorted(candidate_counts.items())
        } if snapshot_count else "UNAVAILABLE_NO_FRESH_PHASE1_SNAPSHOTS",
        "eligible_candidate_event_rate": (
            "UNAVAILABLE_NO_FRESH_PHASE1_SNAPSHOTS" if not snapshot_count else
            "UNAVAILABLE_FROZEN_DECISION_GAPS" if has_frozen_gap else event_rates
        ),
        "outcomes_loaded": False,
        "edge_validation_performed": False,
    }
    json.dump(report, sys.stdout, indent=2, sort_keys=True)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
