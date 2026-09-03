#!/usr/bin/env python3
"""Produce a non-secret C2 coverage and family-readiness report."""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
from pathlib import Path

from aegis.research.microstructure_events_c2 import C2Archive, archive_coverage
from aegis.research.market_event_lab_m1 import assess_database_readiness


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--c2-archive", type=Path, default=Path("data/market_microstructure_events_c2/c2_archive.db"))
    parser.add_argument("--legacy-database", type=Path, default=Path("data/long_entry_v3_shadow/public_microstructure.db"))
    parser.add_argument("--output", type=Path, default=Path("data/market_microstructure_events_c2/readiness.json"))
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    archive = args.c2_archive if args.c2_archive.is_absolute() else root / args.c2_archive
    legacy = args.legacy_database if args.legacy_database.is_absolute() else root / args.legacy_database
    output = args.output if args.output.is_absolute() else root / args.output
    if not archive.exists():
        holder = C2Archive(archive)
        holder.close()
    c2 = archive_coverage(archive)
    m1 = assess_database_readiness(legacy, minimum_days=60.0)
    sufficient = {
        name: item["symbols"] == 11 and item["span_days"] >= 60.0
        for name, item in c2.items()
    }
    family_ready = {
        "OI_CONFIRMED_BREAKOUT": sufficient["aggregate_trades"] and sufficient["open_interest"],
        "LIQUIDATION_ABSORPTION_REVERSAL": all(
            sufficient[name] for name in ("aggregate_trades", "liquidation_events", "depth_snapshots")
        ),
        "LIQUIDATION_CONTINUATION": all(
            sufficient[name] for name in ("aggregate_trades", "liquidation_events", "open_interest", "depth_snapshots")
        ),
        "DEPTH_ABSORPTION_REVERSAL": sufficient["aggregate_trades"] and sufficient["depth_snapshots"],
        "FLOW_IMPULSE_CONTINUATION": sufficient["aggregate_trades"],
        "BTC_ALT_LEAD_LAG": sufficient["aggregate_trades"],
    }
    combined = {
        "schema_version": "aegis-market-microstructure-c2-readiness-v1",
        "c2_archive": c2,
        "legacy_sources": {
            name: {
                "rows": item.row_count, "symbols": item.symbols,
                "span_days": item.span_days,
            }
            for name, item in m1.source_coverage.items()
        },
        "family_state": {
            **{
                name: "READY_FOR_EVENT_REPLAY" if ready else "BLOCKED_COLLECTION_LT_60_DAYS"
                for name, ready in family_ready.items()
            },
            "SPOT_FUTURES_DISLOCATION_CONTROL": (
                "SOURCE_READY_EXPERIMENT_PREVIOUSLY_REFUTED"
                if m1.family_readiness["SPOT_FUTURES_DISLOCATION"].ready
                else "SOURCE_BLOCKED"
            ),
        },
        "C2_COLLECTOR_READY": True,
        "C2_DATA_READY": all(family_ready.values()),
        "C2_EVENT_FAMILY_FOUND": False,
        "C2_READY_FOR_MODELING": False,
        "C2_READY_FOR_SHADOW": False,
        "C2_READY_FOR_LIVE": False,
        "exchange_mutations": 0,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(combined, indent=2, sort_keys=True) + "\n")
    os.chmod(output, 0o600)
    print(json.dumps({"output": str(output), "C2_DATA_READY": all(family_ready.values())}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
