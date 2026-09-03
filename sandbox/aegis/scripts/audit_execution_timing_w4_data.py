#!/usr/bin/env python3
"""Audit whether existing evidence can support W4 without synthetic fills."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sqlite3
import zipfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from aegis.research.execution_timing_w4 import (
    ExecutionDataCapabilities,
    assess_data_quality,
    timestamp_diagnostics,
)


SYMBOLS = (
    "ADAUSDT", "AVAXUSDT", "BNBUSDT", "BTCUSDT", "DOGEUSDT", "ETHUSDT",
    "LINKUSDT", "LTCUSDT", "SOLUSDT", "SUIUSDT", "XRPUSDT",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sqlite_table(connection: sqlite3.Connection, table: str, timestamp: str) -> dict[str, Any]:
    count, symbols, first, last = connection.execute(
        f"SELECT COUNT(*), COUNT(DISTINCT symbol), MIN({timestamp}), MAX({timestamp}) FROM {table}"
    ).fetchone()
    return {
        "rows": int(count), "symbols": int(symbols),
        "first_timestamp_ms": first, "last_timestamp_ms": last,
    }


def sample_archive(path: Path, limit: int = 10_000) -> dict[str, Any]:
    with zipfile.ZipFile(path) as archive:
        member = archive.namelist()[0]
        with archive.open(member) as raw:
            rows = csv.DictReader(line.decode("utf-8") for line in raw)
            timestamps: list[int] = []
            identities: list[int] = []
            for index, row in enumerate(rows):
                if index >= limit:
                    break
                timestamps.append(int(row["transact_time"]))
                identities.append(int(row["agg_trade_id"]))
    return timestamp_diagnostics(timestamps, identities)


def build_audit(root: Path) -> dict[str, Any]:
    archive_root = root / "data/market_event_fast_track_m1a/raw/futures/um/monthly/aggTrades"
    archives = sorted(archive_root.glob("*/*.zip"))
    by_symbol = Counter(path.parent.name for path in archives)
    months = sorted({path.stem.rsplit("-", 2)[-2] + "-" + path.stem.rsplit("-", 1)[-1] for path in archives})
    archive_samples = [sample_archive(path) for path in archives]

    c2_path = root / "data/market_microstructure_events_c2/c2_archive.db"
    legacy_path = root / "data/long_entry_v3_shadow/public_microstructure.db"
    with sqlite3.connect(c2_path) as connection:
        c2 = {
            "aggregate_trades": sqlite_table(connection, "c2_aggregate_trades", "event_time_ms"),
            "depth_snapshots": sqlite_table(connection, "c2_depth_snapshots", "transaction_time_ms"),
        }
    with sqlite3.connect(legacy_path) as connection:
        legacy_depth = sqlite_table(connection, "depth_snapshots", "transaction_time_ms")

    capabilities = ExecutionDataCapabilities(
        agg_trades=len(archives) > 0,
        best_bid_ask=False,
        depth_l2=False,
        local_receive_timestamp=False,
        decision_timestamp=True,
        fee_schedule=False,
    )
    quality = assess_data_quality(capabilities)
    return {
        "schema_version": "aegis-execution-timing-w4-data-audit-v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "W4_DATA_QUALITY_INSUFFICIENT",
        "historical_sources": {
            "futures_aggtrades": {
                "archive_count": len(archives),
                "symbols": dict(sorted(by_symbol.items())),
                "months": [months[0], months[-1]] if months else [],
                "total_bytes": sum(path.stat().st_size for path in archives),
                "fields": [
                    "agg_trade_id", "price", "quantity", "first_trade_id",
                    "last_trade_id", "transact_time", "is_buyer_maker",
                ],
                "exchange_timestamp_available": True,
                "local_receive_timestamp_available": False,
                "sampled_rows": sum(item["rows"] for item in archive_samples),
                "sample_invalid_timestamps": sum(item["invalid_timestamps"] for item in archive_samples),
                "sample_out_of_order_events": sum(item["out_of_order_events"] for item in archive_samples),
                "sample_duplicate_identities": sum(item["duplicate_identities"] for item in archive_samples),
            },
            "c2_archive": c2,
            "legacy_depth": legacy_depth,
            "book_ticker": {"available": False, "rows": 0},
            "sequenced_l2_book": {"available": False, "rows": 0},
            "order_book_queue_events": {"available": False, "rows": 0},
            "real_execution_fills_with_decision_bbo": {"available": False},
        },
        "current_execution": {
            "policy": "MARKET_NOW",
            "order_type": "MARKET",
            "decision_event": "ORDER_SUBMITTED",
            "durable_intent_identity": True,
            "primary_reference_required": "SYNCHRONIZED_MIDPRICE_AT_INTENT_TIMESTAMP",
            "primary_reference_available_historically": False,
        },
        "capabilities": capabilities.__dict__,
        "quality_gate": quality,
        "simulation_scope": {
            "market_now": "NOT_IDENTIFIABLE",
            "wait_100_250_500ms": "DIRECTIONAL_TRADE_PATH_ONLY_NOT_EXECUTION_COST",
            "aggressive_limit": "NOT_IDENTIFIABLE",
            "passive_limit": "PROHIBITED_NO_QUEUE_EVIDENCE",
            "partial_fills": "NOT_IDENTIFIABLE",
            "microprice": "NOT_IDENTIFIABLE",
            "spread": "NOT_IDENTIFIABLE",
            "fees": "NOT_FROZEN",
        },
        "safety": {
            "authenticated_requests": 0,
            "network_requests": 0,
            "exchange_mutations": 0,
            "production_changes": 0,
        },
    }


def render_markdown(audit: dict[str, Any]) -> str:
    agg = audit["historical_sources"]["futures_aggtrades"]
    blockers = audit["quality_gate"]["blockers"]
    return f"""# Aegis W4 Execution Timing Data Audit

## Status

`W4_DATA_QUALITY_INSUFFICIENT`

The current evidence cannot identify execution cost. W4A and W4B must not run
until synchronized top-of-book and intent-time evidence exists.

## Current Aegis Execution

- Entry policy: `MARKET_NOW`.
- Venue call: Binance USD-M `MARKET` with `newOrderRespType=RESULT`.
- The durable lifecycle provides deterministic intent/client-order identity.
- The `ORDER_SUBMITTED` event precedes `marketOpen`, but historical telemetry
  does not contain a synchronized bid/ask snapshot at that timestamp.
- The preregistered primary benchmark is midprice at the intent timestamp.

## Historical Evidence

- Futures aggTrade archives: {agg['archive_count']} files, {len(agg['symbols'])}
  symbols, {agg['months'][0]} through {agg['months'][1]}.
- Sampled archive rows: {agg['sampled_rows']:,}; invalid timestamps:
  {agg['sample_invalid_timestamps']}; out-of-order rows:
  {agg['sample_out_of_order_events']}; duplicate aggregate-trade IDs:
  {agg['sample_duplicate_identities']}.
- aggTrade provides exchange transaction time, price, quantity and aggressor
  side. It does not provide bid, ask, queue state or local receive time.
- Historical `bookTicker`: 0 rows.
- Sequenced L2 history: 0 rows.
- C2 depth history: {audit['historical_sources']['c2_archive']['depth_snapshots']['rows']} rows.
- Legacy depth: {audit['historical_sources']['legacy_depth']['rows']} isolated
  snapshots across {audit['historical_sources']['legacy_depth']['symbols']} symbols;
  this is not a time series.

## Blocking Conditions

""" + "".join(f"- `{item}`\n" for item in blockers) + """

Without BBO, `MARKET_NOW` implementation shortfall cannot be separated into
spread and slippage. Without receive timestamps, 100-500 ms latency cannot be
simulated. Without sequenced L2 and queue evidence, passive fills, fill
probability and partial fills cannot be reconstructed honestly.

aggTrade-only future price movement could be modeled, but that would answer a
directional short-horizon question and risk turning W4 into a hidden second
brain. It is therefore not used as a substitute.

## Decision

- `W4_DATA_QUALITY_SUFFICIENT = FALSE`
- W4A not executed.
- W4B not executed.
- FINAL_HOLDOUT_W4 remains `SEALED` and unpopulated.
- No synthetic BBO, spread, queue or fills were created.
- Authenticated requests: 0.
- Exchange mutations: 0.
"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    root = args.root.resolve()
    audit = build_audit(root)
    private = root / "data/execution_timing_w4"
    report = root / "reports/governance/aegis_prospective_validation/live/execution_timing_w4"
    private.mkdir(parents=True, exist_ok=True)
    report.mkdir(parents=True, exist_ok=True)
    audit_path = private / "aegis_execution_timing_w4_data_audit.json"
    report_path = report / "aegis_execution_timing_w4_data_audit.md"
    audit_path.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")
    report_path.write_text(render_markdown(audit))
    audit_path.chmod(0o600)
    print(json.dumps({
        "status": audit["status"],
        "audit": str(audit_path),
        "report": str(report_path),
        "audit_sha256": sha256(audit_path),
    }, indent=2))


if __name__ == "__main__":
    main()
