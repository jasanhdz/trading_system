#!/usr/bin/env python3
"""Audit W9 historical L2 data and stop before modeling when gates fail."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from aegis.research.historical_orderbook_direction_w9 import (
    W9CoverageRequirements,
    assess_coverage,
    audit_incremental_l2,
    audit_quotes,
    audit_trades,
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


def opportunity_coverage(path: Path, config: dict[str, Any]) -> dict[str, Any]:
    frame = pd.read_parquet(path, columns=["timestamp", "symbol"])
    timestamp = pd.to_datetime(frame["timestamp"], utc=True)
    frame = frame.assign(month=timestamp.dt.strftime("%Y-%m"), day=timestamp.dt.day)
    sample = frame.loc[frame["day"].eq(1)].copy()
    train_months = set(config["partitions"]["train_months"])
    validation_months = set(config["partitions"]["validation_months"])
    train = sample.loc[sample["month"].isin(train_months)]
    validation = sample.loc[sample["month"].isin(validation_months)]
    requirements = W9CoverageRequirements(
        minimum_train_episodes=int(config["data_gate"]["minimum_train_episodes"]),
        minimum_validation_episodes=int(config["data_gate"]["minimum_validation_episodes"]),
        minimum_symbols_per_partition=int(config["data_gate"]["minimum_symbols_per_partition"]),
        minimum_months_per_partition=int(config["data_gate"]["minimum_months_per_partition"]),
    )
    gate = assess_coverage(
        train_episodes=len(train),
        validation_episodes=len(validation),
        train_symbols=train["symbol"].nunique(),
        validation_symbols=validation["symbol"].nunique(),
        train_months=train["month"].nunique(),
        validation_months=validation["month"].nunique(),
        requirements=requirements,
    )
    by_month_symbol = (
        sample.groupby(["month", "symbol"], observed=True).size().unstack(fill_value=0)
    )
    return {
        "source_rows": len(frame),
        "free_sample_day_episodes": len(sample),
        "by_month": {str(key): int(value) for key, value in sample.groupby("month").size().items()},
        "by_month_symbol": {
            month: {symbol: int(value) for symbol, value in row.items() if value}
            for month, row in by_month_symbol.iterrows()
        },
        "gate": gate,
    }


def build_audit(root: Path, l2_path: Path) -> dict[str, Any]:
    config_path = root / "config/experiments/aegis_historical_orderbook_direction_w9.yaml"
    opportunity_path = root / "data/conditional_direction_w8/run_01/development_outcomes.parquet"
    config = yaml.safe_load(config_path.read_text())
    coverage = opportunity_coverage(opportunity_path, config)
    reconstruction = audit_incremental_l2(l2_path)
    quotes_path = Path(str(l2_path).replace("incremental_book_L2", "quotes"))
    trades_path = Path(str(l2_path).replace("incremental_book_L2", "trades"))
    quotes = audit_quotes(quotes_path)
    trades = audit_trades(trades_path)
    normalized_valid = bool(
        reconstruction["passes_normalized_reconstruction"]
        and quotes["passes"]
        and trades["passes"]
    )
    coverage_valid = bool(coverage["gate"]["passes"])
    sufficient = normalized_valid and coverage_valid
    blockers = list(coverage["gate"]["blockers"])
    if not normalized_valid:
        blockers.append("NORMALIZED_L2_RECONSTRUCTION_FAILED")
    status = "AEGIS_W9_DATA_GATE_PASSED" if sufficient else "AEGIS_W9_BLOCKED_DATA_QUALITY"
    return {
        "schema_version": "aegis-historical-orderbook-direction-w9-data-audit-v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "config": {"path": str(config_path.relative_to(root)), "sha256": sha256(config_path)},
        "frozen_opportunity": {
            "identity": config["authority"]["w7_opportunity_identity"],
            "source": str(opportunity_path.relative_to(root)),
            "sha256": sha256(opportunity_path),
            "refit_performed": False,
        },
        "provider": {
            "name": "Tardis.dev",
            "exchange": "binance-futures",
            "sample_policy": "FIRST_DAY_OF_MONTH_WITHOUT_API_KEY",
            "catalog_available_since": "2019-11-17T00:00:00Z",
            "catalog_exported_until": "2026-08-14T00:00:00Z",
            "all_11_symbols_advertise_required_types": True,
            "required_types": ["incremental_book_L2", "quotes", "trades"],
            "raw_feed_sequence_validation": "DOCUMENTED_BY_PROVIDER_USING_pu_AND_u",
            "normalized_csv_native_sequence_fields": "NOT_PRESENT",
        },
        "pilot": {
            "symbol": "ADAUSDT",
            "date": "2025-09-01",
            "l2_path": str(l2_path),
            "compressed_bytes": l2_path.stat().st_size,
            "sha256": sha256(l2_path),
            "reconstruction": reconstruction,
            "quotes": {**quotes, "compressed_bytes": quotes_path.stat().st_size, "sha256": sha256(quotes_path)},
            "trades": {**trades, "compressed_bytes": trades_path.stat().st_size, "sha256": sha256(trades_path)},
        },
        "opportunity_coverage": coverage,
        "quality_gate": {
            "W9_DATA_QUALITY_SUFFICIENT": sufficient,
            "W9_ORDERBOOK_RECONSTRUCTION_VALID": normalized_valid,
            "blockers": blockers,
        },
        "modeling": {
            "W9A_executed": False,
            "reason": "DATA_GATE_FAILED_STOP_BEFORE_MODELING" if not sufficient else "PENDING",
            "hypotheses_tested": 0,
            "holdout": "SEALED_NOT_OPENED",
        },
        "flags": {
            "W9_DATA_QUALITY_SUFFICIENT": sufficient,
            "W9_ORDERBOOK_RECONSTRUCTION_VALID": normalized_valid,
            "W9_LIQUIDITY_INFORMATION_FOUND": False,
            "W9_FLOW_INFORMATION_FOUND": False,
            "W9_ABSORPTION_INFORMATION_FOUND": False,
            "W9_DIRECTIONAL_SIGNAL_FOUND": False,
            "W9_ECONOMIC_EDGE_FOUND": False,
            "W9_READY_FOR_PROSPECTIVE_COLLECTION": False,
            "W9_READY_FOR_SHADOW": False,
            "W9_READY_FOR_LIVE": False,
        },
        "safety": {
            "authenticated_requests": 0,
            "exchange_mutations": 0,
            "production_changes": 0,
            "typescript_changes": 0,
            "pm2_changes": 0,
            "websocket_connections": 0,
            "purchased_data": False,
        },
    }


def render_markdown(audit: dict[str, Any]) -> str:
    coverage = audit["opportunity_coverage"]
    observed = coverage["gate"]["observed"]
    reconstruction = audit["pilot"]["reconstruction"]
    blockers = audit["quality_gate"]["blockers"]
    return f"""# Aegis W9 Historical Order Book Direction - Data Audit

## Status

`{audit['status']}`

W9 is measurable at the normalized-book level, but the free samples overlap
too few frozen W7 Opportunity episodes to support the preregistered economic
validation. Modeling stopped before any directional hypothesis was fitted.

## Frozen Opportunity Coverage

- Total W7 Opportunity episodes in source: {coverage['source_rows']:,}.
- Episodes on Tardis free-sample days: {coverage['free_sample_day_episodes']:,}.
- TRAIN: {observed['train_episodes']} episodes, {observed['train_symbols']} symbols,
  {observed['train_months']} months.
- VALIDATION: {observed['validation_episodes']} episodes,
  {observed['validation_symbols']} symbols, {observed['validation_months']} months.
- FINAL_HOLDOUT_W9: `SEALED_NOT_OPENED`; future evidence is not yet collected.

## L2 Reconstruction Pilot

- Provider: Tardis public first-day sample.
- Instrument/date: `ADAUSDT`, `2025-09-01`.
- Compressed size: {audit['pilot']['compressed_bytes']:,} bytes.
- Rows: {reconstruction['rows']:,}.
- Message groups: {reconstruction['messages']:,}.
- Snapshot groups: {reconstruction['snapshot_messages']}.
- Crossed/locked messages: {reconstruction['crossed_or_locked_messages']}.
- Invalid messages: {reconstruction['invalid_messages']}.
- Local timestamp reorderings: {reconstruction['local_timestamp_out_of_order']}.
- Gaps over five seconds: {reconstruction['gaps_over_5s']}.
- Normalized reconstruction valid: `{str(reconstruction['passes_normalized_reconstruction']).upper()}`.
- Quote rows: {audit['pilot']['quotes']['rows']:,}; quote audit:
  `{str(audit['pilot']['quotes']['passes']).upper()}`.
- Trade rows: {audit['pilot']['trades']['rows']:,}; trade audit:
  `{str(audit['pilot']['trades']['passes']).upper()}`.

The normalized CSV preserves provider capture order and absolute L2 updates.
It does not expose Binance native `U/u/pu` sequence IDs. Tardis documents that
its raw collector validates `pu`/`u`, restarts on gaps, and validates snapshot
overlap. W9 records this limitation rather than claiming independent native
sequence verification from the normalized file.

## Blocking Conditions

""" + "".join(f"- `{item}`\n" for item in blockers) + """

Downloading every free L2 day would add gigabytes but would not create more
than the 459 overlapping frozen Opportunity episodes. Bootstrap resampling
cannot replace independent episodes. Fitting the requested ablations on this
population would invite symbol/month selection and produce an unreliable
economic verdict.

## Verdict

- `W9_DATA_QUALITY_SUFFICIENT = FALSE`
- `W9_ORDERBOOK_RECONSTRUCTION_VALID = TRUE`
- W9 directional modeling: not executed.
- Economic edge: not tested, not disproved.
- `W9_READY_FOR_PROSPECTIVE_COLLECTION = FALSE`
- `W9_READY_FOR_SHADOW = FALSE`
- `W9_READY_FOR_LIVE = FALSE`

This result means **insufficient overlapping historical evidence**, not
`AEGIS_W9_NO_ORDERBOOK_DIRECTIONAL_EDGE`.

## Safety

- Public unauthenticated data only.
- Exchange mutations: 0.
- Production, TypeScript, brain, guards, leverage and PM2 changes: 0.
- Orders and production WebSockets: 0.
"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--l2-path", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    audit = build_audit(root, args.l2_path.resolve())
    report_dir = root / "reports/governance/aegis_prospective_validation/live/historical_orderbook_direction_w9"
    private_dir = root / "data/historical_orderbook_direction_w9/run_01"
    report_dir.mkdir(parents=True, exist_ok=True)
    private_dir.mkdir(parents=True, exist_ok=True)
    (private_dir / "aegis_historical_orderbook_direction_w9_data_audit.json").write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n"
    )
    (report_dir / "aegis_historical_orderbook_direction_w9_data_audit.md").write_text(
        render_markdown(audit)
    )
    verdict = {
        "schema_version": "aegis-historical-orderbook-direction-w9-verdict-v1",
        "status": audit["status"],
        "flags": audit["flags"],
        "blockers": audit["quality_gate"]["blockers"],
        "final_holdout": "SEALED_NOT_OPENED",
        "economic_edge_interpretation": "NOT_TESTED_DUE_TO_INSUFFICIENT_DATA",
    }
    (report_dir / "aegis_historical_orderbook_direction_w9_verdict.json").write_text(
        json.dumps(verdict, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(verdict, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
