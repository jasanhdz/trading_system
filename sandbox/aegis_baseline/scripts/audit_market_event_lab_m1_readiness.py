#!/usr/bin/env python3
"""Audit M1 evidence coverage without network or exchange access."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from dataclasses import asdict
from pathlib import Path

from aegis.research.market_event_lab_m1 import (
    AppendOnlyTrialLedger,
    TrialRecord,
    assess_database_readiness,
    utc_now_string,
)
from aegis.utils import sha256_file


def _commit(root: Path) -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--database",
        type=Path,
        default=Path("data/long_entry_v3_shadow/public_microstructure.db"),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("data/market_event_lab_m1/readiness"),
    )
    parser.add_argument("--trial-id", default=None)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    database = args.database if args.database.is_absolute() else root / args.database
    output_root = args.output_root if args.output_root.is_absolute() else root / args.output_root
    output_root.mkdir(parents=True, exist_ok=True)
    os.chmod(output_root, 0o700)

    report = assess_database_readiness(database)
    report_path = output_root / "market_event_lab_m1_readiness.json"
    report_path.write_text(json.dumps(asdict(report), indent=2, sort_keys=True) + "\n")
    os.chmod(report_path, 0o600)

    preregistration = root / "config/experiments/aegis_market_event_lab_m1.yaml"
    trial_id = args.trial_id or f"M1-READINESS-{utc_now_string()}"
    ledger = AppendOnlyTrialLedger(output_root / "trial_ledger.jsonl")
    ledger.append(
        TrialRecord(
            trial_id=trial_id,
            created_at_utc=utc_now_string(),
            preregistration_sha256=sha256_file(preregistration),
            configuration_sha256=sha256_file(preregistration),
            code_commit=_commit(root),
            dataset_sha256={"public_microstructure_db": sha256_file(database)},
            status="READY" if report.M1_READY_FOR_EXPERIMENTS else "BLOCKED_DATA_MATURITY",
            result_summary={
                "ready_families": report.ready_families,
                "M1_READY_FOR_EXPERIMENTS": report.M1_READY_FOR_EXPERIMENTS,
                "report_hash": report.report_hash,
                "network_calls": 0,
                "exchange_mutations": 0,
            },
        )
    )
    print(json.dumps({"report": str(report_path), **asdict(report)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
