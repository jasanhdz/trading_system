#!/usr/bin/env python3
"""Audit local M1A public archives and manifests without network access."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from aegis.config import CANONICAL_SYMBOLS
from aegis.research.binance_public_archive import ArchiveRequest, month_range
from aegis.utils import sha256_file


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start-month", default="2024-01")
    parser.add_argument("--end-month", default="2026-07")
    parser.add_argument(
        "--archive-root", type=Path, default=Path("data/market_event_fast_track_m1a/raw")
    )
    parser.add_argument(
        "--manifest", type=Path,
        default=Path("data/market_event_fast_track_m1a/archive_manifest.jsonl"),
    )
    parser.add_argument(
        "--output", type=Path,
        default=Path("data/market_event_fast_track_m1a/archive_coverage_report.json"),
    )
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    archive_root = args.archive_root if args.archive_root.is_absolute() else root / args.archive_root
    manifest_path = args.manifest if args.manifest.is_absolute() else root / args.manifest
    output = args.output if args.output.is_absolute() else root / args.output
    manifest = {}
    if manifest_path.exists():
        for line in manifest_path.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            if row["url"] in manifest:
                raise SystemExit("AEGIS_M1A_DUPLICATE_MANIFEST_URL")
            manifest[row["url"]] = row
    expected = []
    for month in month_range(args.start_month, args.end_month):
        for symbol in CANONICAL_SYMBOLS:
            expected.extend(
                (
                    ArchiveRequest("futures/um", "klines", symbol, month, "1m"),
                    ArchiveRequest("spot", "klines", symbol, month, "1m"),
                )
            )
    present = []
    absent = []
    conflicts = []
    by_symbol = {symbol: {"expected": 0, "present": 0} for symbol in CANONICAL_SYMBOLS}
    for request in expected:
        relative = Path(request.market) / "monthly" / request.data_type / request.symbol / "1m" / request.filename
        path = archive_root / relative
        by_symbol[request.symbol]["expected"] += 1
        if not path.is_file():
            absent.append(request.url)
            continue
        digest = sha256_file(path)
        manifest_row = manifest.get(request.url)
        if manifest_row is None or digest != manifest_row.get("actual_sha256") or digest != manifest_row.get("expected_sha256"):
            conflicts.append(request.url)
            continue
        present.append(request.url)
        by_symbol[request.symbol]["present"] += 1
    report = {
        "schema_version": "aegis-m1a-archive-coverage-report-v1",
        "start_month": args.start_month,
        "end_month": args.end_month,
        "expected_archives": len(expected),
        "present_verified_archives": len(present),
        "absent_archives": absent,
        "manifest_or_hash_conflicts": conflicts,
        "coverage_fraction": len(present) / len(expected),
        "by_symbol": by_symbol,
        "M1A_KLINE_HISTORY_READY": not absent and not conflicts,
        "authenticated_requests": 0,
        "network_calls": 0,
        "exchange_mutations": 0,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    os.chmod(output, 0o600)
    print(json.dumps({key: report[key] for key in ("expected_archives", "present_verified_archives", "coverage_fraction", "M1A_KLINE_HISTORY_READY")}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
