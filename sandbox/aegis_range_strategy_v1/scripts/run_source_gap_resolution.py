#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from aegis_range_v1.source_gap import (
    audit_status,
    build_gap_resolved_derived,
    download_and_audit,
    write_audit,
    write_source_manifest,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Resolve only the approved Range R2 mark-price source gaps")
    subparsers = parser.add_subparsers(dest="command", required=True)
    audit = subparsers.add_parser("audit")
    audit.add_argument("--repo-root", type=Path, required=True)
    audit.add_argument("--raw-root", type=Path, required=True)
    audit.add_argument("--output", type=Path, required=True)
    manifest = subparsers.add_parser("manifest")
    manifest.add_argument("--repo-root", type=Path, required=True)
    manifest.add_argument("--audit", type=Path, required=True)
    manifest.add_argument("--output", type=Path, required=True)
    build = subparsers.add_parser("build")
    build.add_argument("--repo-root", type=Path, required=True)
    build.add_argument("--source-manifest", type=Path, required=True)
    build.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()

    if args.command == "audit":
        audits = download_and_audit(args.repo_root.resolve(), args.raw_root.resolve())
        write_audit(args.output.resolve(), audits)
        status = audit_status(audits)
        print(json.dumps({"daily_files": len(audits), "overlap_rows_compared": sum(item.overlap_rows_compared for item in audits), "mismatches": sum(item.mismatches for item in audits), "recovered_minutes": sum(item.daily_recovered_minutes for item in audits), "remaining_missing_minutes": sum(item.remaining_missing_minutes for item in audits), "status": status}, sort_keys=True))
        if status != "DAILY_VALID_FOR_CONTRACTUAL_GAP_FILL":
            return 2
    elif args.command == "manifest":
        digest = write_source_manifest(args.repo_root.resolve(), args.audit.resolve(), args.output.resolve())
        print(json.dumps({"source_manifest_sha256": digest}, sort_keys=True))
    else:
        result = build_gap_resolved_derived(args.repo_root.resolve(), args.source_manifest.resolve(), args.output_root.resolve())
        print(json.dumps({key: result[key] for key in ("funding_events_total", "funding_events_mapped", "funding_events_missing_mark_price", "mark_price_missing_minutes", "logical_sha256", "status")}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
