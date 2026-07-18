#!/usr/bin/env python3
"""Future Phase-F command; requires a real CANDIDATE bundle."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from aegis.freeze import BundleLifecycleState
from aegis.training.benchmark import build_paired_benchmark, write_benchmark_report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--current-decisions", type=Path, required=True)
    parser.add_argument("--candidate-bundle-id", required=True)
    parser.add_argument("--candidate-state", choices=["CANDIDATE"], required=True)
    parser.add_argument("--gen2-decisions", type=Path, default=Path("/home/jasan/Develop/aegis_gen2/forward/forward_decisions.jsonl"))
    parser.add_argument("--output", type=Path, default=ROOT / "reports" / "parity_benchmark" / "paired_report.json")
    args = parser.parse_args()
    with args.current_decisions.open("r", encoding="utf-8") as handle:
        rows = tuple(json.loads(line) for line in handle if line.strip())
    report = build_paired_benchmark(
        current_rows=rows, gen2_decisions_path=args.gen2_decisions,
        bundle_id=args.candidate_bundle_id, bundle_state=BundleLifecycleState(args.candidate_state),
    )
    write_benchmark_report(report, args.output)
    print(f"matched_rows={report.matched_rows}")
    print(f"report_hash={report.report_hash}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

