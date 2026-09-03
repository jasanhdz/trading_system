#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from aegis_range_v1.sweep_reclaim_discovery import execute_discovery, reproducible_sample


def main() -> int:
    parser = argparse.ArgumentParser(description="Run TRAIN-only Aegis Range V2 sweep/reclaim discovery")
    parser.add_argument("--repo-root", type=Path)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--sample", action="store_true")
    args = parser.parse_args()
    if args.sample:
        result = reproducible_sample()
        print(json.dumps({"sha256": result["sha256"], "opportunities": len(result["opportunities"])}, sort_keys=True))
        return 0
    if args.repo_root is None or args.output_root is None:
        parser.error("--repo-root and --output-root are required unless --sample is used")
    result = execute_discovery(args.repo_root.resolve(), args.output_root.resolve())
    print(json.dumps({key: result[key] for key in ("status", "diagnostics_manifest_sha256")}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
