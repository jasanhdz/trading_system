#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from aegis_range_v1.sweep_reclaim_phase1 import execute_phase1


def main() -> int:
    parser = argparse.ArgumentParser(description="Aegis Range V2 sweep-reclaim Phase 1: first-passage asymmetry")
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    result = execute_phase1(args.repo_root.resolve(), args.output_root.resolve())
    print(json.dumps({key: result[key] for key in ("status", "diagnostics_manifest_sha256")}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
