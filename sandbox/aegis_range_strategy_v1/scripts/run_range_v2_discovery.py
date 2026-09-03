#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from aegis_range_v1.range_v2_discovery import execute_discovery


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the TRAIN-only Aegis Range V2 hypothesis-generation diagnostic")
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    result = execute_discovery(args.repo_root.resolve(), args.output_root.resolve())
    print(json.dumps({key: result[key] for key in ("status", "diagnostics_manifest_sha256")}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
