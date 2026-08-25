#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from aegis_range_v1.train_backtest import execute_train_run


def main() -> int:
    parser = argparse.ArgumentParser(description="Execute the authorized Aegis Range R2 TRAIN backtest")
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--reuse-regime-root", type=Path)
    args = parser.parse_args()
    result = execute_train_run(
        args.repo_root.resolve(),
        args.output_root.resolve(),
        args.workers,
        None if args.reuse_regime_root is None else args.reuse_regime_root.resolve(),
    )
    print(
        json.dumps(
            {
                "candidate_count": result["candidate_count"],
                "run_manifest_sha256": result["run_manifest_sha256"],
                "status": result["status"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
