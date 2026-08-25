#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

from aegis_range_v1.replay_benchmark import parallel_replay_counts
from aegis_range_v1.sweep_reclaim_discovery import verify_authority


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the TRAIN sweep/reclaim replay in parallel by symbol")
    parser.add_argument("--repo-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--workers", type=int, default=min(4, os.cpu_count() or 1))
    args = parser.parse_args()

    repo_root = args.repo_root.resolve()
    output_root = args.output_root.resolve()
    started = time.monotonic()
    authority = verify_authority(repo_root, output_root)
    results = parallel_replay_counts(
        repo_root,
        authority["run_a"],
        workers=args.workers,
        progress=True,
    )
    totals = {
        key: sum(int(result[key]) for result in results)
        for key in ("opportunities", "entries", "paths", "passages")
    }
    print(
        json.dumps(
            {
                "workers": args.workers,
                "elapsed_seconds": time.monotonic() - started,
                "symbols": results,
                "totals": totals,
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
