#!/usr/bin/env python3
"""CLI for the isolated Stage-0 compatibility control."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from scripts.diagnostics.compat_replay.replay import HistoricalReproductionMismatch, run_stage_zero
from scripts.diagnostics.compat_replay.schemas import ReplayConfig


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path(__file__).with_name("replay_v1.yaml"))
    args = parser.parse_args()
    try:
        result = run_stage_zero(ReplayConfig.load(args.config))
    except HistoricalReproductionMismatch as error:
        print(json.dumps({"verdict": "HISTORICAL_REPRODUCTION_MISMATCH", "error": str(error)}))
        return 3
    except Exception as error:
        print(json.dumps({"verdict": "COMPATIBILITY_REPLAY_BLOCKED", "error": str(error)}))
        return 2
    print(json.dumps({"verdict": "HISTORICAL_HARNESS_REPRODUCTION_CONFIRMED", "stage_0": result}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
