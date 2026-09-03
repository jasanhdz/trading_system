#!/usr/bin/env python3
"""Run the preregistered E4A experiment twice and verify determinism."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from scripts.diagnostics.e4a_mechanical_exit.experiment import (
    execute_attempt,
    finalize_attempts,
    write_run_manifest,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repository", type=Path, default=Path(__file__).resolve().parents[3])
    parser.add_argument(
        "--preregistration", type=Path,
        default=Path("reports/experiments/e4a_mechanical_exit/preregistration.yaml"),
    )
    parser.add_argument(
        "--output", type=Path,
        default=Path("reports/experiments/e4a_mechanical_exit"),
    )
    args = parser.parse_args()
    repository = args.repository.resolve()
    preregistration = args.preregistration if args.preregistration.is_absolute() else repository / args.preregistration
    output = args.output if args.output.is_absolute() else repository / args.output
    for attempt in (1, 2):
        root = output / f"attempt_{attempt}"
        result = execute_attempt(repository, preregistration, root)
        write_run_manifest(
            root / "run_manifest.json", attempt=attempt, repository=repository,
            preregistration_path=preregistration, result=result,
        )
    final = finalize_attempts(repository, preregistration, output)
    print(json.dumps({
        "technical_status": final["summary"]["technical_status"],
        "scientific_classification": final["summary"]["scientific_classification"],
        "next_decision": final["summary"]["next_decision"],
        "aggregate_hash": final["determinism"]["aggregate_hash"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
