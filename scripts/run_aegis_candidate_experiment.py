#!/usr/bin/env python3
"""Run the preregistered Aegis candidate experiment on local data only."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from aegis.training.experiment import run_experiment, write_experiment_result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "config" / "candidate_experiment.yaml")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "reports" / "aegis_phase2")
    args = parser.parse_args()
    result = run_experiment(args.config)
    report, bundle = write_experiment_result(result, args.output_dir)
    print(f"classification={result.classification}")
    print(f"report={report}")
    print(f"bundle={bundle}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
