#!/usr/bin/env python3
"""CLI for the isolated compatibility control and closed causal ablations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from scripts.diagnostics.compat_replay.ablations import (
    AblationNondeterministic,
    AblationProtocolError,
    run_all_ablations,
)
from scripts.diagnostics.compat_replay.replay import HistoricalReproductionMismatch, run_stage_zero
from scripts.diagnostics.compat_replay.schemas import ReplayConfig


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path(__file__).with_name("replay_v1.yaml"))
    parser.add_argument("--mode", choices=("stage-0", "ablations"), default="stage-0")
    parser.add_argument(
        "--e2-econ-reference", type=Path,
        default=Path("/tmp/aegis-e2-validation-2/aegis-short-candidate-e2/runs/b49f1ec7c71fcf70/econ_report.json"),
    )
    args = parser.parse_args()
    try:
        config = ReplayConfig.load(args.config)
        if args.mode == "stage-0":
            result = run_stage_zero(config)
            verdict = "HISTORICAL_HARNESS_REPRODUCTION_CONFIRMED"
        else:
            result = run_all_ablations(config, args.e2_econ_reference)
            verdict = "CAUSAL_ABLATIONS_COMPLETE_READY_FOR_REVIEW"
    except HistoricalReproductionMismatch as error:
        print(json.dumps({"verdict": "HISTORICAL_REPRODUCTION_MISMATCH", "error": str(error)}))
        return 3
    except AblationNondeterministic as error:
        print(json.dumps({"verdict": "ABLATION_NONDETERMINISTIC", "error": str(error)}))
        return 4
    except AblationProtocolError as error:
        print(json.dumps({"verdict": "ABLATION_PROTOCOL_BLOCKED", "error": str(error)}))
        return 5
    except Exception as error:
        print(json.dumps({"verdict": "COMPATIBILITY_REPLAY_BLOCKED", "error": str(error)}))
        return 2
    print(json.dumps({"verdict": verdict, "result": result}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
