#!/usr/bin/env python3
"""Safe CLI for the persistent Phase-E orchestration state machine."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from aegis.training.phase_e import (
    FULL_RUN_AUTHORIZATION, PhaseEOrchestrator, PhaseEPreflight,
    SimulatedScientificBackend,
)
from aegis.training.run_state import PhaseEErrorCode, PhaseETechnicalError, RunMode
from aegis.utils import to_primitive


class ProductionBackendNotConfigured:
    """Fail-closed marker until the preregistered calibration split is unambiguous."""

    def __getattr__(self, name: str):
        raise PhaseETechnicalError(
            PhaseEErrorCode.PRECHECK_FAILED,
            "production Phase-E backend is not configured; calibration split and sampling contract are incomplete",
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=Path,
        default=ROOT / "config" / "experiments" / "aegis_short_candidate_e1.yaml",
    )
    parser.add_argument(
        "--competition-config", type=Path,
        default=ROOT / "config" / "scientific_competition_v1.yaml",
    )
    parser.add_argument("--mode", choices=[item.value for item in RunMode], default=RunMode.DRY_RUN.value)
    parser.add_argument("--owner-authorization")
    parser.add_argument("--smoke-outcome", choices=("candidate", "rejected"), default="candidate")
    parser.add_argument("--reports-root", type=Path, default=ROOT / "reports" / "experiments")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    mode = RunMode(args.mode)
    if mode is RunMode.FULL_RUN and args.owner_authorization != FULL_RUN_AUTHORIZATION:
        print("error=FULL_RUN_OWNER_AUTHORIZATION_REQUIRED", file=sys.stderr)
        return 2
    preflight = PhaseEPreflight(
        repository_root=ROOT,
        typescript_root=ROOT / "binance-futures-bot-ts",
        preregistration_path=args.config,
        competition_path=args.competition_config,
    )
    backend = (
        SimulatedScientificBackend(approving=args.smoke_outcome == "candidate")
        if mode is RunMode.SMOKE_RUN else ProductionBackendNotConfigured()
    )
    runner = PhaseEOrchestrator(
        preflight=preflight, backend=backend, reports_root=args.reports_root,
        mode=mode, owner_authorization=args.owner_authorization,
        smoke_approving=args.smoke_outcome == "candidate",
    )
    try:
        result = runner.run()
    except PhaseETechnicalError as exc:
        print(f"error={exc.code.value}", file=sys.stderr)
        print(f"message={exc}", file=sys.stderr)
        return 3
    print(json.dumps(to_primitive(result), sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
