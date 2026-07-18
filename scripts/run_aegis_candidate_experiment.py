#!/usr/bin/env python3
"""Validate the Phase-E preregistration; full execution requires owner authorization."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from aegis.training.preregistration import load_and_validate_preregistration


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "config" / "experiments" / "aegis_short_candidate_e1.yaml")
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--execute-full", action="store_true")
    parser.add_argument("--owner-authorization")
    args = parser.parse_args()
    if args.execute_full and args.owner_authorization != "OWNER_AUTHORIZED_PHASE_E_FULL_RUN":
        parser.error("full run requires the exact owner authorization phrase")
    _, audit = load_and_validate_preregistration(args.config, audit_source=True)
    print(f"experiment_id={audit.experiment_id}")
    print(f"preregistration_hash={audit.content_hash}")
    print("status=PRE_REGISTERED_NOT_EXECUTED")
    if args.execute_full:
        raise SystemExit("full Phase-E execution is intentionally unavailable in this implementation-only stage")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
