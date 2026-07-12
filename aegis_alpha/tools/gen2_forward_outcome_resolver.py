#!/usr/bin/env python3
"""Research-only Gen2 H12 outcome resolver.

It only resolves mature paper records. It never changes policy, retrains,
promotes models, opens orders, or reads immature future data.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import timedelta
from pathlib import Path
from typing import Any

import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

import aegis_alpha.tools.gen2_canary_exec as execv2  # noqa: E402
from aegis_alpha.tools.audit_tail_risk_targets_d2 import json_default  # noqa: E402
from aegis_alpha.tools.gen2_d3_common import utc_now, validate_gen2_path  # noqa: E402


def mature_h12(timestamp: str, now: pd.Timestamp | None = None) -> bool:
    now = now or pd.Timestamp(utc_now())
    ts = pd.Timestamp(timestamp)
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    return now >= ts + timedelta(minutes=60)


def resolve_records(candidate_id: str, records: list[dict[str, Any]], now: pd.Timestamp | None = None) -> dict[str, Any]:
    mature = [r for r in records if r.get("timestamp") and mature_h12(str(r["timestamp"]), now)]
    unresolved = len(records) - len(mature)
    payload = {
        "schema": "gen2_forward_outcome_resolution_v1",
        "candidate_id": candidate_id,
        "records": len(records),
        "mature_h12": len(mature),
        "unresolved_immature": unresolved,
        "policy_changed": False,
        "models_retrained": False,
        "orders_submitted": 0,
        "FORWARD_OUTCOMES_RESEARCH_ONLY": True,
    }
    out = execv2.core.canary_dir(candidate_id) / "outcome_resolution_latest.json"
    validate_gen2_path(out)
    out.write_text(json.dumps(payload, indent=2, default=json_default))
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Resolve mature Gen2 H12 outcomes research-only")
    parser.add_argument("--candidate-id", default=execv2.DEFAULT_CANDIDATE_ID)
    args = parser.parse_args(argv)
    print(json.dumps(resolve_records(args.candidate_id, []), indent=2, default=json_default))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
