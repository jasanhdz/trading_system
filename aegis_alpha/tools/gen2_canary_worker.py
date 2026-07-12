#!/usr/bin/env python3
"""Manual Gen2 canary worker.

Not installed in PM2. It starts disarmed, records paper/live parity metadata,
and delegates execution attempts to gen2_canary_exec with dry-run default.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

import aegis_alpha.tools.gen2_canary_core as core  # noqa: E402
import aegis_alpha.tools.gen2_canary_exec as execv2  # noqa: E402
from aegis_alpha.tools.audit_tail_risk_targets_d2 import json_default  # noqa: E402
from aegis_alpha.tools.gen2_d3_common import utc_now, validate_gen2_path  # noqa: E402


def feature_snapshot_hash(features: dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(features, sort_keys=True, default=json_default).encode()).hexdigest()


def build_paper_record(opportunity: dict[str, Any]) -> dict[str, Any]:
    features = opportunity.get("features", {})
    return {
        "schema": "gen2_canary_paper_live_parity_v1",
        "candidate_id": opportunity.get("candidate_id", execv2.DEFAULT_CANDIDATE_ID),
        "opportunity_id": opportunity.get("opportunity_id"),
        "signal_id": opportunity.get("signal_id"),
        "timestamp": opportunity.get("timestamp"),
        "symbol": opportunity.get("symbol"),
        "side": opportunity.get("side", "SHORT"),
        "feature_snapshot_hash": feature_snapshot_hash(features),
        "trrm_score": opportunity.get("trrm_score"),
        "qmae": opportunity.get("qmae"),
        "eqm_score": opportunity.get("eqm_score"),
        "veto": opportunity.get("veto", False),
        "selection_rank": opportunity.get("selection_rank"),
        "paper_decision": opportunity.get("paper_decision", "NO_DECISION"),
        "canary_eligibility_decision": "NOT_EVALUATED",
        "enforcement_action": "NONE",
        "recorded_at_utc": utc_now().isoformat(),
    }


def run_once(candidate_id: str, opportunity: dict[str, Any] | None = None, dry_run: bool = True) -> dict[str, Any]:
    core.init_canary(candidate_id)
    cdir = core.canary_dir(candidate_id)
    armed, arm_reason, _ = execv2.verify_arm_token_v2(candidate_id)
    heartbeat = {
        "schema": "gen2_canary_worker_heartbeat_v1",
        "candidate_id": candidate_id,
        "armed": armed,
        "arm_reason": arm_reason,
        "dry_run": dry_run,
        "real_order_submission_enabled": execv2.REAL_ORDER_SUBMISSION_ENABLED,
        "recorded_at_utc": utc_now().isoformat(),
    }
    if opportunity is None:
        heartbeat["status"] = "NO_OPPORTUNITY"
        out = cdir / "worker_heartbeat.json"
        validate_gen2_path(out)
        out.write_text(json.dumps(heartbeat, indent=2, default=json_default))
        return heartbeat
    paper = build_paper_record({**opportunity, "candidate_id": candidate_id})
    execv2.append_jsonl(cdir / "paper_live_parity.jsonl", paper)
    if not armed:
        paper["canary_eligibility_decision"] = arm_reason
        paper["status"] = "UNARMED_NO_ORDER"
        return paper
    paper["status"] = "ARMED_PATH_NOT_EXECUTED_IN_THIS_TASK" if dry_run else "ARMED"
    return paper


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Manual Gen2 canary worker")
    parser.add_argument("--candidate-id", default=execv2.DEFAULT_CANDIDATE_ID)
    parser.add_argument("--dry-run", action="store_true", default=True)
    parser.add_argument("--mode", choices=["heartbeat", "wait-for-next-eligible-opportunity"], default="heartbeat")
    args = parser.parse_args(argv)
    result = run_once(args.candidate_id, None, dry_run=args.dry_run)
    print(json.dumps(result, indent=2, default=json_default))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
