#!/usr/bin/env python3
"""Gen2 canary bracket/reconciliation daily report helpers.

Read-only over canary JSONL streams. It does not submit, cancel, close, or
mutate exchange state. Output is a research/ops report under the candidate dir.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

import aegis_alpha.tools.gen2_canary_core as core  # noqa: E402
from aegis_alpha.tools.audit_tail_risk_targets_d2 import json_default  # noqa: E402
from aegis_alpha.tools.gen2_d3_common import validate_gen2_path  # noqa: E402


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def bracket_summary(candidate_dir: Path) -> dict[str, Any]:
    rows = read_jsonl(candidate_dir / "brackets.jsonl")
    failures = [r for r in rows if r.get("ok") is not True]
    return {
        "stream": "BRACKETS",
        "rows": len(rows),
        "confirmed": sum(1 for r in rows if r.get("ok") is True),
        "failures": len(failures),
        "failure_reasons": sorted({str(r.get("reason")) for r in failures}),
    }


def reconciliation_summary(candidate_dir: Path) -> dict[str, Any]:
    rows = read_jsonl(candidate_dir / "reconciliations.jsonl")
    failures = [r for r in rows if r.get("status") != "RECONCILED"]
    incident_types = sorted({item for row in failures for item in row.get("incidents", [])})
    return {
        "stream": "RECONCILIATIONS",
        "rows": len(rows),
        "reconciled": sum(1 for r in rows if r.get("status") == "RECONCILED"),
        "fail_closed": len(failures),
        "incident_types": incident_types,
    }


def live_order_summary(candidate_dir: Path) -> dict[str, Any]:
    rows = read_jsonl(candidate_dir / "live_orders.jsonl")
    return {
        "stream": "LIVE_ORDERS",
        "rows": len(rows),
        "orders_submitted": sum(1 for r in rows if r.get("order_action") == "SUBMIT_ORDER"),
        "no_order": sum(1 for r in rows if r.get("order_action") == "NO_ORDER"),
        "reasons": sorted({str(r.get("reason")) for r in rows if r.get("reason")}),
    }


def build_daily_report(candidate_id: str) -> dict[str, Any]:
    candidate_dir = core.canary_dir(candidate_id)
    validate_gen2_path(candidate_dir)
    phase = core.phase_o_conflict_audit()
    payload = {
        "schema": "gen2_canary_daily_reconciliation_v1",
        "candidate_id": candidate_id,
        "armed": core.verify_arm_token(candidate_id)[0],
        "kill_switch": core.kill_switch_engaged(candidate_id),
        "phase_o_conflict": phase,
        "live_orders": live_order_summary(candidate_dir),
        "brackets": bracket_summary(candidate_dir),
        "reconciliations": reconciliation_summary(candidate_dir),
        "orders_submitted": live_order_summary(candidate_dir)["orders_submitted"],
        "FORWARD_OUTCOMES_NOT_EVALUATED": True,
    }
    out = candidate_dir / "daily_reports" / "latest_reconciliation_report.json"
    validate_gen2_path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True, default=json_default), encoding="utf-8")
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build Gen2 canary reconciliation report")
    parser.add_argument("--candidate-id", default="gen2-20260711T202935Z")
    args = parser.parse_args(argv)
    print(json.dumps(build_daily_report(args.candidate_id), indent=2, default=json_default))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
