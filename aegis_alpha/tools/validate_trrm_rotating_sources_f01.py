#!/usr/bin/env python3
"""Validate rotating Turbo signal sources for FASE-F0.1.

Read-only source inspection. Does not collect labels, outcomes, or scores.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from aegis_alpha.tools.audit_tail_risk_targets_d2 import json_default
from aegis_alpha.tools.trrm_forward_common_f0 import (
    DEFAULT_SIGNAL_GLOB,
    atomic_write_text,
    compact_utc_stamp,
    load_json,
    load_rotating_signal_events,
    safe_research_path,
    utc_now,
    write_json,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Validate F0.1 rotating Turbo signal sources")
    p.add_argument("--candidate-dir", default="")
    p.add_argument("--source-glob", default="")
    p.add_argument("--source-dir", default="")
    p.add_argument("--source-pattern", default="turbo_signals_*.jsonl")
    p.add_argument("--since", default="")
    p.add_argument("--until", default="")
    p.add_argument("--dry-run", default="true")
    p.add_argument("--strict-read-only", default="true")
    p.add_argument("--write-report", default="true")
    return p.parse_args(argv)


def parse_bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).lower() not in {"0", "false", "no", "off"}


def run_validate(args: argparse.Namespace) -> dict[str, object]:
    previous_state = {}
    frozen_at = None
    candidate_id = None
    if args.candidate_dir:
        cdir = Path(args.candidate_dir)
        manifest = load_json(cdir / "candidate_manifest.json")
        candidate_id = manifest.get("candidate_id")
        frozen_at = manifest.get("frozen_at_utc")
        state = cdir / "collection_state.json"
        previous_state = load_json(state) if state.exists() else {}
    since = args.since or frozen_at or ""
    until = args.until or utc_now()
    effective_glob = args.source_glob or ("" if args.source_dir else DEFAULT_SIGNAL_GLOB)
    events, report = load_rotating_signal_events(
        None,
        effective_glob or None,
        args.source_dir or None,
        args.source_pattern,
        since,
        until,
        previous_state,
    )
    decision = report["readiness"]
    if decision == "ROTATING_SOURCE_EMPTY" and report.get("files_matched", 0):
        decision = "ROTATING_SOURCE_EMPTY"
    payload = {
        "phase": "F0.1",
        "candidate_id": candidate_id,
        "decision": decision,
        "source_glob": effective_glob or (str(Path(args.source_dir) / args.source_pattern) if args.source_dir else DEFAULT_SIGNAL_GLOB),
        "since": since,
        "until": until,
        "files_matched": report.get("files_matched", 0),
        "files_readable": report.get("files_readable", 0),
        "files_empty": report.get("files_empty", 0),
        "incomplete_trailing_lines": report.get("incomplete_trailing_lines", 0),
        "earliest_event": report.get("earliest_event"),
        "latest_event": report.get("latest_event"),
        "total_events": report.get("total_events", 0),
        "unique_source_event_ids": report.get("unique_source_event_ids", 0),
        "duplicates": report.get("duplicates", 0),
        "mutations": report.get("mutations", []),
        "files_containing_post_freeze_events": report.get("post_freeze_files", []),
        "rotation_detected": report.get("rotation_detected", False),
        "truncation_detected": report.get("truncation_detected", False),
        "files": report.get("files", []),
        "labels_read": False,
        "performance_evaluated": False,
        "enforcement_action": "NONE",
        "FORWARD_OUTCOMES_NOT_EVALUATED": True,
        "dry_run": parse_bool(args.dry_run),
        "strict_read_only": parse_bool(args.strict_read_only),
        "events_loaded_for_validation": len(events),
    }
    if parse_bool(args.write_report):
        if args.candidate_dir:
            out_dir = Path(args.candidate_dir) / "collection_runs"
        else:
            out_dir = Path("/home/jasan/Develop/aegis_forward_research/trrm_f01/source_validation")
        safe_research_path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        stamp = compact_utc_stamp()
        js = out_dir / f"aegis_phase_f01_rotating_source_{stamp}.json"
        md = out_dir / f"aegis_phase_f01_rotating_source_{stamp}.md"
        write_json(js, payload)
        atomic_write_text(
            md,
            "\n".join(
                [
                    "# FASE-F0.1 Rotating Source Validation",
                    "",
                    f"- decision: {decision}",
                    f"- files_matched: {payload['files_matched']}",
                    f"- total_events: {payload['total_events']}",
                    f"- duplicates: {payload['duplicates']}",
                    f"- incomplete_trailing_lines: {payload['incomplete_trailing_lines']}",
                    "- FORWARD_OUTCOMES_NOT_EVALUATED",
                    "",
                ]
            ),
        )
        payload["report_json"] = str(js)
        payload["report_md"] = str(md)
    print(json.dumps(payload, indent=2, default=json_default))
    return payload


def main(argv: list[str] | None = None) -> int:
    payload = run_validate(parse_args(argv))
    return 0 if payload["decision"] == "ROTATING_SOURCE_READY" else 2


if __name__ == "__main__":
    raise SystemExit(main())
