#!/usr/bin/env python3
"""FASE-F0 passive TRRM forward score collector.

Research-only, append-only, no labels, no enforcement.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from aegis_alpha.tools.audit_tail_risk_targets_d2 import json_default
from aegis_alpha.tools.trrm_forward_common_f0 import (
    CHAMPION_BUDGET,
    CHAMPION_ENGINE,
    CHAMPION_WINDOW_DAYS,
    DEFAULT_SIGNAL_GLOB,
    DIAGNOSTIC_HORIZONS,
    PRIMARY_HORIZON,
    SCHEMA_VERSION,
    append_jsonl,
    atomic_write_text,
    compact_utc_stamp,
    contains_label_columns,
    deterministic_id,
    existing_by_id,
    guard_no_enforcement_imports,
    inspect_turbo_signal_source,
    iter_turbo_signal_events,
    load_json,
    opportunity_id_from_event,
    parse_dt,
    read_jsonl,
    safe_research_path,
    sha256_json,
    threshold_from_history,
    utc_now,
    write_json,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Collect passive F0 TRRM forward scores")
    p.add_argument("--candidate-dir", required=True)
    p.add_argument("--source-kind", default="turbo_signals_jsonl")
    p.add_argument("--source-path", default=DEFAULT_SIGNAL_GLOB)
    p.add_argument("--source-query", default="")
    p.add_argument("--since", default="")
    p.add_argument("--until", default="")
    p.add_argument("--max-opportunities", type=int, default=0)
    p.add_argument("--dry-run", default="false")
    p.add_argument("--append", default="true")
    p.add_argument("--write-report", default="true")
    p.add_argument("--strict-read-only", default="true")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args(argv)


def parse_bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).lower() not in {"0", "false", "no", "off"}


def source_fingerprint(event: dict[str, Any]) -> str:
    clean = {k: v for k, v in event.items() if not str(k).startswith("_")}
    return sha256_json(clean)


def event_side(event: dict[str, Any]) -> str:
    return str(event.get("raw_action") or event.get("final_action") or event.get("action") or "UNKNOWN").upper()


def event_strategy(event: dict[str, Any]) -> str:
    return str(event.get("strategy") or event.get("mode") or "AEGIS_TURBO")


def is_candidate_event(event: dict[str, Any]) -> bool:
    actions = {str(event.get(k) or "").upper() for k in ("raw_action", "final_action", "gated_action", "action")}
    return bool(actions & {"LONG", "SHORT"} or event.get("gate_allowed") is True or event.get("would_execute") is True)


def extract_scores(event: dict[str, Any]) -> dict[int, float | None]:
    out: dict[int, float | None] = {}
    for horizon in (6, 12, 24):
        for key in (f"score_h{horizon}", f"trrm_score_h{horizon}", f"risk_score_h{horizon}"):
            if key in event and event[key] is not None:
                out[horizon] = float(event[key])
                break
        else:
            out[horizon] = None
    return out


def make_opportunity_record(
    manifest: dict[str, Any],
    event: dict[str, Any],
    threshold: float | None,
    threshold_meta: dict[str, Any],
    recorded_at: str,
) -> dict[str, Any]:
    scores = extract_scores(event)
    score_h12 = scores[12]
    if score_h12 is None:
        decision = "NO_DECISION"
        reason = "MISSING_H12_FEATURES"
    elif threshold is None:
        decision = "NO_DECISION"
        reason = "INSUFFICIENT_HISTORY"
    elif float(score_h12) >= float(threshold):
        decision = "REJECT"
        reason = None
    else:
        decision = "RETAIN"
        reason = None
    source_ref = f"{event.get('_source_path')}:{event.get('_source_line')}"
    source_id = str(event.get("signal_id") or "")
    opp_id = opportunity_id_from_event(event, manifest.get("source_type", "turbo_signals_jsonl"))
    return {
        "schema_version": SCHEMA_VERSION,
        "candidate_id": manifest["candidate_id"],
        "stream": "OPPORTUNITY_STREAM",
        "opportunity_id": opp_id,
        "source_event_id": source_id,
        "source_event_type": "turbo_signal",
        "source_timestamp": str(parse_dt(str(event.get("timestamp")))),
        "recorded_at_utc": recorded_at,
        "symbol": event.get("symbol"),
        "side": event_side(event),
        "strategy_name": event_strategy(event),
        "strategy_version": str(event.get("mode") or ""),
        "timeframe": "5m",
        "primary_horizon": PRIMARY_HORIZON,
        "diagnostic_horizons": DIAGNOSTIC_HORIZONS,
        "score_h6": scores[6],
        "score_h12": score_h12,
        "score_h24": scores[24],
        "primary_threshold": threshold,
        "hypothetical_decision": decision,
        "no_decision_reason": reason,
        "policy_engine": CHAMPION_ENGINE,
        "policy_budget": CHAMPION_BUDGET,
        "rolling_window_days": CHAMPION_WINDOW_DAYS,
        "history_rows": threshold_meta.get("history_rows", 0),
        "feature_hash": manifest["feature_hash"],
        "model_hash": manifest["model_hash"],
        "source_reference": source_ref,
        "source_event_hash": source_fingerprint(event),
        "enforcement_action": "NONE",
        "labels_resolved": False,
    }


def make_model_monitor_records(manifest: dict[str, Any], opportunity: dict[str, Any], threshold: float | None, threshold_meta: dict[str, Any], recorded_at: str) -> list[dict[str, Any]]:
    rows = []
    for horizon in (6, 12, 24):
        score = opportunity.get(f"score_h{horizon}")
        if score is None:
            continue
        rows.append(
            {
                "schema_version": SCHEMA_VERSION,
                "candidate_id": manifest["candidate_id"],
                "stream": "MODEL_MONITOR_STREAM",
                "recorded_at_utc": recorded_at,
                "market_timestamp": opportunity["source_timestamp"],
                "symbol": opportunity["symbol"],
                "horizon": horizon,
                "score": score,
                "threshold": threshold,
                "hypothetical_row_decision": "REJECT" if threshold is not None and score >= threshold else "RETAIN" if threshold is not None else "NO_DECISION",
                "policy_engine": CHAMPION_ENGINE,
                "policy_budget": CHAMPION_BUDGET,
                "rolling_window_days": CHAMPION_WINDOW_DAYS,
                "history_rows": threshold_meta.get("history_rows", 0),
                "history_start": threshold_meta.get("history_start"),
                "history_end": threshold_meta.get("history_end"),
                "feature_hash": manifest["feature_hash"],
                "model_hash": manifest["model_hash"],
                "source_reference": opportunity["source_reference"],
                "enforcement_action": "NONE",
            }
        )
    return rows


def diagnostics(records: list[dict[str, Any]]) -> dict[str, Any]:
    h12 = [r for r in records if r.get("hypothetical_decision") in {"REJECT", "RETAIN"}]
    rejected = [r for r in h12 if r.get("hypothetical_decision") == "REJECT"]
    scores = {h: [r.get(f"score_h{h}") for r in records if r.get(f"score_h{h}") is not None] for h in (6, 12, 24)}
    reasons = Counter(str(r.get("no_decision_reason")) for r in records if r.get("hypothetical_decision") == "NO_DECISION")
    return {
        "stream": "OPPORTUNITY_STREAM",
        "population": "new source events processed in this run",
        "period": "run window",
        "engine": CHAMPION_ENGINE,
        "aggregation": "run_total",
        "h12_rejection_rate": len(rejected) / len(h12) if h12 else None,
        "rolling_7d_rejection_rate": None,
        "rolling_14d_rejection_rate": None,
        "outside_band": None,
        "consecutive_days_outside_band": 0,
        "no_decision_count": sum(reasons.values()),
        "no_decision_reasons": dict(reasons),
        "score_distributions": {
            f"H{h}": {
                "count": len(vals),
                "mean": float(np.mean(vals)) if vals else None,
                "median": float(np.median(vals)) if vals else None,
                "p10": float(np.quantile(vals, 0.10)) if vals else None,
                "p90": float(np.quantile(vals, 0.90)) if vals else None,
            }
            for h, vals in scores.items()
        },
        "FORWARD_OUTCOMES_NOT_EVALUATED": True,
    }


def run_collect(args: argparse.Namespace) -> dict[str, Any]:
    guard_no_enforcement_imports(Path(__file__))
    candidate_dir = Path(args.candidate_dir)
    safe_research_path(candidate_dir)
    manifest = load_json(candidate_dir / "candidate_manifest.json")
    if manifest.get("enforcement_enabled") or manifest.get("labels_enabled"):
        raise ValueError("ENFORCEMENT_PATH_DETECTED")
    source = inspect_turbo_signal_source(args.source_path)
    if source["decision"] != "OPPORTUNITY_SOURCE_READY":
        raise ValueError("OPPORTUNITY_SEMANTICS_NOT_READY")
    since = args.since or manifest["frozen_at_utc"]
    until = args.until or utc_now()
    source_events = iter_turbo_signal_events(args.source_path, since, until, None)
    events = [event for event in source_events if is_candidate_event(event)]
    if args.max_opportunities:
        events = events[: args.max_opportunities]
    recorded_at = utc_now()
    history = read_jsonl(candidate_dir / "history_seed.jsonl")
    opportunity_path = candidate_dir / "opportunity_scores.jsonl"
    monitor_path = candidate_dir / "model_monitor_scores.jsonl"
    existing = existing_by_id(opportunity_path, "opportunity_id")
    new_rows: list[dict[str, Any]] = []
    monitor_rows: list[dict[str, Any]] = []
    duplicates = 0
    conflicts = []
    for event in events:
        if contains_label_columns(event):
            raise ValueError("FORWARD_LABEL_COLUMN_DETECTED")
        opp_id = opportunity_id_from_event(event, manifest.get("source_type", "turbo_signals_jsonl"))
        fp = source_fingerprint(event)
        if opp_id in existing:
            if existing[opp_id].get("source_event_hash") != fp:
                conflicts.append({"opportunity_id": opp_id, "reason": "SOURCE_EVENT_MUTATION_DETECTED"})
            else:
                duplicates += 1
            continue
        ts = parse_dt(str(event.get("timestamp")))
        threshold, meta = threshold_from_history(history, ts, CHAMPION_BUDGET, CHAMPION_WINDOW_DAYS) if ts is not None else (None, {"history_rows": 0})
        row = make_opportunity_record(manifest, event, threshold, meta, recorded_at)
        if contains_label_columns(row):
            raise ValueError("FORWARD_LABEL_COLUMN_DETECTED")
        new_rows.append(row)
        monitor_rows.extend(make_model_monitor_records(manifest, row, threshold, meta, recorded_at))
    if conflicts:
        status = "SOURCE_EVENT_MUTATION_DETECTED"
    elif not events:
        status = "STARTED_NO_NEW_OPPORTUNITIES"
    elif new_rows:
        status = "COLLECTION_STARTED"
    else:
        status = "NO_NEW_UNIQUE_OPPORTUNITIES"
    if parse_bool(args.append) and not parse_bool(args.dry_run) and not conflicts:
        append_jsonl(opportunity_path, new_rows)
        append_jsonl(monitor_path, monitor_rows)
        state = {
            "candidate_id": manifest["candidate_id"],
            "last_run": recorded_at,
            "last_since": since,
            "last_until": until,
            "last_status": status,
            "updated_at_utc": recorded_at,
            "total_existing_before_run": len(existing),
        }
        write_json(candidate_dir / "collection_state.json", state)
    run_stamp = compact_utc_stamp()
    payload = {
        "schema_version": SCHEMA_VERSION,
        "candidate_id": manifest["candidate_id"],
        "decision": "F0_COLLECTION_STARTED" if status == "COLLECTION_STARTED" else "F0_FROZEN_NO_NEW_OPPORTUNITIES" if status in {"STARTED_NO_NEW_OPPORTUNITIES", "NO_NEW_UNIQUE_OPPORTUNITIES"} else "RESEARCH_NOT_READY",
        "collection_status": status,
        "source_inspection": source,
        "since": since,
        "until": until,
        "source_events_seen": len(source_events),
        "events_found": len(events),
        "opportunities_appended": len(new_rows) if parse_bool(args.append) and not parse_bool(args.dry_run) and not conflicts else 0,
        "model_monitor_rows": len(monitor_rows) if parse_bool(args.append) and not parse_bool(args.dry_run) and not conflicts else 0,
        "duplicates_skipped": duplicates,
        "source_conflicts": conflicts,
        "no_decision_count": sum(1 for r in new_rows if r["hypothetical_decision"] == "NO_DECISION"),
        "diagnostics": diagnostics(new_rows),
        "labels_read": False,
        "performance_evaluated": False,
        "enforcement_action": "NONE",
        "dry_run": parse_bool(args.dry_run),
        "append": parse_bool(args.append),
        "FORWARD_OUTCOMES_NOT_EVALUATED": True,
    }
    if parse_bool(args.write_report):
        run_dir = candidate_dir / "collection_runs"
        run_dir.mkdir(parents=True, exist_ok=True)
        js = run_dir / f"{run_stamp}.json"
        md = run_dir / f"{run_stamp}.md"
        write_json(js, payload)
        atomic_write_text(
            md,
            "\n".join(
                [
                    "# FASE-F0 Collection Run",
                    "",
                    f"- decision: {payload['decision']}",
                    f"- status: {status}",
                    f"- since: {since}",
                    f"- until: {until}",
                    f"- events_found: {len(events)}",
                    f"- opportunities_appended: {payload['opportunities_appended']}",
                    f"- duplicates_skipped: {duplicates}",
                    f"- no_decision_count: {payload['no_decision_count']}",
                    "- FORWARD_OUTCOMES_NOT_EVALUATED",
                    "- enforcement_action: NONE",
                    "",
                ]
            ),
        )
        payload["report_json"] = str(js)
        payload["report_md"] = str(md)
    print(json.dumps(payload, indent=2, default=json_default))
    return payload


def main(argv: list[str] | None = None) -> int:
    try:
        run_collect(parse_args(argv))
        return 0
    except ValueError as exc:
        msg = str(exc)
        decision = "RESEARCH_NOT_READY"
        if "OPPORTUNITY_SEMANTICS_NOT_READY" in msg:
            decision = "OPPORTUNITY_SEMANTICS_NOT_READY"
        elif "ENFORCEMENT_PATH_DETECTED" in msg:
            decision = "ENFORCEMENT_PATH_DETECTED"
        print(json.dumps({"decision": decision, "reason": msg}))
        return 2 if decision != "RESEARCH_NOT_READY" else 1


if __name__ == "__main__":
    raise SystemExit(main())
