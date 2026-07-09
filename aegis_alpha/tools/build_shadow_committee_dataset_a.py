#!/usr/bin/env python3
"""Research-only Shadow Committee dataset builder."""
from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DEFAULT_OUT_DIR = Path("/home/jasan/Develop")
SCHEMA_FIELDS = [
    "timestamp",
    "symbol",
    "side",
    "score",
    "bucket",
    "turbo_votes_7d",
    "turbo_votes_14d",
    "turbo_votes_30d",
    "clean_entry_would_block",
    "clean_entry_reason",
    "setup_grade",
    "entry_quality_action",
    "entry_quality_score",
    "entry_quality_reason",
    "decision_brain_action",
    "decision_brain_reason",
    "event_risk_action",
    "event_risk_level",
    "regime_action",
    "regime_name",
    "final_action",
    "was_executed",
    "was_blocked",
    "linked_trade_id",
    "realized_outcome",
    "clean_entry_v4",
    "bad_entry_v4",
    "management_dependent_v4",
    "no_trade_v4",
    "outcome_class",
]
BASELINE_FIELDS = [
    "timestamp",
    "symbol",
    "linked_trade_id",
    "block_if_2_or_more_guards_block",
    "block_if_setup_grade_weak",
    "block_if_entry_quality_shadow_block",
    "block_if_event_risk_high",
    "block_if_regime_no_trade",
    "baseline_block_count",
]


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def normalize_action(value: Any) -> str:
    return str(value or "").strip().upper()


def guard_blocks(row: dict[str, Any]) -> int:
    guards = [
        bool(row.get("clean_entry_would_block")),
        normalize_action(row.get("entry_quality_action")) in {"BLOCK", "SHADOW_BLOCK"},
        normalize_action(row.get("decision_brain_action")) == "BLOCK",
        normalize_action(row.get("event_risk_action")) in {"BLOCK", "PAUSE"},
        normalize_action(row.get("regime_action")) in {"NO_TRADE", "BLOCK"},
    ]
    return sum(1 for x in guards if x)


def build_rows_from_events(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for ev in events:
        if normalize_action(ev.get("event_type")) != "SIGNAL":
            continue
        row = {k: "" for k in SCHEMA_FIELDS}
        row.update({
            "timestamp": ev.get("timestamp", ""),
            "symbol": ev.get("symbol", ""),
            "side": ev.get("side", ""),
            "score": ev.get("score", ""),
            "bucket": ev.get("bucket", ""),
            "linked_trade_id": ev.get("trade_id", ""),
            "was_executed": False,
            "was_blocked": False,
            "final_action": "UNKNOWN",
        })
        rows.append(row)
    return rows


def apply_rule_baselines(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in rows:
        weak = str(row.get("setup_grade", "")).upper() in {"WEAK", "F", "D"}
        eq_block = normalize_action(row.get("entry_quality_action")) in {"BLOCK", "SHADOW_BLOCK"}
        event_high = str(row.get("event_risk_level", "")).upper() in {"HIGH", "EXTREME"}
        regime_no = normalize_action(row.get("regime_action")) in {"NO_TRADE", "BLOCK"}
        count = guard_blocks(row)
        out.append({
            "timestamp": row.get("timestamp", ""),
            "symbol": row.get("symbol", ""),
            "linked_trade_id": row.get("linked_trade_id", ""),
            "block_if_2_or_more_guards_block": count >= 2,
            "block_if_setup_grade_weak": weak,
            "block_if_entry_quality_shadow_block": eq_block,
            "block_if_event_risk_high": event_high,
            "block_if_regime_no_trade": regime_no,
            "baseline_block_count": count,
        })
    return out


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fields})


def run_builder(args: argparse.Namespace) -> dict[str, Any]:
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    # Logs are intentionally optional in this phase; empty schema is valid.
    rows: list[dict[str, Any]] = []
    baseline_rows = apply_rule_baselines(rows)
    status = "OK" if rows else "INSUFFICIENT_SHADOW_LOGS"
    timestamp = utc_now().strftime("%Y%m%dT%H%M%SZ")
    coverage = [{"field": f, "non_empty": sum(1 for r in rows if r.get(f)), "row_count": len(rows)} for f in SCHEMA_FIELDS]
    result = {
        "schema_version": "shadow_committee_dataset_a_v1",
        "status": status,
        "generated_at": timestamp,
        "row_count": len(rows),
        "schema_fields": SCHEMA_FIELDS,
        "training_readiness": "not_ready_wait_for_forward_logs",
        "note": "No model training was run.",
    }
    json_path = out_dir / f"aegis_shadow_committee_dataset_a_{timestamp}.json"
    md_path = out_dir / f"aegis_shadow_committee_dataset_a_{timestamp}.md"
    csv_path = out_dir / f"aegis_shadow_committee_dataset_a_{timestamp}.csv"
    base_path = out_dir / f"aegis_shadow_committee_rule_baselines_{timestamp}.csv"
    cov_path = out_dir / f"aegis_shadow_committee_coverage_{timestamp}.csv"
    json_path.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    md_path.write_text(
        "\n".join([
            "# Aegis Shadow Committee Dataset A",
            "",
            f"- status: {status}",
            f"- rows: {len(rows)}",
            "- no_train: true",
            "",
            "Schema is ready for forward logs and future snapshots.",
            "",
            "Fields:",
            *[f"- {f}" for f in SCHEMA_FIELDS],
        ]) + "\n",
        encoding="utf-8",
    )
    write_csv(csv_path, rows, SCHEMA_FIELDS)
    write_csv(base_path, baseline_rows, BASELINE_FIELDS)
    write_csv(cov_path, coverage, ["field", "non_empty", "row_count"])
    result["outputs"] = {"json": str(json_path), "md": str(md_path), "csv": str(csv_path), "baselines_csv": str(base_path), "coverage_csv": str(cov_path)}
    return result


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Build research-only Shadow Committee dataset.")
    p.add_argument("--from", dest="from_time", default="now-7d")
    p.add_argument("--to", dest="to_time", default="now")
    p.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    p.add_argument("--include-rule-baselines", action="store_true")
    p.add_argument("--no-train", action="store_true")
    return p


def main() -> int:
    result = run_builder(build_parser().parse_args())
    print(json.dumps({k: result[k] for k in ("status", "row_count", "outputs")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
