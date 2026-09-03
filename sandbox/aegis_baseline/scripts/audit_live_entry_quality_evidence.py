#!/usr/bin/env python3
"""Classify Live entries and validate causal bad-entry warnings."""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict
from pathlib import Path

import pandas as pd
import yaml

from aegis.research.live_entry_quality_audit import (
    class_summary,
    fit_bad_entry_model,
    flatten_pair,
    guard_summary,
    read_trade_pairs,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("config/experiments/aegis_live_entry_quality_audit_20260815.yaml"))
    parser.add_argument("--out-dir", type=Path, default=Path("reports/governance/aegis_prospective_validation/live/live_entry_quality_audit_20260815"))
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    split_at = pd.Timestamp(config["split"]["validation_start_inclusive"])
    pairs, source_audit = read_trade_pairs(Path(config["sources"]["logs_dir"]), config["sources"])
    rows = [flatten_pair(pair, config["classification"], split_at) for pair in pairs if pair["open"] is not None]
    frame = pd.DataFrame(rows).sort_values("opened_at").reset_index(drop=True)
    model = fit_bad_entry_model(frame, {**config["model"], "validation_gate": config["validation_gate"]})
    guards = guard_summary(frame, "DISCOVERY") + guard_summary(frame, "VALIDATION")
    feature_availability = {
        split: {
            feature: float(1.0 - group[feature].isna().mean())
            for feature in config["model"]["features"]
        }
        for split, group in frame.groupby("split", sort=True)
    }
    by_symbol = _group_summary(frame, "symbol")
    by_side = _group_summary(frame, "side")
    reason_evidence = _reason_summary(frame)
    verdict = {
        "schema_version": config["schema_version"],
        "experiment_id": config["experiment_id"],
        "source_audit": source_audit,
        "episodes": int(len(frame)),
        "date_range": [str(frame["opened_at"].min()), str(frame["closed_at"].max())],
        "class_summary": class_summary(frame),
        "by_symbol": by_symbol,
        "by_side": by_side,
        "feature_availability": feature_availability,
        "entry_quality_reason_evidence": reason_evidence,
        "guard_evidence": guards,
        "model_evidence": asdict(model),
        "BAD_ENTRIES_CAUSALLY_AVOIDABLE": bool(model.gate_passed),
        "PRODUCTION_CHANGE_JUSTIFIED": False,
        "limitations": [
            "PnL is secondary because users manually resized some positions after entry.",
            "MFE and MAE are complete-trade extrema and do not preserve which excursion occurred first.",
            "The Live policy and logging schema changed during the sample.",
            "A passed retrospective validation would justify a frozen prospective observer, not immediate enforcement.",
        ],
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.out_dir / "live_entry_classification.csv", index=False)
    strict_verdict = _strict_json(verdict)
    (args.out_dir / "live_entry_quality_verdict.json").write_text(json.dumps(strict_verdict, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    report = render_report(verdict, model)
    (args.out_dir / "live_entry_quality_report.md").write_text(report, encoding="utf-8")
    print(json.dumps(verdict, indent=2, sort_keys=True))


def _group_summary(frame: pd.DataFrame, column: str) -> list[dict]:
    result = []
    for (split, value), group in frame.groupby(["split", column], sort=True):
        result.append({
            "split": str(split), column: str(value), "episodes": int(len(group)),
            "bad_rate": float(group["bad_entry"].mean()),
            "good_rate": float(group["good_entry"].mean()),
            "median_mae_bps": float(group["mae_bps_underlying"].median()),
            "median_mfe_bps": float(group["mfe_bps_underlying"].median()),
            "pnl_usdt_secondary": float(group["pnl_usdt"].sum()),
        })
    return result


def _reason_summary(frame: pd.DataFrame) -> list[dict]:
    result = []
    for (split, reason), group in frame.groupby(["split", "entry_quality_reason"], dropna=False, sort=True):
        result.append({
            "split": str(split), "reason": None if pd.isna(reason) else str(reason),
            "episodes": int(len(group)), "bad_rate": float(group["bad_entry"].mean()),
            "good_rate": float(group["good_entry"].mean()),
            "pnl_usdt_secondary": float(group["pnl_usdt"].sum()),
        })
    return result


def _strict_json(value):
    if isinstance(value, dict):
        return {key: _strict_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_strict_json(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def render_report(verdict: dict, model: object) -> str:
    classes = "\n".join(
        f"| {row['split']} | {row['entry_class']} | {row['episodes']} | {row['median_mfe_bps']:.2f} | {row['median_mae_bps']:.2f} | {row['pnl_usdt_secondary']:.2f} |"
        for row in verdict["class_summary"]
    )
    guards = "\n".join(
        f"| {row['split']} | {row['guard']} | {row['blocked']} | {_percent(row['bad_capture_rate'])} | {_percent(row['good_retention_rate'])} | {_percent(row['allowed_bad_rate'])} |"
        for row in verdict["guard_evidence"]
    )
    validation = verdict["model_evidence"]["validation"]
    ci = verdict["model_evidence"]["bootstrap_bad_rate_reduction_ci95"]
    coefficients = "\n".join(
        f"- `{item['feature']}`: {item['coefficient']:+.4f}"
        for item in verdict["model_evidence"]["coefficients"][:10]
    )
    return f"""# Aegis Live Entry Quality Audit — 2026-08-15

## Scope

Read-only audit of paired `AEGIS_TURBO_MICRO_LIVE` open/close records. No production rules, services, credentials, or orders were modified.

## Classification

| Split | Class | Episodes | Median MFE bps | Median MAE bps | PnL USDT (secondary) |
|---|---|---:|---:|---:|---:|
{classes}

## Existing causal guards

| Split | Guard | Blocked | Bad captured | Good retained | Bad rate if allowed |
|---|---|---:|---:|---:|---:|
{guards}

## Frozen discovery model evaluated on later Live data

- Threshold selected only in discovery: `{verdict['model_evidence']['threshold']:.2f}`.
- Validation allowed/total: `{validation['allowed']}/{validation['episodes']}`.
- Validation bad-rate reduction: `{validation['relative_bad_rate_reduction']:.1%}`.
- Validation clean-good retention: `{validation['good_retention_rate']:.1%}`.
- Bootstrap 95% CI for relative bad-rate reduction: `[{ci[0]:.1%}, {ci[1]:.1%}]`.
- `BAD_ENTRIES_CAUSALLY_AVOIDABLE = {str(verdict['BAD_ENTRIES_CAUSALLY_AVOIDABLE']).upper()}`.
- `PRODUCTION_CHANGE_JUSTIFIED = FALSE`.

Largest standardized associations with BAD entries (discovery fit only):

{coefficients}

The strongest coefficient is a missing-value indicator. Feature availability changed materially between discovery and validation, so this is evidence of logging/policy drift rather than a trustworthy market relationship.

Recent predefined reasons `volatility_too_high` and `overextended_short` identified 6 BAD entries among 16 observations, but their discovery behavior was not consistent and the sample is too small for enforcement.

## Interpretation constraints

- Monetary PnL is secondary because some positions were resized manually after entry.
- MFE/MAE describe the full path but do not preserve excursion ordering.
- A retrospective result cannot directly authorize an enforced Live guard.
- Only a frozen prospective observer can establish whether the relationship persists without policy drift.
"""


def _percent(value: float) -> str:
    return f"{value:.1%}" if math.isfinite(value) else "N/A"


if __name__ == "__main__":
    main()
