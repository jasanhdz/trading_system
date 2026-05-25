#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def load_rows(paths: list[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in paths:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        for row in payload.get("summaries", []):
            rows.append({**row, "source_report": path})
    return rows


def _num(value: Any) -> str:
    return "null" if value is None else f"{float(value):.4f}"


def write_outputs(rows: list[dict[str, Any]], out_dir: Path) -> dict[str, str]:
    token = stamp()
    paths = {
        "md": out_dir / f"aegis_turbo_v2_feature_set_comparison_{token}.md",
        "json": out_dir / f"aegis_turbo_v2_feature_set_comparison_{token}.json",
        "csv": out_dir / f"aegis_turbo_v2_feature_set_comparison_{token}.csv",
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    columns = (
        "symbol", "side", "feature_set", "recommendation", "valid_fold_count",
        "baseline_hit8_mean", "v2_hit8_auc_mean", "hit8_top_decile_lift_mean",
        "quality_top_decile_lift_mean", "v2_quality_corr_mean", "v2_danger_auc_mean",
        "stability_score", "latest_fold_hit8_lift", "latest_fold_quality_lift",
        "latest_fold_quality_p90_mae", "latest_fold_baseline_p90_mae",
        "latest_fold_baseline_danger_rate", "source_report",
    )
    with paths["csv"].open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    report = {
        "schema_version": "aegis_turbo_v2_feature_set_comparison_v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "mode": "RESEARCH_ONLY",
        "rows": rows,
        "paths": {key: str(value) for key, value in paths.items()},
    }
    paths["json"].write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    lines = [
        "# Aegis Turbo V2 Feature Set Comparison",
        "",
        "- Mode: `RESEARCH_ONLY`",
        "- Source: walk-forward summaries; no active models or inference were changed.",
        "",
        "| Symbol / Side | Feature Set | Status | Hit8 AUC | Hit8 Lift | Quality Lift | Quality Corr | Danger AUC | Stability | Latest Quality Lift |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in sorted(rows, key=lambda item: (str(item.get("symbol")), str(item.get("feature_set")))):
        lines.append(
            f"| {row.get('symbol')} {row.get('side')} | {row.get('feature_set', 'base')} | {row.get('recommendation')} | "
            f"{_num(row.get('v2_hit8_auc_mean'))} | {_num(row.get('hit8_top_decile_lift_mean'))} | "
            f"{_num(row.get('quality_top_decile_lift_mean'))} | {_num(row.get('v2_quality_corr_mean'))} | "
            f"{_num(row.get('v2_danger_auc_mean'))} | {_num(row.get('stability_score'))} | "
            f"{_num(row.get('latest_fold_quality_lift'))} |"
        )
    paths["md"].write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {key: str(value) for key, value in paths.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare research-only Turbo V2 walk-forward feature-set reports.")
    parser.add_argument("--reports", nargs="+", required=True)
    parser.add_argument("--out-dir", default="/home/jasan/Develop")
    args = parser.parse_args()
    paths = write_outputs(load_rows(args.reports), Path(args.out_dir))
    print(json.dumps(paths, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
