#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


STATUS_ORDER = {"STRONG_BEST": 3, "MIXED_BEST": 2, "BAD_BEST": 1, "NO_VALID_CONFIG": 0}


def compare_best(v2_report: dict[str, Any], v3_report: dict[str, Any]) -> list[dict[str, Any]]:
    v2_by_symbol = {str(row["symbol"]): row for row in v2_report.get("best_by_symbol", [])}
    rows: list[dict[str, Any]] = []
    for current in v3_report.get("best_by_symbol", []):
        symbol = str(current["symbol"])
        previous = v2_by_symbol.get(symbol, {})
        previous_status = str(previous.get("best_status", "NO_PRIOR_RESULT"))
        current_status = str(current.get("best_status", "NO_VALID_CONFIG"))
        before = STATUS_ORDER.get(previous_status, -1)
        after = STATUS_ORDER.get(current_status, -1)
        rows.append({
            "symbol": symbol,
            "v2_status": previous_status,
            "v2_feature_set": previous.get("feature_set"),
            "v2_lookback_days": previous.get("lookback_days"),
            "v2_horizon_candles": previous.get("horizon_candles"),
            "v3_status": current_status,
            "v3_feature_set": current.get("feature_set"),
            "v3_lookback_days": current.get("lookback_days"),
            "v3_horizon_candles": current.get("horizon_candles"),
            "v3_selection_score": current.get("selection_score"),
            "v3_hit8_auc": current.get("v2_hit8_auc_mean"),
            "v3_hit8_lift": current.get("hit8_top_decile_lift_mean"),
            "v3_quality_lift": current.get("quality_top_decile_lift_mean"),
            "v3_quality_corr": current.get("v2_quality_corr_mean"),
            "v3_latest_quality_lift": current.get("latest_fold_quality_lift"),
            "v3_p90_delta": current.get("latest_p90_mae_delta"),
            "status_change": "IMPROVED" if after > before else ("REGRESSED" if after < before else "UNCHANGED"),
            "promoted_to_strong": current_status == "STRONG_BEST" and previous_status != "STRONG_BEST",
        })
    return rows


def _num(value: Any) -> str:
    return "null" if value is None else f"{float(value):.4f}"


def write_report(v2_path: Path, v3_path: Path, out_dir: Path) -> dict[str, Any]:
    v2 = json.loads(v2_path.read_text(encoding="utf-8"))
    v3 = json.loads(v3_path.read_text(encoding="utf-8"))
    rows = compare_best(v2, v3)
    token = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    paths = {
        "md": out_dir / f"aegis_short_v3_feature_optimization_comparison_{token}.md",
        "json": out_dir / f"aegis_short_v3_feature_optimization_comparison_{token}.json",
        "csv": out_dir / f"aegis_short_v3_feature_optimization_comparison_{token}.csv",
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0].keys()) if rows else ["symbol", "v2_status", "v3_status", "status_change"]
    with paths["csv"].open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    report = {
        "schema_version": "aegis_short_v2_v3_optimization_comparison_v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "mode": "RESEARCH_ONLY",
        "v2_report": str(v2_path),
        "v3_report": str(v3_path),
        "rows": rows,
        "promoted_to_strong": [row["symbol"] for row in rows if row["promoted_to_strong"]],
        "regressions": [row["symbol"] for row in rows if row["status_change"] == "REGRESSED"],
        "paths": {name: str(path) for name, path in paths.items()},
    }
    paths["json"].write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    lines = [
        "# Aegis SHORT V2 vs V3 Feature Optimization",
        "",
        "- Mode: `RESEARCH_ONLY`",
        "- No model artifacts, active manifest or live inference changes.",
        "",
        "| Symbol | V2 Best | V3 Best | Change | V3 Set | Window | H | Hit8 AUC | Hit8 Lift | Quality Lift | Latest Quality |",
        "|---|---|---|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['symbol']} | {row['v2_status']} | {row['v3_status']} | {row['status_change']} | "
            f"{row['v3_feature_set']} | {row['v3_lookback_days']} | {row['v3_horizon_candles']} | "
            f"{_num(row['v3_hit8_auc'])} | {_num(row['v3_hit8_lift'])} | {_num(row['v3_quality_lift'])} | "
            f"{_num(row['v3_latest_quality_lift'])} |"
        )
    lines.extend([
        "",
        f"- Promoted to STRONG: `{report['promoted_to_strong']}`.",
        f"- Regressions: `{report['regressions']}`.",
        "",
    ])
    paths["md"].write_text("\n".join(lines), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare research-only SHORT optimization reports for feature V2 versus V3.")
    parser.add_argument("--v2-report", required=True)
    parser.add_argument("--v3-report", required=True)
    parser.add_argument("--out-dir", default="/home/jasan/Develop")
    args = parser.parse_args()
    report = write_report(Path(args.v2_report), Path(args.v3_report), Path(args.out_dir))
    print(json.dumps({
        "paths": report["paths"],
        "promoted_to_strong": report["promoted_to_strong"],
        "regressions": report["regressions"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
