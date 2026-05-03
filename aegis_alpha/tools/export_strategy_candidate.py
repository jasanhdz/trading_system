#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DEFAULT_SOURCE_REPORT = Path("aegis_alpha/logs/edge/long_edge_dynamic_sizing_grid_20260503T041147Z.json")
DEFAULT_OUTPUT = Path("aegis_alpha/models/strategy_candidates/aegis_long_edge_dynamic_v042.json")


def _load_best_config(source_report: Path) -> dict[str, Any]:
    report = json.loads(source_report.read_text(encoding="utf-8"))
    best = report.get("best_config") or (report.get("ranking") or [None])[0]
    if not best:
        raise RuntimeError(f"No best config found in {source_report}")
    return report, best


def export_strategy_candidate(source_report: Path, output_path: Path) -> Path:
    report, best = _load_best_config(source_report)
    candidate = {
        "schema_version": "aegis_strategy_candidate_v1",
        "created_at": datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
        "status": "OFFLINE_CANDIDATE",
        "candidate_id": "aegis_long_edge_dynamic_v042",
        "source_experiment": "Aegis Alpha v0.4.2",
        "source_report_path": str(source_report),
        "edge_model_path": "aegis_alpha/models/edge/aegis_edge_model_v030.joblib",
        "meta_filter_path": "aegis_alpha/models/edge/aegis_long_edge_meta_filter_v040.joblib",
        "policy": {
            "side": "LONG_ONLY",
            "entry_gate": "top_3pct_expected_return_long",
            "gate_threshold": report.get("policy", {}).get("gate_threshold"),
            "allowed_regimes": ["mixed", "chop", "high_vol"],
            "risk_guard": "loss7_pause48_pause2_48_maxday3",
            "dynamic_sizing": {
                "full_size": 0.25,
                "reduced_size": 0.125,
                "meta_high_threshold": 0.60,
                "meta_low_threshold": None,
                "fee_multiplier": 1.0,
            },
            "short_entries": False,
        },
        "freeze": {
            "best_config_id": best.get("config_id"),
            "median_balance": best.get("median_balance"),
            "p25_balance": best.get("p25_balance"),
            "worst_balance": best.get("worst_balance"),
            "median_pf": best.get("median_pf"),
            "p25_pf": best.get("p25_pf"),
            "profitable_window_pct": best.get("profitable_window_pct"),
            "median_trades": best.get("median_trades"),
            "worst_max_dd": best.get("worst_max_dd"),
        },
        "notes": {
            "from_report": report.get("created_at"),
            "selection_basis": "best ranked config from v0.4.2 dynamic sizing grid",
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(candidate, indent=2, sort_keys=True), encoding="utf-8")
    print(f"Candidate exported -> {output_path}")
    print(f"Source report -> {source_report}")
    print(f"Best config -> {best.get('config_id')}")
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-report", default=str(DEFAULT_SOURCE_REPORT))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    args = parser.parse_args()
    export_strategy_candidate(Path(args.source_report), Path(args.output))


if __name__ == "__main__":
    main()
