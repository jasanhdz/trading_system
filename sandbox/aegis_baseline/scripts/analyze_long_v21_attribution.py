#!/usr/bin/env python3
"""Summarize LONG v2.1 attribution without refitting or changing policy."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Mapping


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def build_attribution(report: Mapping[str, Any]) -> Mapping[str, Any]:
    validation = report["validation"]
    archetypes: dict[str, Any] = {}
    for name, result in validation["archetypes"].items():
        folds = [row for row in result["folds"] if row["status"] == "EVALUATED"]
        metrics = [row["metrics"] for row in folds]
        archetypes[name] = {
            "evaluated_folds": len(folds),
            "positive_net_folds": sum(
                float(row["selected_protected_worst_net"]) > 0.0 for row in metrics
            ),
            "mae_improved_folds": sum(
                float(row["selected_mae"]) < float(row["baseline_mae"])
                for row in metrics
            ),
            "underwater_improved_folds": sum(
                float(row["selected_underwater_bars"])
                < float(row["baseline_underwater_bars"])
                for row in metrics
            ),
            "mean_selected_protected_net": _mean(
                [float(row["selected_protected_worst_net"]) for row in metrics]
            ),
            "mean_utility_correlation": _mean(
                [float(row["utility_prediction_correlation"]) for row in metrics]
            ),
            "promotable": bool(result["passed"]),
        }

    symbols: dict[str, Any] = {}
    loso_symbols = validation["leave_one_symbol_out"]["symbols"]
    for symbol, result in loso_symbols.items():
        metrics = result.get("metrics", {})
        symbols[symbol] = {
            "status": result["status"],
            "generalized_without_regression": bool(
                result.get("generalized_without_regression", False)
            ),
            "policy_valid_without_symbol": bool(
                result.get("policy_valid_without_symbol", False)
            ),
            "baseline_protected_net": metrics.get("baseline_protected_worst_net"),
            "selected_protected_net": metrics.get("selected_protected_worst_net"),
            "baseline_mae": metrics.get("baseline_mae"),
            "selected_mae": metrics.get("selected_mae"),
            "selected_rows": metrics.get("selected_rows", 0),
        }

    return {
        "schema_id": "aegis-long-entry-v21-attribution-v1",
        "source_validation_verdict": validation["verdict"],
        "source_validation_pass": bool(validation["validation_pass"]),
        "archetypes": archetypes,
        "symbols": symbols,
        "symbols_without_regression": validation["leave_one_symbol_out"][
            "symbols_without_regression"
        ],
        "diagnosis": {
            "ranking_signal": "WEAK",
            "mae_reduction": "PRESENT_BUT_INSUFFICIENT",
            "protected_net_edge": "NOT_DEMONSTRATED",
            "regime_attribution": "REQUIRES_ROW_LEVEL_RECOMPUTATION_IN_V22",
            "threshold_relaxation_authorized": False,
        },
        "selection_effect": "NONE",
        "live_effect": "NONE",
        "exchange_mutations": 0,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/long_entry_v21_shadow/validation.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/long_entry_v21_shadow/attribution.json"),
    )
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    source = args.input if args.input.is_absolute() else root / args.input
    output = args.output if args.output.is_absolute() else root / args.output
    report = json.loads(source.read_text(encoding="utf-8"))
    attribution = build_attribution(report)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(
        json.dumps(attribution, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.chmod(temporary, 0o600)
    temporary.replace(output)
    print(json.dumps({"output": str(output), "diagnosis": attribution["diagnosis"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
