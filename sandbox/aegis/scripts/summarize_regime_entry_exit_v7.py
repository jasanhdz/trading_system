#!/usr/bin/env python3
"""Create compact tracked V7 validation reports from private evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

from aegis.research.regime_entry_exit_v7_gate import evaluate_v7_gate
from aegis.utils import sha256_file


def compact_summary(
    validation: Mapping[str, Any], manifest: Mapping[str, Any], validation_sha256: str
) -> Mapping[str, Any]:
    gate = evaluate_v7_gate(validation)
    sides = {}
    for side, result in validation["sides"].items():
        folds = []
        for fold in result["folds"]:
            folds.append(
                {
                    "fold": fold["fold"],
                    "status": fold["status"],
                    "selected": fold.get("selected"),
                    "control_identity": fold.get("control_identity"),
                    "control": fold.get("control"),
                    "regime_router": fold.get("regime_router"),
                    "trained_archetypes": fold.get("trained_archetypes", []),
                    "test_abstained": fold.get("test_abstained"),
                    "diagnostic_ablations": fold.get("diagnostic_ablations", {}),
                    "passed": fold.get("passed", False),
                }
            )
        sides[side] = {
            "rows": result["rows"],
            "evaluated_folds": result["evaluated_folds"],
            "passing_folds": result["passing_folds"],
            "router_skilled_folds": result["router_skilled_folds"],
            "worst_fold_non_negative": result["worst_fold_non_negative"],
            "leave_one_symbol_out": result["leave_one_symbol_out"],
            "validation_pass": result["validation_pass"],
            "folds": folds,
        }
    return {
        "schema_id": "aegis-regime-entry-exit-v7-compact-summary-v1",
        "experiment_id": validation["experiment_id"],
        "evidence_start": validation["evidence_start"],
        "evidence_end": validation["evidence_end"],
        "dataset_sha256": validation["dataset_sha256"],
        "dataset_manifest_sha256": validation["dataset_manifest_sha256"],
        "validation_sha256": validation_sha256,
        "trajectory_responsibility_counts": manifest[
            "trajectory_responsibility_counts"
        ],
        "hindsight_best_profile_counts": manifest["hindsight_best_profile_counts"],
        "current_protection_replay_mismatches": manifest[
            "current_protection_replay_mismatches"
        ],
        "sides": sides,
        "verdict": validation["verdict"],
        "gate": gate,
        "runtime_effect": "NONE",
        "exchange_calls": 0,
        "exchange_mutations": 0,
    }


def markdown(summary: Mapping[str, Any]) -> str:
    lines = [
        "# Aegis Regime Entry/Exit V7 Validation",
        "",
        f"- Experiment: `{summary['experiment_id']}`",
        f"- Evidence: `{summary['evidence_start']}` to `{summary['evidence_end']}`",
        f"- Verdict: `{summary['verdict']}`",
        f"- Gate: `{summary['gate']['decision']}`",
        "- Runtime effect: `NONE`",
        "- Exchange calls: `0`",
        "- Exchange mutations: `0`",
        "",
        "## Trajectory attribution",
        "",
    ]
    for name, count in summary["trajectory_responsibility_counts"].items():
        lines.append(f"- `{name}`: {count}")
    lines.extend(
        [
            "",
            "## Directional folds",
            "",
            "| Side | Fold | Selected | Net | MAE | Capture | P95 gap | Passed |",
            "|---|---:|---:|---:|---:|---:|---:|---|",
        ]
    )
    for side, result in summary["sides"].items():
        for fold in result["folds"]:
            metrics = fold.get("selected") or {}
            if fold["status"] != "EVALUATED":
                lines.append(
                    f"| {side} | {fold['fold']} | 0 | n/a | n/a | n/a | n/a | False |"
                )
                continue
            lines.append(
                f"| {side} | {fold['fold']} | {metrics['count']} | "
                f"{metrics['mean_net']:.4%} | {metrics['mean_mae']:.4%} | "
                f"{metrics['mean_capture_efficiency']:.2%} | "
                f"{metrics['p95_gap_hours']:.1f}h | {fold['passed']} |"
            )
    lines.extend(["", "## Interpretation", ""])
    for side, result in summary["sides"].items():
        evaluated = [fold for fold in result["folds"] if fold["status"] == "EVALUATED"]
        improved = sum(
            float(fold["selected"]["mean_net"]) > float(fold["control"]["mean_net"])
            for fold in evaluated
        )
        lines.append(
            f"- `{side}` improved net loss relative to its frozen control in "
            f"`{improved}/{len(evaluated)}` folds, but had `0` positive folds."
        )
    lines.extend(
        [
            "- The regime router was skilled in `2/4` folds for each side, below",
            "  the frozen `3/4` requirement.",
            "- Lower MAE and faster paths reduced some losses but did not establish",
            "  positive expectancy after costs.",
            "- Leave-one-symbol-out was not run because the primary historical gate",
            "  failed; running it could not make this version promotion-eligible.",
            "- Protection profile counts are hindsight diagnostics only and are not",
            "  evidence that one fixed profile should be deployed.",
            "",
            "## Hindsight protection profile counts",
            "",
        ]
    )
    lines.extend(
        f"- `{name}`: {count}"
        for name, count in summary["hindsight_best_profile_counts"].items()
    )
    lines.extend(["", "## Gate blockers", ""])
    lines.extend(f"- `{value}`" for value in summary["gate"]["blockers"])
    lines.extend(
        [
            "",
            "V7 remains research-only unless every frozen gate passes. This report does",
            "not activate Shadow, alter Live selection, export a model, or authorize",
            "exchange activity.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--validation",
        type=Path,
        default=Path("data/regime_entry_exit_v7/validation.json"),
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/regime_entry_exit_v7/dataset_manifest.json"),
    )
    parser.add_argument(
        "--json-output",
        type=Path,
        default=Path(
            "reports/governance/aegis_prospective_validation/live/"
            "regime_entry_exit_v7/validation_summary.json"
        ),
    )
    parser.add_argument(
        "--markdown-output",
        type=Path,
        default=Path(
            "reports/governance/aegis_prospective_validation/live/"
            "regime_entry_exit_v7/validation_summary.md"
        ),
    )
    args = parser.parse_args()
    validation = json.loads(args.validation.read_text())
    manifest = json.loads(args.manifest.read_text())
    summary = compact_summary(validation, manifest, sha256_file(args.validation))
    args.json_output.parent.mkdir(parents=True, exist_ok=True)
    args.json_output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    args.markdown_output.write_text(markdown(summary))
    print(json.dumps(summary["gate"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
