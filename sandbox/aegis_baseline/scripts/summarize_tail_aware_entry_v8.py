#!/usr/bin/env python3
"""Create compact tracked evidence from private V8 validation output."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

from aegis.research.tail_aware_entry_v8_gate import evaluate_v8_gate
from aegis.utils import Sha256HashProvider, sha256_file


def _fold_summary(fold: Mapping[str, Any]) -> Mapping[str, Any]:
    if fold.get("status") != "EVALUATED":
        return {
            "fold": fold.get("fold"),
            "status": fold.get("status"),
            "passed": False,
        }
    return {
        "fold": fold["fold"],
        "status": fold["status"],
        "selected": fold["selected"],
        "control_identity": fold["control_identity"],
        "control": fold["control"],
        "regime_router": fold["regime_router"],
        "late_entry_detector": fold["late_entry_detector"],
        "stress_net_bootstrap": fold["stress_net_bootstrap"],
        "passed": fold["passed"],
    }


def summarize(
    validation: Mapping[str, Any], validation_sha256: str
) -> Mapping[str, Any]:
    gate = evaluate_v8_gate(validation)
    sides = {
        side: {
            "rows": value["rows"],
            "folds": [_fold_summary(fold) for fold in value["folds"]],
            "passing_folds": value["passing_folds"],
            "router_skilled_folds": value["router_skilled_folds"],
            "late_detector_skilled_folds": value["late_detector_skilled_folds"],
            "worst_fold_non_negative": value["worst_fold_non_negative"],
            "primary_gate": value["primary_gate"],
            "leave_one_symbol_out": value["leave_one_symbol_out"],
            "validation_pass": value["validation_pass"],
        }
        for side, value in validation["sides"].items()
    }
    result = {
        "schema_id": "aegis-tail-aware-entry-v8-validation-summary-v1",
        "experiment_id": validation["experiment_id"],
        "private_validation_sha256": validation_sha256,
        "dataset_sha256": validation["dataset_sha256"],
        "config_sha256": validation["config_sha256"],
        "evidence_start": validation["evidence_start"],
        "evidence_end": validation["evidence_end"],
        "sides": sides,
        "validation_pass": validation["validation_pass"],
        "verdict": validation["verdict"],
        "gate": gate,
        "runtime_effect": "NONE",
        "shadow_runtime_enabled": False,
        "model_exported": False,
        "exchange_calls": 0,
        "exchange_mutations": 0,
        "content_hash": "",
    }
    result["content_hash"] = Sha256HashProvider().digest_value(result)
    return result


def render_markdown(summary: Mapping[str, Any]) -> str:
    lines = [
        "# Tail-aware Entry V8 Validation",
        "",
        f"- Verdict: `{summary['verdict']}`",
        f"- Promotion gate: `{'PASS' if summary['gate']['passed'] else 'FAIL'}`",
        "- Runtime effect: `NONE`",
        "- Model exported: `false`",
        "- Exchange calls/mutations: `0/0`",
        "",
    ]
    for side, value in summary["sides"].items():
        lines.extend(
            [
                f"## {side}",
                "",
                f"- Passing folds: `{value['passing_folds']}/4`",
                f"- Skilled regime-router folds: `{value['router_skilled_folds']}/4`",
                f"- Skilled late-detector folds: `{value['late_detector_skilled_folds']}/4`",
                f"- Worst fold non-negative: `{str(value['worst_fold_non_negative']).lower()}`",
                f"- LOSO: `{value['leave_one_symbol_out']['status']}`",
                "",
                "| Fold | Selected | Stress mean | Stress CVaR | Payoff | Mean MAE | Control stress | Router | Late detector | Pass |",
                "|---:|---:|---:|---:|---:|---:|---:|:---:|:---:|:---:|",
            ]
        )
        for fold in value["folds"]:
            if fold["status"] != "EVALUATED":
                lines.append(
                    f"| {fold['fold']} | 0 | n/a | n/a | n/a | n/a | n/a | FAIL | FAIL | FAIL |"
                )
                continue
            selected = fold["selected"]
            control = fold["control"]
            number = lambda value: "n/a" if value is None else f"{float(value):.6f}"
            lines.append(
                f"| {fold['fold']} | {selected['count']} | {number(selected['mean_stress_net'])} | "
                f"{number(selected['stress_cvar'])} | {number(selected['payoff_ratio'])} | "
                f"{number(selected['mean_mae'])} | {number(control['mean_stress_net'])} | "
                f"{'PASS' if fold['regime_router']['passed'] else 'FAIL'} | "
                f"{'PASS' if fold['late_entry_detector']['passed'] else 'FAIL'} | "
                f"{'PASS' if fold['passed'] else 'FAIL'} |"
            )
        lines.append("")
    lines.extend(
        [
            "## Decision",
            "",
            f"`{summary['gate']['decision']}`",
            "",
            "This result cannot alter Shadow or Live without separate authorization.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--validation",
        type=Path,
        default=Path("data/tail_aware_entry_v8/validation.json"),
    )
    parser.add_argument(
        "--json-output",
        type=Path,
        default=Path(
            "reports/governance/aegis_prospective_validation/live/tail_aware_entry_v8/validation_summary.json"
        ),
    )
    parser.add_argument(
        "--markdown-output",
        type=Path,
        default=Path(
            "reports/governance/aegis_prospective_validation/live/tail_aware_entry_v8/validation_summary.md"
        ),
    )
    args = parser.parse_args()
    validation = json.loads(args.validation.read_text())
    summary = summarize(validation, sha256_file(args.validation))
    args.json_output.parent.mkdir(parents=True, exist_ok=True)
    args.json_output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    args.markdown_output.write_text(render_markdown(summary))
    print(
        json.dumps(
            {"verdict": summary["verdict"], "gate": summary["gate"]},
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
