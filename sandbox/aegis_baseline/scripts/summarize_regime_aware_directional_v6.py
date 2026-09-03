#!/usr/bin/env python3
"""Render compact, non-private v6 validation evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

from aegis.research.regime_aware_directional_v6_gate import assess_shadow_gate
from aegis.utils import Sha256HashProvider, sha256_file


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be an object")
    return value


def compact_summary(report: Mapping[str, Any]) -> Mapping[str, Any]:
    gate = assess_shadow_gate(report)
    sides: dict[str, Any] = {}
    for side in ("LONG", "SHORT"):
        source = _mapping(_mapping(report["sides"], "sides")[side], side)
        folds = []
        for fold in source["folds"]:
            fold = _mapping(fold, "fold")
            folds.append(
                {
                    "fold": fold["fold"],
                    "status": fold["status"],
                    "passed": fold["passed"],
                    "baseline": fold.get("baseline"),
                    "current_brain_control": fold.get("current_brain_control"),
                    "probabilistic_regime_router": fold.get(
                        "probabilistic_regime_router"
                    ),
                    "selected": fold.get("selected"),
                    "opportunity_gap_gate_passed": fold.get(
                        "opportunity_gap_gate_passed"
                    ),
                    "diagnostic_ablations": fold.get("diagnostic_ablations", {}),
                }
            )
        sides[side] = {
            "rows": source["rows"],
            "evaluated_folds": source["evaluated_folds"],
            "passing_folds": source["passing_folds"],
            "regime_router_skilled_folds": source["regime_router_skilled_folds"],
            "minimum_regime_router_skilled_folds": source[
                "minimum_regime_router_skilled_folds"
            ],
            "worst_fold_non_negative": source["worst_fold_non_negative"],
            "leave_one_symbol_out": source["leave_one_symbol_out"],
            "validation_pass": source["validation_pass"],
            "verdict": source["verdict"],
            "folds": folds,
        }
    ablations: dict[str, Any] = {}
    for side in ("LONG", "SHORT"):
        grouped: dict[str, list[Mapping[str, Any]]] = {}
        for fold in sides[side]["folds"]:
            for variant, value in fold["diagnostic_ablations"].items():
                grouped.setdefault(variant, []).append(value["metrics"])
        rolled = []
        for variant, values in grouped.items():
            count = sum(int(value["count"]) for value in values)
            weighted_net = (
                sum(
                    int(value["count"]) * float(value["mean_protected_net"])
                    for value in values
                    if int(value["count"]) > 0
                )
                / count
                if count
                else None
            )
            rolled.append(
                {
                    "variant": variant,
                    "selected": count,
                    "positive_folds": sum(
                        value["mean_protected_net"] is not None
                        and float(value["mean_protected_net"]) > 0.0
                        for value in values
                    ),
                    "weighted_protected_net": weighted_net,
                }
            )
        ablations[side] = sorted(
            rolled,
            key=lambda value: (
                value["weighted_protected_net"] is None,
                (
                    -float(value["weighted_protected_net"])
                    if value["weighted_protected_net"] is not None
                    else 0.0
                ),
            ),
        )
    summary = {
        "schema_id": "aegis-regime-aware-directional-v6-summary-v1",
        "experiment_id": report["experiment_id"],
        "evidence_start": report["source_evidence_start"],
        "evidence_end": report["source_evidence_end"],
        "validation_content_hash": report["content_hash"],
        "validation_pass": report["validation_pass"],
        "verdict": report["verdict"],
        "shadow_gate": {
            "eligible": gate.eligible,
            "status": gate.status,
            "blockers": list(gate.blockers),
        },
        "sides": sides,
        "diagnostic_ablation_rollup": ablations,
        "runtime_effect": "NONE",
        "exchange_calls": report["exchange_calls"],
        "exchange_mutations": report["exchange_mutations"],
        "content_hash": "",
    }
    summary["content_hash"] = Sha256HashProvider().digest_value(
        {**summary, "content_hash": ""}
    )
    return summary


def external_controls(root: Path) -> Mapping[str, Any]:
    definitions = {
        "LONG_V4": (
            Path("data/long_entry_v4_shadow/validation.json"),
            ("tournament", "verdict"),
        ),
        "LONG_V5": (
            Path("data/long_entry_v5_shadow/validation.json"),
            ("validation", "verdict"),
        ),
        "LONG_V51_ABLATION": (
            Path("data/long_entry_v51_ablation_shadow/validation.json"),
            ("verdict",),
        ),
    }
    result: dict[str, Any] = {}
    for identity, (relative, keys) in definitions.items():
        path = root / relative
        payload: Any = json.loads(path.read_text())
        for key in keys:
            payload = payload[key]
        result[identity] = {
            "path": str(relative),
            "sha256": sha256_file(path),
            "verdict": payload,
            "comparison_class": "EXTERNAL_REFERENCE_DIFFERENT_POPULATION",
        }
    return result


def _percent(value: Any) -> str:
    return "N/A" if value is None else f"{float(value) * 100:+.4f}%"


def markdown(summary: Mapping[str, Any], validation_sha256: str) -> str:
    lines = [
        "# Aegis Regime-Aware Directional V6 Validation",
        "",
        f"- Experiment: `{summary['experiment_id']}`",
        f"- Evidence: `{summary['evidence_start']}` to `{summary['evidence_end']}`",
        f"- Validation file SHA-256: `{validation_sha256}`",
        f"- Historical verdict: `{summary['verdict']}`",
        f"- Shadow gate: `{summary['shadow_gate']['status']}`",
        f"- Runtime effect: `{summary['runtime_effect']}`",
        f"- Exchange calls: `{summary['exchange_calls']}`",
        f"- Exchange mutations: `{summary['exchange_mutations']}`",
        "",
        "## Directional Results",
        "",
        "| Side | Rows | Evaluated folds | Passing folds | Worst fold non-negative | Validation |",
        "|---|---:|---:|---:|---|---|",
    ]
    for side in ("LONG", "SHORT"):
        row = summary["sides"][side]
        lines.append(
            f"| {side} | {row['rows']} | {row['evaluated_folds']} | "
            f"{row['passing_folds']} | {row['worst_fold_non_negative']} | "
            f"{row['validation_pass']} |"
        )
    lines.extend(
        [
            "",
            "## Fold Outcomes",
            "",
            "| Side | Fold | Selected | Protected net | Mean MAE | Protectable | P95 opportunity gap | Router skilled |",
            "|---|---:|---:|---:|---:|---:|---:|---|",
        ]
    )
    for side in ("LONG", "SHORT"):
        for fold in summary["sides"][side]["folds"]:
            selected = fold["selected"]
            router = fold["probabilistic_regime_router"]
            lines.append(
                f"| {side} | {fold['fold']} | {selected['count']} | "
                f"{_percent(selected['mean_protected_net'])} | "
                f"{_percent(selected['mean_mae'])} | "
                f"{_percent(selected['protectable_rate'])} | "
                f"{selected['p95_gap_hours']}h | {router['passed']} |"
            )
    lines.extend(["", "## External Controls", ""])
    for identity, control in summary.get("external_controls", {}).items():
        lines.append(
            f"- `{identity}`: `{control['verdict']}` at `{control['path']}` "
            f"(SHA-256 `{control['sha256']}`)."
        )
    lines.extend(
        [
            "",
            "The external controls use different historical populations and are references,",
            "not direct head-to-head estimates. The current-brain control is embedded in each",
            "fold of the JSON summary.",
            "",
            "## Interpretation",
            "",
        ]
    )
    for side in ("LONG", "SHORT"):
        best = summary["diagnostic_ablation_rollup"][side][0]
        lines.append(
            f"- `{side}` best diagnostic ablation: `{best['variant']}` with "
            f"{best['selected']} selections, `{best['positive_folds']}/4` positive "
            f"folds, and {_percent(best['weighted_protected_net'])} weighted protected net."
        )
    lines.extend(
        [
            "- The regime router showed skill in `2/4` folds, below the frozen `3/4` requirement.",
            "- Low predicted MAE reduced adverse excursion but did not establish positive direction or net edge.",
            "- V6 therefore remains research-only; thresholds were not relaxed after observing test outcomes.",
        ]
    )
    lines.extend(["", "## Gate Blockers", ""])
    blockers = summary["shadow_gate"]["blockers"]
    lines.extend(f"- `{value}`" for value in blockers)
    if not blockers:
        lines.append(
            "- None. Prospective Shadow still requires separate authorization."
        )
    lines.extend(
        [
            "",
            "This report does not activate Shadow, export a model, alter Live selection,",
            "or authorize exchange activity.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--validation",
        type=Path,
        default=Path("data/regime_aware_directional_v6/validation.json"),
    )
    parser.add_argument(
        "--json-output",
        type=Path,
        default=Path(
            "reports/governance/aegis_prospective_validation/live/"
            "regime_aware_directional_v6/validation_summary.json"
        ),
    )
    parser.add_argument(
        "--markdown-output",
        type=Path,
        default=Path(
            "reports/governance/aegis_prospective_validation/live/"
            "regime_aware_directional_v6/validation_summary.md"
        ),
    )
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    validation = (
        args.validation if args.validation.is_absolute() else root / args.validation
    )
    json_output = (
        args.json_output if args.json_output.is_absolute() else root / args.json_output
    )
    markdown_output = (
        args.markdown_output
        if args.markdown_output.is_absolute()
        else root / args.markdown_output
    )
    report = _mapping(json.loads(validation.read_text()), "validation")
    summary = dict(compact_summary(report))
    summary["external_controls"] = external_controls(root)
    summary["content_hash"] = ""
    summary["content_hash"] = Sha256HashProvider().digest_value(summary)
    json_output.parent.mkdir(parents=True, exist_ok=True)
    json_output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    markdown_output.write_text(markdown(summary, sha256_file(validation)))
    print(json.dumps(summary["shadow_gate"], sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
