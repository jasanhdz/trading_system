#!/usr/bin/env python3
"""Run frozen Directional Alpha V1 simple models; never open holdout."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd


EXPERIMENT = Path(__file__).resolve().parents[1]
SANDBOX = EXPERIMENT.parents[1]
REPOSITORY = SANDBOX.parents[1]
for path in (EXPERIMENT / "src", SANDBOX / "src", REPOSITORY / "src"):
    sys.path.insert(0, str(path))

from directional_alpha_v1.features import assert_allowlist  # noqa: E402
from directional_alpha_v1.modeling import run  # noqa: E402


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=EXPERIMENT / "config/preregistration_v1.json")
    parser.add_argument("--dataset", type=Path, default=EXPERIMENT / "artifacts/dataset_v1")
    parser.add_argument("--output", type=Path, default=EXPERIMENT / "artifacts/run_01")
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    frame = pd.read_parquet(args.dataset / "development_labeled.parquet")
    dictionary = json.loads((args.dataset / "feature_dictionary.json").read_text())
    features = [item["name"] for item in dictionary["features"]]
    assert_allowlist(features)
    holdout = pd.read_parquet(args.dataset / "final_holdout_features_sealed.parquet")
    prohibited = [column for column in frame if any(token in column.lower() for token in ("aegis", "candidate_strategy", "committee"))]
    audit = {
        "future_availability_rows": int((pd.to_datetime(frame.max_feature_available_at, utc=True) > pd.to_datetime(frame.decision_at, utc=True)).sum()),
        "prohibited_columns": prohibited, "holdout_target_columns": [column for column in holdout if column.startswith("target__")],
        "directional_features": len(features), "passed": False,
    }
    audit["passed"] = audit["future_availability_rows"] == 0 and not prohibited and not audit["holdout_target_columns"]
    if not audit["passed"]:
        raise RuntimeError("LEAKAGE_AUDIT_FAILED")
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "leakage_report.json").write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")
    split = {
        "frozen_splits": config["splits"], "rows": frame.split.value_counts().sort_index().to_dict(),
        "states": frame.groupby("split").market_state_group_id.nunique().to_dict(),
        "primary_population_rows": frame.loc[frame.opportunity_top_90].split.value_counts().sort_index().to_dict(),
        "holdout_rows": len(holdout), "holdout_labels_built": False,
    }
    (args.output / "split_manifest.json").write_text(json.dumps(split, indent=2, sort_keys=True) + "\n")
    (args.output / "l2_only_result.json").write_text(json.dumps({
        "status": "NOT_RUN_NO_CLEAN_ELIGIBLE_L2_PERIOD",
        "signal_found": False,
        "reason": "Available valid Tardis days are discovery-contaminated or overlap sealed prior holdouts.",
    }, indent=2, sort_keys=True) + "\n")
    (args.output / "positioning_source_audit.json").write_text(json.dumps({
        "status": "NOT_RUN_INSUFFICIENT_AUTHENTIC_COVERAGE",
        "open_interest": "MISSING_BROAD_CAUSAL_HISTORY",
        "liquidations": "MISSING_BROAD_CLEAN_CAUSAL_HISTORY",
        "funding": "AVAILABLE_BUT_NOT_USED_WITHOUT_MATCHED_POSITIONING_SUBEXPERIMENT",
    }, indent=2, sort_keys=True) + "\n")
    result = run(dataset=frame, dictionary=dictionary, config=config, output=args.output)
    result["dataset_manifest_sha256"] = sha256(args.dataset / "dataset_manifest.json")
    result["leakage_report_sha256"] = sha256(args.output / "leakage_report.json")
    (args.output / "result.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    (args.output / "final_report.md").write_text(report(result))
    write_manifest(args, result)
    print(json.dumps({"verdict": result["verdict"], "flags": result["flags"]}, indent=2, sort_keys=True))
    return 0


def write_manifest(args: argparse.Namespace, result: dict) -> None:
    sources = sorted((EXPERIMENT / "src").rglob("*.py")) + sorted((EXPERIMENT / "tools").glob("*.py"))
    inputs = [args.config, args.dataset / "dataset_manifest.json", args.dataset / "feature_dictionary.json", *sources]
    outputs = [path for path in sorted(args.output.iterdir()) if path.is_file() and path.name != "artifact_manifest.json" and path.suffix != ".joblib"]
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPOSITORY, check=True, capture_output=True, text=True).stdout.strip()
    manifest = {
        "schema": "directional-alpha-v1-artifact-manifest", "git_commit": commit,
        "final_holdout_state": result["final_holdout_state"],
        "inputs": {str(path.relative_to(REPOSITORY)): sha256(path) for path in inputs},
        "outputs": {str(path.relative_to(REPOSITORY)): sha256(path) for path in outputs},
        "rerun": [
            "PYTHONPATH=src .venv/bin/python sandbox/aegis_strategy_router/experiments/directional_alpha_v1/tools/download_sources.py",
            "PYTHONPATH=src .venv/bin/python sandbox/aegis_strategy_router/experiments/directional_alpha_v1/tools/prepare_sources.py",
            "PYTHONPATH=sandbox/aegis_strategy_router/src:src:sandbox/aegis_strategy_router/experiments/independent_entry_quality_discovery_v1/src:sandbox/aegis_strategy_router/experiments/directional_alpha_v1/src .venv/bin/python sandbox/aegis_strategy_router/experiments/directional_alpha_v1/tools/build_dataset.py",
            "PYTHONPATH=sandbox/aegis_strategy_router/src:src:sandbox/aegis_strategy_router/experiments/directional_alpha_v1/src .venv/bin/python sandbox/aegis_strategy_router/experiments/directional_alpha_v1/tools/run_models.py"
        ],
    }
    (args.output / "artifact_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")


def report(result: dict) -> str:
    primary, baseline = result["primary"], result["opportunity_comparison"]["opportunity_only_same_period"]
    cards = {row["family"]: row for row in result["model_cards"]}
    lines = [
        "# Directional Alpha V1 Result", "", f"Verdict: `{result['verdict']}`.", "",
        "FINAL_HOLDOUT remains `SEALED_NOT_OPENED`.", "", "## Support", "",
        f"- TRAIN primary rows: {result['support']['train_rows']:,}.",
        f"- VALIDATION primary rows: {result['support']['validation_rows']:,}; required: 2,500.", "",
        "## Direction models", "",
    ]
    lines.extend(f"- {name}: AUC {card['roc_auc']:.4f}, FDR q {card['fdr_q']:.4g}." for name, card in cards.items())
    lines.extend([
        "", "## Primary top-10% directional confidence", "",
        f"- N: {int(primary['n']):,}; effective blocks: {int(primary['effective_blocks']):,}.",
        f"- Favorable-first: {primary['favorable_first']:.2%} vs {baseline['favorable_first']:.2%} Opportunity-only.",
        f"- MFE/MAE: {primary['mfe_bps']:.2f}/{primary['mae_bps']:.2f} bps; ratio {primary['mfe_mae_ratio']:.3f}.",
        f"- Gross/net: {primary['gross_bps']:.2f}/{primary['net_bps']:.2f} bps.",
        f"- Net CI: [{primary['net_ci_lower_bps']:.2f}, {primary['net_ci_upper_bps']:.2f}] bps.", "",
        "## Frozen Opportunity population diagnostics", "",
    ])
    lines.extend(
        f"- {row['population']}: {int(row['population_states']):,} states; selected net {row['net_bps']:.2f} bps."
        for row in result["opportunity_population_diagnostics"]
    )
    lines.extend([
        "",
        "## Flags", "",
    ])
    lines.extend(f"- `{name} = {value}`" for name, value in result["flags"].items())
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    raise SystemExit(main())
