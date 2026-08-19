#!/usr/bin/env python3
"""Fit frozen simple models on TRAIN/CALIBRATION and evaluate VALIDATION."""

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

from independent_entry_quality_v1.features import assert_feature_allowlist  # noqa: E402
from independent_entry_quality_v1.modeling import run_experiment  # noqa: E402


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
    config = json.loads(args.config.read_text(encoding="utf-8"))
    manifest = json.loads((args.dataset / "dataset_manifest.json").read_text(encoding="utf-8"))
    frame = pd.read_parquet(args.dataset / "development_labeled.parquet")
    dictionary = json.loads((args.dataset / "feature_dictionary.json").read_text(encoding="utf-8"))
    features = [item["name"] for item in dictionary["features"]]
    assert_feature_allowlist(features)
    if not set(frame.split.unique()).issubset({"TRAIN", "CALIBRATION", "VALIDATION"}):
        raise RuntimeError("NON_DEVELOPMENT_SPLIT_IN_LABELED_DATASET")
    availability = pd.to_datetime(frame.max_feature_available_at, utc=True)
    decision = pd.to_datetime(frame.decision_at, utc=True)
    leakage_rows = int((availability > decision).sum())
    prohibited = [
        column for column in frame.columns
        if any(token in column.lower() for token in ("aegis", "candidate_strategy", "committee", "confidence", "enter_wait_skip"))
    ]
    holdout = pd.read_parquet(args.dataset / "final_holdout_features_sealed.parquet")
    holdout_target_columns = [column for column in holdout if column.startswith("target__")]
    leakage_audit = {
        "schema": "independent-entry-quality-leakage-audit-v1",
        "future_availability_rows": leakage_rows,
        "prohibited_decision_columns": prohibited,
        "feature_allowlist_count": len(features),
        "target_columns_in_model_allowlist": sorted(set(features) & {column for column in frame if column.startswith("target__")}),
        "holdout_target_columns": holdout_target_columns,
        "holdout_rows": len(holdout),
        "passed": leakage_rows == 0 and not prohibited and not holdout_target_columns,
    }
    if not leakage_audit["passed"]:
        raise RuntimeError("LEAKAGE_AUDIT_FAILED")
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "leakage_audit.json").write_text(json.dumps(leakage_audit, indent=2, sort_keys=True) + "\n")
    split_manifest = {
        "schema": "independent-entry-quality-split-manifest-v1",
        "frozen_splits": config["splits"],
        "rows": frame.split.value_counts().sort_index().to_dict(),
        "market_state_groups": frame.groupby("split").market_state_group_id.nunique().to_dict(),
        "temporal_blocks": frame.groupby("split").temporal_block_id.nunique().to_dict(),
        "symbols": frame.groupby("split").symbol.nunique().to_dict(),
        "final_holdout_rows": len(holdout),
        "final_holdout_labels_built": False,
        "purge_embargo_present": True,
    }
    (args.output / "split_manifest.json").write_text(json.dumps(split_manifest, indent=2, sort_keys=True) + "\n")
    result = run_experiment(dataset=frame, feature_dictionary=dictionary, config=config, output=args.output)
    result["dataset_manifest_sha256"] = sha256(args.dataset / "dataset_manifest.json")
    result["split_manifest_sha256"] = sha256(args.output / "split_manifest.json")
    result["leakage_audit_sha256"] = sha256(args.output / "leakage_audit.json")
    (args.output / "result.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    (args.output / "final_report.md").write_text(report(result), encoding="utf-8")
    write_artifact_manifest(args=args, result=result)
    print(json.dumps({"flags": result["flags"], "recommendation": result["recommendation"]}, indent=2, sort_keys=True))
    return 0


def write_artifact_manifest(*, args: argparse.Namespace, result: dict) -> None:
    source_files = sorted((EXPERIMENT / "src").rglob("*.py")) + sorted((EXPERIMENT / "tools").glob("*.py"))
    input_files = [
        args.config,
        args.dataset / "dataset_manifest.json",
        args.dataset / "feature_dictionary.json",
        args.output / "split_manifest.json",
        args.output / "leakage_audit.json",
        *source_files,
    ]
    output_files = sorted(
        path for path in args.output.iterdir()
        if path.is_file() and path.name not in {"artifact_manifest.json", "development_models.joblib"}
    )
    try:
        git_commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=REPOSITORY, check=True,
            capture_output=True, text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        git_commit = "UNAVAILABLE"
    manifest = {
        "schema": "independent-entry-quality-artifact-manifest-v1",
        "experiment": result["schema"],
        "git_commit": git_commit,
        "final_holdout_state": result["final_holdout_state"],
        "inputs": {str(path.relative_to(REPOSITORY)): sha256(path) for path in input_files},
        "outputs": {str(path.relative_to(REPOSITORY)): sha256(path) for path in output_files},
        "rerun_commands": [
            "PYTHONPATH=sandbox/aegis_strategy_router/src:src:sandbox/aegis_strategy_router/experiments/independent_entry_quality_discovery_v1/src .venv/bin/python sandbox/aegis_strategy_router/experiments/independent_entry_quality_discovery_v1/tools/build_dataset.py",
            "PYTHONPATH=sandbox/aegis_strategy_router/src:src:sandbox/aegis_strategy_router/experiments/independent_entry_quality_discovery_v1/src .venv/bin/python sandbox/aegis_strategy_router/experiments/independent_entry_quality_discovery_v1/tools/run_models.py",
        ],
    }
    (args.output / "artifact_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def report(result: dict) -> str:
    top = result["primary_top10"]
    opp = result["model_cards"]["opportunity_logistic"]["validation"]
    direction = result["model_cards"]["direction_logistic"]["validation"]
    lines = [
        "# Independent Entry Quality Discovery V1 Result", "",
        "Classification: `RETROSPECTIVE_DISCOVERY_WITH_TEMPORAL_OOS_VALIDATION`.", "",
        "FINAL_HOLDOUT remains `SEALED_NOT_OPENED`.", "",
        "## Support", "",
        f"- TRAIN: {result['support']['train_rows']:,} rows / {result['support']['train_hour_groups']:,} UTC-hour blocks.",
        f"- VALIDATION: {result['support']['validation_rows']:,} rows / {result['support']['validation_hour_groups']:,} UTC-hour blocks.", "",
        "## Simple model signal", "",
        f"- Opportunity logistic: AUC {opp['roc_auc']:.4f}, log loss {opp['log_loss']:.4f} vs constant {opp['constant_log_loss']:.4f}, ECE {opp['ece']:.4f}.",
        f"- Direction logistic: AUC {direction['roc_auc']:.4f}, log loss {direction['log_loss']:.4f} vs constant {direction['constant_log_loss']:.4f}, ECE {direction['ece']:.4f}.", "",
        "Statistical detectability is not economic sufficiency; the direction AUC only narrowly clears the frozen 0.55 gate.", "",
        "## Primary top-10% quality population", "",
        f"- Rows: {int(top['rows']):,}; effective UTC-hour groups: {int(top['effective_groups']):,}.",
        f"- Favorable first: {top['favorable_first_rate']:.2%}.",
        f"- MFE / MAE: {top['mfe_mean_bps']:.2f} / {top['mae_mean_bps']:.2f} bps.",
        f"- Gross / conservative net: {top['gross_mean_bps']:.2f} / {top['net_mean_bps']:.2f} bps.",
        f"- Net block-bootstrap CI: [{top['net_block_ci_lower_bps']:.2f}, {top['net_block_ci_upper_bps']:.2f}] bps.",
        f"- Ranking Spearman across frozen coverage levels: {result['ranking_spearman']:.4f}.", "",
        "The top 10% remains negative before latency stress and under every non-zero frozen cost scenario.", "",
        "## Stability", "",
        f"- Positive/negative symbols: {result['stability']['positive_symbols']}/{result['stability']['negative_symbols']}.",
        f"- Positive weekly blocks: {result['stability']['positive_week_fraction']:.1%}.",
        f"- LONG/SHORT net: {result['stability']['net_by_side']}.", "",
        "## Flags", "",
    ]
    lines.extend(f"- `{name} = {value}`" for name, value in result["flags"].items())
    lines.extend(["", "## Recommendation", "", f"`{result['recommendation']}`", ""])
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
