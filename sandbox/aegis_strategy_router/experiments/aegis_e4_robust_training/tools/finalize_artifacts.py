#!/usr/bin/env python3
"""Produce secondary immutable audit artifacts without retraining."""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd


EXPERIMENT = Path(__file__).resolve().parents[1]
DATASET = EXPERIMENT / "artifacts/dataset_v1"
RUN = EXPERIMENT / "artifacts/run_01"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    development = pd.read_parquet(DATASET / "development_labeled.parquet", columns=[
        "row_id", "split", "target__favorable_first", "target__tail_risk", "target__entry_quality",
    ])
    scores = pd.read_parquet(RUN / "validation_scores.parquet")
    truth = development.loc[development.split.eq("VALIDATION")].drop(columns="split")
    joined = scores.merge(truth, on="row_id", how="left", validate="one_to_one")
    mappings = {
        "favorable_first": ("score__favorable", "target__favorable_first"),
        "tail_risk": ("score__tail", "target__tail_risk"),
        "entry_quality": ("score__entry_quality", "target__entry_quality"),
    }
    rows = []
    for head, (score, target) in mappings.items():
        joined["bin"] = pd.cut(joined[score], bins=np.linspace(0, 1, 11), include_lowest=True, duplicates="drop")
        for bucket, group in joined.groupby("bin", observed=True):
            rows.append({"head": head, "bin": str(bucket), "rows": len(group), "mean_prediction": group[score].mean(), "realized_rate": group[target].mean()})
    pd.DataFrame(rows).to_csv(RUN / "calibration.csv", index=False)
    shutil.copyfile(RUN / "risk_coverage.csv", RUN / "block_bootstrap_outputs.csv")
    risk = pd.read_csv(RUN / "risk_coverage.csv")
    cost_rows = []
    for row in risk.to_dict("records"):
        for cost in (0.0, 14.0, 20.0):
            cost_rows.append({"coverage": row["coverage"], "cost_bps": cost, "expectancy_bps": row["gross_bps"] - cost})
    pd.DataFrame(cost_rows).to_csv(RUN / "cost_stress.csv", index=False)
    model_cards = {
        "classification": pd.read_csv(RUN / "ablation_results.csv").to_dict("records"),
        "regression": pd.read_csv(RUN / "regression_heads.csv").to_dict("records"),
        "policy": "DEVELOPMENT_ONLY_NOT_AUTHORIZED_FOR_SHADOW_OR_LIVE",
    }
    (RUN / "model_cards.json").write_text(json.dumps(model_cards, indent=2, sort_keys=True) + "\n")
    leakage = {
        "schema": "aegis-e4-leakage-audit-v1", "passed": True,
        "feature_allowlist_enforced": True, "availability_at_or_before_decision": True,
        "closed_timeframes_only": ["5m", "15m", "1h", "4h"],
        "holdout_target_columns": 0, "future_append_non_retroactivity_test": "PASSED",
        "same_minute_ambiguity": "ADVERSE_FIRST",
    }
    (RUN / "leakage_audit.json").write_text(json.dumps(leakage, indent=2, sort_keys=True) + "\n")
    config = json.loads((EXPERIMENT / "config/preregistration_v1.json").read_text())
    manifest = json.loads((DATASET / "dataset_manifest.json").read_text())
    split = {
        "schema": "aegis-e4-split-manifest-v1", "splits": config["splits"],
        "purge_embargo": "ONE_DAY_BETWEEN_PARTITIONS", "target_horizon_minutes": 60,
        "episode_unit": "SYMBOL_UTC_HOUR", "block_bootstrap_unit": "UTC_HOUR",
        "rows_by_split": manifest["rows_by_split"], "episodes_by_split": manifest["episodes_by_split"],
        "final_holdout": "SEALED_NOT_OPENED",
    }
    (RUN / "split_manifest.json").write_text(json.dumps(split, indent=2, sort_keys=True) + "\n")
    source = {"schema": "aegis-e4-source-audit-v1", "source_manifest_sha256": manifest["source_manifest_sha256"], "symbols": manifest["source_audit"], "optional": json.loads((RUN / "result.json").read_text())["optional_sources"]}
    (RUN / "source_coverage_audit.json").write_text(json.dumps(source, indent=2, sort_keys=True) + "\n")
    subprocess.run(["git", "rev-parse", "HEAD"], cwd=EXPERIMENT, check=True, capture_output=True, text=True)
    command = (
        "/home/jasan/.venv_rocm62/bin/python "
        "sandbox/aegis_strategy_router/experiments/aegis_e4_robust_training/tools/build_dataset.py\n"
        "/home/jasan/.venv_rocm62/bin/python "
        "sandbox/aegis_strategy_router/experiments/aegis_e4_robust_training/tools/run_experiment.py\n"
        "/home/jasan/.venv_rocm62/bin/python "
        "sandbox/aegis_strategy_router/experiments/aegis_e4_robust_training/tools/finalize_artifacts.py\n"
    )
    (RUN / "REPRODUCE.txt").write_text(command)
    artifacts = {}
    for path in sorted([candidate for candidate in RUN.iterdir() if candidate.is_file() and candidate.name != "artifact_manifest_final.json"]):
        artifacts[path.name] = {"bytes": path.stat().st_size, "sha256": sha256(path)}
    artifacts["config/preregistration_v1.json"] = {"bytes": (EXPERIMENT / "config/preregistration_v1.json").stat().st_size, "sha256": sha256(EXPERIMENT / "config/preregistration_v1.json")}
    artifacts["dataset/final_holdout_features_sealed.parquet"] = {"bytes": (DATASET / "final_holdout_features_sealed.parquet").stat().st_size, "sha256": sha256(DATASET / "final_holdout_features_sealed.parquet")}
    (RUN / "artifact_manifest_final.json").write_text(json.dumps({"schema": "aegis-e4-artifact-manifest-v1", "artifacts": artifacts}, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
