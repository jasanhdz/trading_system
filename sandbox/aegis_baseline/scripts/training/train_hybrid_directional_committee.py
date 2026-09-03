"""Train and validate the hybrid directional committee on purged folds."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml

from aegis.config import CANONICAL_SYMBOLS
from aegis.data import CanonicalSeriesSource, DataPurpose
from aegis.features import FEATURE_HASH, FEATURE_NAMES, FEATURE_SCHEMA_VERSION
from aegis.training.dataset import (
    build_e2_hourly_long_dataset,
    build_e2_hourly_short_dataset,
)
from aegis.training.hybrid_directional import (
    DirectionalSide,
    HybridDirectionalArtifact,
    HybridDirectionalRow,
    fit_hybrid_directional_committee,
    paired_directional_rows,
    write_hybrid_directional_artifact,
)
from aegis.training.run_state import atomic_write_json
from aegis.utils import sha256_file

ROOT = Path(__file__).resolve().parents[1]


def _mapping(value: Any, identity: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{identity} must be a mapping")
    return value


def _utc(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise ValueError("hybrid timestamp must be timezone-aware")
    return parsed.astimezone(timezone.utc)


def _block(
    rows: Sequence[HybridDirectionalRow],
    start: datetime,
    end: datetime,
    *,
    exclude_symbol: str | None = None,
    only_symbol: str | None = None,
) -> tuple[HybridDirectionalRow, ...]:
    start_ns = int(start.timestamp() * 1_000_000_000)
    end_ns = int(end.timestamp() * 1_000_000_000)
    return tuple(
        row
        for row in rows
        if start_ns <= row.timestamp_ns <= end_ns
        and (exclude_symbol is None or row.symbol != exclude_symbol)
        and (only_symbol is None or row.symbol == only_symbol)
    )


def _fit_fold(
    rows: Sequence[HybridDirectionalRow],
    fold: Mapping[str, Any],
    config: Mapping[str, Any],
    *,
    exclude_symbol: str | None = None,
    scoring_symbol: str | None = None,
) -> HybridDirectionalArtifact:
    training = _mapping(config["training"], "training")
    protocol = _mapping(config["fold_protocol"], "fold_protocol")
    train = _block(
        rows,
        _utc(str(fold["train_start"])),
        _utc(str(fold["train_end"])),
        exclude_symbol=exclude_symbol,
    )
    calibration = _block(
        rows,
        _utc(str(fold["calibration_start"])),
        _utc(str(fold["calibration_end"])),
        exclude_symbol=exclude_symbol,
    )
    scoring = _block(
        rows,
        _utc(str(fold["scoring_start"])),
        _utc(str(fold["scoring_end"])),
        only_symbol=scoring_symbol,
    )
    return fit_hybrid_directional_committee(
        train,
        calibration,
        scoring,
        seed=int(training["seed"]) + int(fold["id"]),
        embargo_minutes=int(protocol["embargo_minutes"]),
        round_trip_cost_fraction=float(config["labels"]["round_trip_cost_fraction"]),
        classifier_parameters=_mapping(training["classifier"], "classifier"),
        regressor_parameters=_mapping(training["regressor"], "regressor"),
    )


def _checks(
    artifact: HybridDirectionalArtifact, config: Mapping[str, Any]
) -> Mapping[str, bool]:
    evaluation = _mapping(config["evaluation"], "evaluation")
    metrics = artifact.metrics
    return {
        "opportunity_average_precision_lift": all(
            metrics.opportunity_average_precision[side.value]
            > metrics.opportunity_prevalence[side.value]
            for side in DirectionalSide
        ),
        "opportunity_calibration": all(
            metrics.opportunity_ece[side.value]
            <= float(evaluation["maximum_opportunity_ece"])
            for side in DirectionalSide
        ),
        "mae_q90_coverage": (
            float(evaluation["q90_coverage_minimum"])
            <= metrics.mae_q90_coverage
            <= float(evaluation["q90_coverage_maximum"])
        ),
        "mae_q90_beats_baseline": (
            metrics.mae_q90_pinball < metrics.mae_q90_baseline_pinball
        ),
        "positive_shadow_expectancy": all(
            float(metrics.shadow_selection[side.value]["mean_net_expectancy"]) > 0.0
            for side in DirectionalSide
        ),
        "symbol_concentration": all(
            float(metrics.shadow_selection[side.value]["symbol_concentration"])
            <= float(evaluation["maximum_symbol_concentration"])
            for side in DirectionalSide
        ),
    }


def _load_rows(
    config: Mapping[str, Any],
) -> tuple[tuple[HybridDirectionalRow, ...], Mapping[str, Any]]:
    source_config = _mapping(config["source"], "source")
    source = CanonicalSeriesSource(
        Path(str(source_config["path"])),
        DataPurpose.TRAINING,
        expected_manifest_sha256=str(source_config["manifest_sha256"]),
    )
    audit = source.audit(verify_content=True)
    if not audit.finality_verified:
        raise ValueError("hybrid source finality is not verified")
    sampling = _mapping(config["sampling"], "sampling")
    first = _utc(str(sampling["expected_rows"]["first_anchor_utc"]))
    last = _utc(str(sampling["expected_rows"]["last_dev_anchor_utc"]))
    history = int(sampling["history_bars"])
    horizon = int(sampling["horizon_bars"])
    lockbox_start = _utc(str(config["lockbox"]["start"]))
    start = first - timedelta(minutes=history * 5)
    end = last + timedelta(minutes=horizon * 5)
    if end > lockbox_start:
        raise ValueError("hybrid lockbox access prohibited")
    series = source.load(start=start, end=end)
    long_build = build_e2_hourly_long_dataset(
        series,
        sampling,
        dataset_id=f"{config['experiment_id']}-long",
        source_finality_verified=True,
    )
    short_build = build_e2_hourly_short_dataset(
        series,
        sampling,
        dataset_id=f"{config['experiment_id']}-short",
        source_finality_verified=True,
    )
    rows = paired_directional_rows(
        long_build.dataset,
        short_build.dataset,
        round_trip_cost_fraction=float(config["labels"]["round_trip_cost_fraction"]),
    )
    return rows, {
        "source_manifest_sha256": audit.manifest_sha256,
        "source_finality_verified": audit.finality_verified,
        "long_dataset_sha256": long_build.dataset.artifact_hash,
        "short_dataset_sha256": short_build.dataset.artifact_hash,
        "directional_rows": len(rows),
        "base_rows_per_direction": long_build.dataset.row_count,
        "rows_by_symbol": long_build.rows_by_symbol,
        "first_anchor": long_build.first_anchor.isoformat(),
        "last_anchor": long_build.last_anchor.isoformat(),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config/experiments/aegis_hybrid_directional_committee_v1.yaml"),
    )
    args = parser.parse_args()
    config_path = (ROOT / args.config).resolve()
    config = _mapping(yaml.safe_load(config_path.read_text(encoding="utf-8")), "config")
    if (
        config.get("schema_version") != "aegis-hybrid-directional-preregistration-v1"
        or config.get("runtime_authority") != "SHADOW_ONLY"
        or config.get("feature_schema_version") != FEATURE_SCHEMA_VERSION
        or int(config.get("feature_count", 0)) != len(FEATURE_NAMES)
        or config.get("feature_hash") != FEATURE_HASH
        or tuple(config.get("symbols", ())) != CANONICAL_SYMBOLS
        or config["lockbox"]["access"] != "FORBIDDEN"
        or config["promotion"]["automatic_live_activation"] is not False
        or config["promotion"]["current_live_eligible"] is not False
    ):
        raise SystemExit("AEGIS_HYBRID_DIRECTIONAL_CONFIG_INVALID")

    rows, dataset_report = _load_rows(config)
    folds = []
    latest_artifact: HybridDirectionalArtifact | None = None
    for fold in config["fold_protocol"]["folds"]:
        artifact = _fit_fold(rows, _mapping(fold, "fold"), config)
        checks = _checks(artifact, config)
        folds.append(
            {
                "fold_id": int(fold["id"]),
                "metrics": asdict(artifact.metrics),
                "checks": checks,
                "passed": all(checks.values()),
            }
        )
        latest_artifact = artifact
    assert latest_artifact is not None

    last_fold = _mapping(config["fold_protocol"]["folds"][-1], "last_fold")
    ada_artifact = _fit_fold(
        rows,
        last_fold,
        config,
        exclude_symbol="ADAUSDT",
        scoring_symbol="ADAUSDT",
    )
    ada_report = {
        "training_population": "TEN_SYMBOLS_EXCLUDING_ADAUSDT",
        "scoring_population": "ADAUSDT_ONLY",
        "metrics": asdict(ada_artifact.metrics),
        "memorization_possible": False,
        "diagnostic_only": True,
    }

    positive_folds = sum(bool(fold["passed"]) for fold in folds)
    offline_passed = positive_folds == len(folds)
    output = _mapping(config["outputs"], "outputs")
    report_path = ROOT / str(output["validation_report"])
    artifact_path = ROOT / str(output["artifact"])
    readiness_path = ROOT / str(output["readiness"])
    write_hybrid_directional_artifact(artifact_path, latest_artifact)
    report = {
        "schema_id": "aegis-hybrid-directional-validation-v1",
        "experiment_id": config["experiment_id"],
        "config_path": str(config_path.relative_to(ROOT)),
        "config_sha256": sha256_file(config_path),
        "feature_schema": FEATURE_SCHEMA_VERSION,
        "feature_count": len(FEATURE_NAMES),
        "dataset": dataset_report,
        "folds": folds,
        "folds_passed": positive_folds,
        "fold_count": len(folds),
        "offline_validation_passed": offline_passed,
        "ada_leave_one_symbol_out": ada_report,
        "runtime_authority": "SHADOW_ONLY",
        "automatic_live_activation": False,
        "exchange_authority": False,
        "exchange_mutations": 0,
    }
    atomic_write_json(report_path, report)
    atomic_write_json(
        readiness_path,
        {
            "schema_id": "aegis-hybrid-directional-readiness-v1",
            "state": (
                "SHADOW_EVIDENCE_REQUIRED"
                if offline_passed
                else "OFFLINE_VALIDATION_FAILED"
            ),
            "offline_validation_passed": offline_passed,
            "runtime_authority": "SHADOW_ONLY",
            "current_live_eligible": False,
            "automatic_live_activation": False,
            "owner_authorization_required": True,
            "artifact_path": str(artifact_path.relative_to(ROOT)),
            "artifact_sha256": sha256_file(artifact_path),
            "validation_report_path": str(report_path.relative_to(ROOT)),
            "validation_report_sha256": sha256_file(report_path),
        },
    )
    print(
        yaml.safe_dump(
            {
                "offline_validation_passed": offline_passed,
                "folds_passed": positive_folds,
                "fold_count": len(folds),
                "artifact": str(artifact_path.relative_to(ROOT)),
                "report": str(report_path.relative_to(ROOT)),
                "runtime_authority": "SHADOW_ONLY",
            },
            sort_keys=True,
        )
    )
    return 0 if offline_passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
