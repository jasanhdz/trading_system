#!/usr/bin/env python3
"""Execute the preregistered C1 incremental information audit."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from aegis.research.information_value_audit_c1 import (
    CANDIDATES,
    SIDES,
    canonical_features,
    contract_hash,
    day_cluster_bootstrap,
    evaluate_bundle,
    feature_names,
    fit_bundle,
    incremental_metrics,
)
from aegis.research.opportunity_atlas_b1 import partition_mask
from aegis.utils import sha256_file


PARTITIONS = {
    "TRAIN": ("2024-01-22T00:00:00Z", "2025-01-01T00:00:00Z"),
    "CALIBRATION": ("2025-01-02T00:00:00Z", "2025-07-01T00:00:00Z"),
    "VALIDATION": ("2025-07-02T00:00:00Z", "2026-01-01T00:00:00Z"),
    "PSEUDO_FORWARD": ("2026-01-02T00:00:00Z", "2026-08-01T00:00:00Z"),
}
SOURCES = {
    60: ("825a160258d743f5a8c187c4c27400412f6a53b9f258341f4fefe516546c2c2e"),
    240: ("1bb5cea187e40972686fbb86eb3ee1c1d24b3b4f320c059871107d0ebaec0453"),
}


def _safe(value: Any) -> Any:
    if isinstance(value, (np.integer, np.floating)):
        return _safe(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return "INF" if value > 0 else "-INF"
    if isinstance(value, dict):
        return {str(key): _safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_safe(item) for item in value]
    return value


def _aggregate(side_reports: dict[str, dict[str, Any]]) -> dict[str, float]:
    metrics = next(iter(side_reports.values())).keys()
    result = {}
    for metric in metrics:
        values = [report[metric] for report in side_reports.values() if isinstance(report[metric], (int, float))]
        if values:
            result[metric] = float(np.mean(values))
    return result


def _candidate_gate(report: dict[str, Any]) -> dict[str, bool]:
    validation = report["VALIDATION"]
    forward = report["PSEUDO_FORWARD"]
    return {
        "validation_rank_lift": validation["incremental"]["grouped_spearman_lift"] >= .01,
        "forward_rank_lift": forward["incremental"]["grouped_spearman_lift"] >= .01,
        "validation_rank_absolute": validation["aggregate"]["grouped_spearman"] >= .05,
        "forward_rank_absolute": forward["aggregate"]["grouped_spearman"] >= .05,
        "validation_barrier_log_loss": validation["incremental"]["barrier_log_loss_improvement"] > 0,
        "forward_barrier_log_loss": forward["incremental"]["barrier_log_loss_improvement"] > 0,
        "validation_barrier_ap": validation["incremental"]["barrier_average_precision_improvement"] > 0,
        "forward_barrier_ap": forward["incremental"]["barrier_average_precision_improvement"] > 0,
        "validation_mae_lift": validation["incremental"]["mae_spearman_lift"] >= .02,
        "forward_mae_lift": forward["incremental"]["mae_spearman_lift"] >= .02,
        "validation_net": validation["aggregate"]["selected_primary_net"] > 0,
        "forward_net": forward["aggregate"]["selected_primary_net"] > 0,
        "validation_stress": validation["aggregate"]["selected_stress_net"] > 0,
        "forward_stress": forward["aggregate"]["selected_stress_net"] > 0,
        "forward_bootstrap": all(
            item is not None and item["lower_95"] > 0 for item in forward["bootstrap"].values()
        ),
        "minimum_events": all(
            partition["aggregate"]["selected_events"] >= 100
            for partition in (validation, forward)
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=Path("data/information_value_audit_c1/run_01"))
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    output = args.output_root if args.output_root.is_absolute() else root / args.output_root
    output.mkdir(parents=True, exist_ok=True)
    os.chmod(output, 0o700)
    results: dict[str, Any] = {}
    passed: dict[str, list[int]] = {candidate: [] for candidate in CANDIDATES if candidate != "PRICE_STATE"}
    for horizon, expected_hash in SOURCES.items():
        source = root / f"data/residual_regime_alpha_b2/run_01/residual_symbol_sides_{horizon}m.parquet"
        if sha256_file(source) != expected_hash:
            raise RuntimeError("AEGIS_C1_SOURCE_HASH_MISMATCH")
        rows = pd.read_parquet(source)
        features = canonical_features(rows)
        rows = pd.concat([rows.reset_index(drop=True), features.reset_index(drop=True)], axis=1)
        masks = {name: partition_mask(rows, *bounds) for name, bounds in PARTITIONS.items()}
        timestamps = {name: set(rows.loc[mask, "timestamp_ms"].astype(int)) for name, mask in masks.items()}
        horizon_report: dict[str, Any] = {"rows": len(rows), "events": rows.timestamp_ms.nunique(), "candidates": {}}
        evaluated: dict[str, dict[str, Any]] = {}
        for candidate in CANDIDATES:
            evaluated[candidate] = {}
            for side in SIDES:
                side_rows = rows.loc[rows.side.eq(side)]
                bundle = fit_bundle(side_rows, timestamps["TRAIN"], timestamps["CALIBRATION"], candidate)
                for partition in ("VALIDATION", "PSEUDO_FORWARD"):
                    sample = side_rows.loc[side_rows.timestamp_ms.isin(timestamps[partition])]
                    metrics, selected = evaluate_bundle(bundle, sample)
                    evaluated[candidate].setdefault(partition, {})[side] = {
                        "metrics": metrics, "bootstrap": day_cluster_bootstrap(selected),
                    }
        baseline = evaluated["PRICE_STATE"]
        for candidate, partitions in evaluated.items():
            report = {}
            for partition in ("VALIDATION", "PSEUDO_FORWARD"):
                side_metrics = {side: partitions[partition][side]["metrics"] for side in SIDES}
                aggregate = _aggregate(side_metrics)
                base_aggregate = _aggregate({side: baseline[partition][side]["metrics"] for side in SIDES})
                report[partition] = {
                    "sides": side_metrics, "aggregate": aggregate,
                    "incremental": incremental_metrics(aggregate, base_aggregate),
                    "bootstrap": {side: partitions[partition][side]["bootstrap"] for side in SIDES},
                }
            if candidate != "PRICE_STATE":
                gate = _candidate_gate(report)
                report["gate"] = gate
                report["gate_pass"] = all(gate.values())
                if report["gate_pass"]:
                    passed[candidate].append(horizon)
            horizon_report["candidates"][candidate] = report
        results[f"{horizon}m"] = horizon_report
        print(
            f"c1_horizon={horizon} " + " ".join(
                f"{name}={report.get('gate_pass', 'BASELINE')}"
                for name, report in horizon_report["candidates"].items()
            ), flush=True,
        )
    passing = [candidate for candidate, horizons in passed.items() if len(horizons) == 2]
    result = {
        "schema_version": "aegis-information-value-audit-c1-result-v1",
        "experiment_id": "aegis-information-value-audit-c1-01",
        "evidence_class": "CONTAMINATED_INFORMATION_DIAGNOSTIC_NO_PROMOTION_AUTHORITY",
        "source_availability": {
            "PRICE_STATE": "AVAILABLE", "FLOW_ACTIVITY": "AVAILABLE",
            "DERIVATIVES_CARRY": "AVAILABLE", "CROSS_MARKET": "AVAILABLE",
            "CALENDAR_CONTROL": "AVAILABLE", "OPEN_INTEREST": "UNAVAILABLE_REQUIRED_HISTORY",
            "LIQUIDATIONS": "UNAVAILABLE_HISTORICALLY", "ORDER_BOOK": "UNAVAILABLE_HISTORICALLY",
            "NEWS": "NO_POINT_IN_TIME_CORPUS",
        },
        "feature_contracts": {
            candidate: {"names": feature_names(candidate), "sha256": contract_hash(feature_names(candidate))}
            for candidate in CANDIDATES
        },
        "results": results, "passing_candidates": passing,
        "C1_INCREMENTAL_INFORMATION_FOUND": bool(passing),
        "C1_NEW_SOURCE_ACQUISITION_REQUIRED": not bool(passing),
        "C1_READY_FOR_C2": bool(passing), "C1_READY_FOR_SHADOW": False,
        "C1_READY_FOR_LIVE": False, "exchange_calls": 0,
        "exchange_mutations": 0, "runtime_changes": "NONE",
    }
    result_path = output / "result.json"
    result_path.write_text(json.dumps(_safe(result), indent=2, sort_keys=True) + "\n")
    os.chmod(result_path, 0o600)
    print(json.dumps({"result": str(result_path), "passing_candidates": passing}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
