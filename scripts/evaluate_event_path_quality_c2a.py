#!/usr/bin/env python3
"""Evaluate frozen C2A event detectors without opening holdout by default."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from aegis.config import CANONICAL_SYMBOLS
from aegis.research.event_path_quality_c2a import (
    collapse_registered_events,
    day_cluster_bootstrap,
    detect_registered_events,
    deterministic_matched_control,
    economic_summary,
)
from aegis.utils import sha256_file


def _safe(value: Any) -> Any:
    if isinstance(value, float) and not math.isfinite(value):
        return "INF" if value > 0 else "-INF"
    if isinstance(value, dict):
        return {str(key): _safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_safe(item) for item in value]
    return value


def _timestamp(value: str) -> int:
    return int(pd.Timestamp(value).timestamp() * 1_000)


def _partition_bounds(config: dict[str, Any], open_holdout: bool) -> dict[str, tuple[int, int]]:
    result = {}
    for name in ("train", "validation", "final_holdout"):
        if name == "final_holdout" and not open_holdout:
            continue
        start, end = config["partitions"][name]
        result[name] = (_timestamp(start), _timestamp(end))
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-root", type=Path,
        default=Path("data/event_path_quality_c2a/dataset_2025_08_2026_07"),
    )
    parser.add_argument(
        "--config", type=Path,
        default=Path("config/experiments/aegis_event_path_quality_c2a.yaml"),
    )
    parser.add_argument(
        "--output", type=Path,
        default=Path("data/event_path_quality_c2a/evaluation_01.json"),
    )
    parser.add_argument("--open-final-holdout", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    dataset_root = args.dataset_root if args.dataset_root.is_absolute() else root / args.dataset_root
    config_path = args.config if args.config.is_absolute() else root / args.config
    output = args.output if args.output.is_absolute() else root / args.output
    config = yaml.safe_load(config_path.read_text())
    bounds = _partition_bounds(config, args.open_final_holdout)
    cooldown = int(config["evidence_gate"]["event_cooldown_minutes"])
    events_by_partition: dict[str, list[pd.DataFrame]] = {name: [] for name in bounds}
    population_by_partition: dict[str, list[pd.DataFrame]] = {name: [] for name in bounds}
    dataset_hashes = {}
    for symbol in CANONICAL_SYMBOLS:
        path = dataset_root / f"{symbol}.parquet"
        frame = pd.read_parquet(path)
        dataset_hashes[symbol] = sha256_file(path)
        for name, (start, end) in bounds.items():
            population = frame.loc[
                frame.event_timestamp_ms.ge(start) & frame.event_timestamp_ms.lt(end)
            ].copy()
            population_by_partition[name].append(population)
            events = collapse_registered_events(
                detect_registered_events(population, config), cooldown
            )
            events_by_partition[name].append(events)
        print(json.dumps({"symbol": symbol, "state": "EVALUATED"}), flush=True)

    results = {}
    contracts = sorted({
        column.removesuffix("_net_utility")
        for frame in population_by_partition["train"]
        for column in frame.columns if column.endswith("_net_utility")
    })
    for partition in bounds:
        population = pd.concat(population_by_partition[partition], ignore_index=True)
        events = pd.concat(events_by_partition[partition], ignore_index=True)
        partition_result = {}
        for (family, side), selected in events.groupby(["event_family", "side"], sort=True):
            group_result = {}
            matched = deterministic_matched_control(population, selected)
            for contract in contracts:
                utility = f"{contract}_net_utility"
                event_summary = economic_summary(selected, utility)
                control_summary = economic_summary(matched, utility)
                interval = day_cluster_bootstrap(selected, utility)
                group_result[contract] = {
                    "selected": event_summary,
                    "matched_random": control_summary,
                    "bootstrap": interval,
                    "outperforms_matched_random": (
                        float(event_summary["net_expectancy"])
                        > float(control_summary["net_expectancy"])
                    ),
                }
            partition_result[f"{family}:{side}"] = group_result
        results[partition] = partition_result

    validation = results.get("validation", {})
    validation_pass = bool(validation) and any(
        details[contract]["selected"]["events"] >= 1_000
        and details[contract]["selected"]["net_expectancy"] > 0.0
        and details[contract]["bootstrap"]["expectancy_lower_95"] > 0.0
        and details[contract]["selected"]["profit_factor"] > 1.10
        and details[contract]["outperforms_matched_random"]
        for details in validation.values() for contract in details
    )
    report = {
        "schema_version": "aegis-event-path-quality-c2a-evaluation-v1",
        "config_sha256": sha256_file(config_path),
        "dataset_hashes": dataset_hashes,
        "final_holdout_state": "OPENED_ONCE" if args.open_final_holdout else "SEALED",
        "results": results,
        "validation_partial_gate_pass": validation_pass,
        "all_controls_complete": False,
        "C2A_INCREMENTAL_EDGE_FOUND": False,
        "C2A_READY_FOR_MODELING": False,
        "C2A_READY_FOR_SHADOW": False,
        "C2A_READY_FOR_LIVE": False,
        "authenticated_requests": 0,
        "exchange_mutations": 0,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(_safe(report), indent=2, sort_keys=True) + "\n")
    os.chmod(output, 0o600)
    print(json.dumps({
        "output": str(output), "holdout": report["final_holdout_state"],
        "validation_partial_gate_pass": validation_pass,
    }))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
