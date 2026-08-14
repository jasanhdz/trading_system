#!/usr/bin/env python3
"""Evaluate frozen C2A event detectors without opening holdout by default."""

from __future__ import annotations

import argparse
import concurrent.futures
import ctypes
import gc
import json
import math
import multiprocessing
import os
from pathlib import Path
import tempfile
from typing import Any

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
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


def _release_native_memory() -> None:
    """Return Arrow/libc caches between large, independent symbol scans."""

    gc.collect()
    pa.default_memory_pool().release_unused()
    try:
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except (AttributeError, OSError):
        pass


def _partition_bounds(config: dict[str, Any], open_holdout: bool) -> dict[str, tuple[int, int]]:
    result = {}
    for name in ("train", "validation", "final_holdout"):
        if name == "final_holdout" and not open_holdout:
            continue
        start, end = config["partitions"][name]
        result[name] = (_timestamp(start), _timestamp(end))
    return result


def _write_symbol_fragments(
    symbol: str,
    dataset_root: Path,
    config: dict[str, Any],
    bounds: dict[str, tuple[int, int]],
    cooldown: int,
    projected_columns: tuple[str, ...],
    temp_root: Path,
) -> None:
    path = dataset_root / f"{symbol}.parquet"
    frame = pd.read_parquet(path, columns=projected_columns)
    for name, (start, end) in bounds.items():
        population = frame.loc[
            frame.event_timestamp_ms.ge(start) & frame.event_timestamp_ms.lt(end)
        ].copy()
        events = collapse_registered_events(
            detect_registered_events(population, config), cooldown
        )
        if not events.empty:
            events.to_parquet(temp_root / f"{name}-{symbol}-events.parquet", index=False)
            controls = pd.concat([
                deterministic_matched_control(population, selected).assign(
                    event_family=family
                )
                for family, selected in events.groupby("event_family", sort=True)
            ], ignore_index=True)
            controls.to_parquet(
                temp_root / f"{name}-{symbol}-controls.parquet", index=False
            )
    print(json.dumps({"symbol": symbol, "state": "EVALUATED"}), flush=True)


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
    dataset_hashes = {}
    first_path = dataset_root / f"{CANONICAL_SYMBOLS[0]}.parquet"
    available_columns = tuple(pq.read_schema(first_path).names)
    utility_columns = tuple(
        column for column in available_columns if column.endswith("_net_utility")
    )
    detector_columns = (
        "event_timestamp_ms", "symbol", "side", "side_flow_z",
        "trade_count_z", "side_flow_imbalance_3m",
        "side_flow_persistence_5m", "side_price_response_1m",
    )
    projected_columns = (*detector_columns, *utility_columns)
    contracts = sorted(
        column.removesuffix("_net_utility") for column in utility_columns
    )
    families = tuple(config["event_detectors"])
    sides = ("LONG", "SHORT")
    results = {}
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="c2a-evaluation-", dir=output.parent) as temp:
        temp_root = Path(temp)
        fragments: dict[str, dict[str, list[Path]]] = {
            name: {"events": [], "controls": []} for name in bounds
        }
        process_context = multiprocessing.get_context("spawn")
        for symbol in CANONICAL_SYMBOLS:
            path = dataset_root / f"{symbol}.parquet"
            dataset_hashes[symbol] = sha256_file(path)
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=3, mp_context=process_context
        ) as executor:
            futures = {
                executor.submit(
                    _write_symbol_fragments, symbol, dataset_root, config,
                    bounds, cooldown, projected_columns, temp_root,
                ): symbol
                for symbol in CANONICAL_SYMBOLS
            }
            for future in concurrent.futures.as_completed(futures):
                symbol = futures[future]
                try:
                    future.result()
                except Exception as error:
                    raise RuntimeError(
                        f"AEGIS_C2A_SYMBOL_EVALUATION_FAILED:{symbol}"
                    ) from error

        for partition in bounds:
            fragments[partition]["events"] = sorted(
                temp_root.glob(f"{partition}-*-events.parquet")
            )
            fragments[partition]["controls"] = sorted(
                temp_root.glob(f"{partition}-*-controls.parquet")
            )

        metric_columns = ["event_timestamp_ms", "event_family", "side", *utility_columns]
        for partition in bounds:
            partition_result = {}
            for family in families:
                for side in sides:
                    filters = [("event_family", "==", family), ("side", "==", side)]
                    selected = pd.read_parquet(
                        fragments[partition]["events"], columns=metric_columns,
                        filters=filters,
                    )
                    if selected.empty:
                        continue
                    matched = pd.read_parquet(
                        fragments[partition]["controls"], columns=metric_columns,
                        filters=filters,
                    )
                    group_result = {}
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
                    del selected, matched
                    _release_native_memory()
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
