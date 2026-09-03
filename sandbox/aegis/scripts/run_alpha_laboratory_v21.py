#!/usr/bin/env python3
"""Run the preregistered V21 simple-strategy alpha laboratory."""

from __future__ import annotations

import argparse
import bisect
import gzip
import io
import json
import os
import sqlite3
from collections import Counter, defaultdict
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Mapping

import yaml

from aegis.research.alpha_laboratory_v21 import (
    AlphaStrategy,
    apply_event_spacing,
    candidate_record,
    classify_timestamp,
    economic_summary,
    gate_assessment,
    matched_control,
    partition_name,
    prepare_row,
    timestamp_ms,
)
from aegis.utils import Sha256HashProvider, sha256_file


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"V21 {name} must be a mapping")
    return value


@contextmanager
def _deterministic_gzip_text(path: Path):
    with path.open("wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as compressed:
            with io.TextIOWrapper(compressed, encoding="utf-8", newline="\n") as text:
                yield text


def _funding_series(database: Path) -> tuple[dict[str, tuple[list[int], list[float]]], Mapping[str, Any]]:
    connection = sqlite3.connect(f"file:{database.resolve()}?mode=ro", uri=True)
    try:
        table_names = {
            str(row[0])
            for row in connection.execute("SELECT name FROM sqlite_master WHERE type='table'")
        }
        rows = connection.execute(
            "SELECT symbol,funding_time_ms,funding_rate FROM funding_history "
            "ORDER BY symbol,funding_time_ms"
        )
        series: dict[str, tuple[list[int], list[float]]] = {}
        for symbol, timestamp, rate in rows:
            times, rates = series.setdefault(str(symbol), ([], []))
            times.append(int(timestamp))
            rates.append(float(rate))
        inventory = {
            "tables": sorted(table_names),
            "funding_rows": sum(len(values[0]) for values in series.values()),
            "funding_symbols": len(series),
            "spot_history_present": any("spot" in name.lower() for name in table_names),
            "basis_history_present": any("basis" in name.lower() for name in table_names),
        }
    finally:
        connection.close()
    return series, inventory


def _funding_at(
    series: tuple[list[int], list[float]] | None, current_ms: int
) -> tuple[float | None, int | None]:
    if series is None:
        return None, None
    times, rates = series
    index = bisect.bisect_right(times, current_ms) - 1
    if index < 0:
        return None, None
    return rates[index], current_ms - times[index]


def _population_record(source: Mapping[str, Any], profile_name: str, config: Mapping[str, Any]) -> dict[str, Any]:
    profile = source["protection_profiles"][profile_name]
    contract = source["v10_contract_outcomes"]["ROE_10_H12"]
    return {
        "timestamp": str(source["timestamp"]),
        "partition": partition_name(str(source["timestamp"]), config["temporal_protocol"]),
        "symbol": str(source["symbol"]),
        "side": str(source["side"]),
        "protected_net_return": float(profile["worst_net_return"]),
        "contract_utility": float(contract["realized_utility"]),
        "mae_fraction": float(source["mae_fraction"]),
        "mfe_fraction": float(source["mfe_fraction"]),
        "time_underwater_bars": float(source["time_underwater_bars"]),
        "break_even_armed": bool(profile["break_even_armed"]),
        "trailing_armed": bool(profile["trailing_armed"]),
    }


def run(root: Path, config: Mapping[str, Any], output_root: Path) -> Mapping[str, Any]:
    authority = _mapping(config["authority"], "authority")
    bindings = {
        root / str(authority["source_dataset"]): str(authority["source_dataset_sha256"]),
        root / str(authority["source_manifest"]): str(authority["source_manifest_sha256"]),
        root / str(authority["source_validation"]): str(authority["source_validation_sha256"]),
        root / str(authority["funding_database"]): str(authority["funding_database_sha256"]),
    }
    for path, expected in bindings.items():
        if not path.is_file() or sha256_file(path) != expected:
            raise RuntimeError(f"V21 authority mismatch: {path}")

    funding, funding_inventory = _funding_series(root / str(authority["funding_database"]))
    carry_ready = bool(
        funding_inventory["spot_history_present"] and funding_inventory["basis_history_present"]
    )
    if carry_ready:
        raise RuntimeError("V21 carry inputs appeared after preregistration; separate contract required")

    source_rows = 0
    timestamp_groups = 0
    raw_candidates: list[Mapping[str, Any]] = []
    populations: dict[tuple[str, str, str], list[Mapping[str, Any]]] = defaultdict(list)
    profile_by_strategy = {
        strategy.value: str(config["strategies"][strategy.value]["exit_profile"])
        for strategy in (
            AlphaStrategy.CROSS_SECTIONAL_MOMENTUM,
            AlphaStrategy.EXTREME_REVERSAL,
            AlphaStrategy.BREAKOUT_FLOW_FUNDING,
        )
    }

    def process_group(group: list[Mapping[str, Any]]) -> None:
        nonlocal timestamp_groups
        if not group:
            return
        classifications = classify_timestamp(group, config)
        for index, strategies in classifications.items():
            for strategy in strategies:
                raw_candidates.append(candidate_record(group[index], strategy, config))
        timestamp_groups += 1

    group: list[Mapping[str, Any]] = []
    group_timestamp: str | None = None
    with gzip.open(root / str(authority["source_dataset"]), "rt", encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            row = _mapping(json.loads(line), f"source:{line_number}")
            current_timestamp = str(row["timestamp"])
            if group_timestamp is not None and current_timestamp != group_timestamp:
                process_group(group)
                group = []
            group_timestamp = current_timestamp
            current_ms = timestamp_ms(current_timestamp)
            rate, age = _funding_at(funding.get(str(row["symbol"])), current_ms)
            prepared = prepare_row(row, funding_rate=rate, funding_age_ms=age)
            group.append(prepared)
            for strategy, profile in profile_by_strategy.items():
                populations[(str(row["side"]), partition_name(current_timestamp, config["temporal_protocol"]), strategy)].append(
                    _population_record(row, profile, config)
                )
            source_rows += 1
    process_group(group)

    spacing = int(config["temporal_protocol"]["minimum_event_spacing_minutes_per_symbol_side_family"])
    candidates = apply_event_spacing(raw_candidates, spacing)
    output_root.mkdir(parents=True, exist_ok=True)
    dataset_path = output_root / "opportunities.jsonl.gz"
    temporary = dataset_path.with_suffix(".jsonl.gz.tmp")
    with _deterministic_gzip_text(temporary) as target:
        for row in candidates:
            target.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")
    os.replace(temporary, dataset_path)
    os.chmod(dataset_path, 0o600)

    grouped: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in candidates:
        grouped[(str(row["side"]), str(row["strategy"]))].append(row)
    reports: dict[str, Any] = {}
    eligible: list[str] = []
    base_seed = int(config["controls"]["random_seed"])
    strategy_order = list(profile_by_strategy)
    for side in authority["sides"]:
        for strategy in strategy_order:
            identity = f"{side}::{strategy}"
            rows = grouped[(str(side), strategy)]
            periods = {
                partition: [row for row in rows if row["partition"] == partition]
                for partition in ("DISCOVERY", "VALIDATION", "FINAL_HOLDOUT")
            }
            holdout_population = populations[(str(side), "FINAL_HOLDOUT", strategy)]
            random_rows = matched_control(
                holdout_population,
                len(periods["FINAL_HOLDOUT"]),
                seed=base_seed + (100 if side == "SHORT" else 0) + strategy_order.index(strategy),
            )
            assessment = gate_assessment(periods, random_rows, config["gate"])
            if assessment["passed"]:
                eligible.append(identity)
            current_ts_same_events = [
                {**row, "protected_net_return": float(row["current_ts_net_return"])}
                for row in periods["FINAL_HOLDOUT"]
            ]
            reports[identity] = {
                **assessment,
                "event_counts": {name: len(values) for name, values in periods.items()},
                "controls": {
                    "no_trade": {"mean_protected_net": 0.0},
                    "unconditional_side_period": economic_summary(holdout_population),
                    "random_side_period_matched_count": economic_summary(random_rows),
                    "current_ts_exit_same_events": economic_summary(current_ts_same_events),
                },
                "exit_reason_counts": dict(
                    sorted(Counter(str(row["protected_exit_reason"]) for row in periods["FINAL_HOLDOUT"]).items())
                ),
            }

    report = {
        "schema_id": "aegis-alpha-laboratory-v21-result-v1",
        "experiment_id": str(config["experiment_id"]),
        "config_content_sha256": Sha256HashProvider().digest_value(config),
        "source_dataset_sha256": sha256_file(root / str(authority["source_dataset"])),
        "funding_database_sha256": sha256_file(root / str(authority["funding_database"])),
        "source_rows": source_rows,
        "timestamp_groups": timestamp_groups,
        "raw_candidate_rows": len(raw_candidates),
        "spaced_candidate_rows": len(candidates),
        "event_spacing_minutes": spacing,
        "dataset": str(dataset_path),
        "dataset_sha256": sha256_file(dataset_path),
        "strategy_reports": reports,
        "eligible_side_strategies": eligible,
        "carry_hypothesis": {
            "status": "DATA_GAP",
            "funding_available": funding_inventory["funding_rows"] > 0,
            "spot_history_available": funding_inventory["spot_history_present"],
            "basis_history_available": funding_inventory["basis_history_present"],
            "proxy_substituted": False,
        },
        "funding_inventory": funding_inventory,
        "V21_READY_FOR_MODELING": bool(eligible),
        "V21_READY_FOR_SHADOW": False,
        "V21_READY_FOR_LIVE": False,
        "model_training_executed": False,
        "model_exported": False,
        "holdout_opened_once": True,
        "holdout_reusable_for_v21_tuning": False,
        "selection_effect": "NONE",
        "live_changed": False,
        "shadow_changed": False,
        "exchange_calls": 0,
        "exchange_mutations": 0,
    }
    report_path = output_root / "result.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.chmod(report_path, 0o600)
    manifest = {
        "schema_id": "aegis-alpha-laboratory-v21-manifest-v1",
        "dataset": str(dataset_path),
        "dataset_sha256": report["dataset_sha256"],
        "report": str(report_path),
        "report_sha256": sha256_file(report_path),
        "source_rows": source_rows,
        "spaced_candidate_rows": len(candidates),
        "exchange_calls": 0,
        "exchange_mutations": 0,
    }
    manifest_path = output_root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.chmod(manifest_path, 0o600)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("config/experiments/aegis_alpha_laboratory_v21.yaml"))
    parser.add_argument("--output-root", type=Path, default=Path("data/alpha_laboratory_v21"))
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    config_path = args.config if args.config.is_absolute() else root / args.config
    output_root = args.output_root if args.output_root.is_absolute() else root / args.output_root
    config = _mapping(yaml.safe_load(config_path.read_text(encoding="utf-8")), "config")
    report = run(root, config, output_root)
    print(json.dumps({
        "spaced_candidate_rows": report["spaced_candidate_rows"],
        "eligible_side_strategies": report["eligible_side_strategies"],
        "V21_READY_FOR_MODELING": report["V21_READY_FOR_MODELING"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

