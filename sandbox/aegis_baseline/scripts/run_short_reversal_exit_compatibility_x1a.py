#!/usr/bin/env python3
"""Run corrected-source SHORT reversal exit compatibility X1A."""

from __future__ import annotations

import argparse
import gzip
import io
import json
import os
import random
from collections import Counter
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Mapping

import yaml

from aegis.research.alpha_laboratory_v21 import (
    AlphaStrategy,
    apply_event_spacing,
    classify_timestamp,
    prepare_row,
    timestamp_ms,
)
from aegis.research.causal_opportunity_v20 import economic_summary
from aegis.research.feature_information_v14 import TAKER_FLOW_FEATURE_NAMES
from aegis.research.short_reversal_exit_x1 import assessment, profile_record
from aegis.research.split_public_taker_flow import split_public_flow_lookup
from aegis.utils import Sha256HashProvider, sha256_file


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"X1A {name} must be a mapping")
    return value


@contextmanager
def _deterministic_gzip_text(path: Path):
    with path.open("wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as compressed:
            with io.TextIOWrapper(compressed, encoding="utf-8", newline="\n") as text:
                yield text


def run(root: Path, config: Mapping[str, Any], output_root: Path) -> Mapping[str, Any]:
    authority = _mapping(config["authority"], "authority")
    bindings = {
        root / str(authority["source_dataset"]): str(authority["source_dataset_sha256"]),
        root / str(authority["source_manifest"]): str(authority["source_manifest_sha256"]),
        root / str(authority["public_candle_delta"]): str(authority["public_candle_delta_sha256"]),
        root / str(authority["public_microstructure_database"]): str(authority["public_microstructure_database_sha256"]),
        root / str(authority["v21_config"]): str(authority["v21_config_sha256"]),
        root / str(authority["x1_config"]): str(authority["x1_config_sha256"]),
        root / str(authority["protection_replay"]): str(authority["protection_replay_sha256"]),
    }
    for path, expected in bindings.items():
        if not path.is_file() or sha256_file(path) != expected:
            raise RuntimeError(f"X1A authority mismatch: {path}")
    v21_config = _mapping(
        yaml.safe_load((root / str(authority["v21_config"])).read_text(encoding="utf-8")),
        "v21_config",
    )
    if config["entry_rule"]["modifications"] != "NONE":
        raise RuntimeError("X1A entry rule modification is prohibited")

    period = config["future_evidence"]
    start, end = str(period["start_inclusive"]), str(period["end_inclusive"])
    source_path = root / str(authority["source_dataset"])
    source_rows: list[Mapping[str, Any]] = []
    required_ms = set()
    with gzip.open(source_path, "rt", encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            row = _mapping(json.loads(line), f"source:{line_number}")
            timestamp = str(row["timestamp"])
            if start <= timestamp <= end:
                source_rows.append(row)
                required_ms.add(timestamp_ms(timestamp))
    flow, flow_inventory = split_public_flow_lookup(
        candle_delta=root / str(authority["public_candle_delta"]),
        microstructure_database=root / str(authority["public_microstructure_database"]),
        required_entry_timestamps_ms=sorted(required_ms),
    )

    groups: dict[str, list[Mapping[str, Any]]] = {}
    population: list[Mapping[str, Any]] = []
    omitted_without_flow = 0
    for source in source_rows:
        if str(source["side"]) != "SHORT":
            continue
        timestamp = str(source["timestamp"])
        values = flow.get((timestamp_ms(timestamp), str(source["symbol"])))
        if values is None:
            omitted_without_flow += 1
            continue
        enriched = {
            **source,
            "v14_taker_flow_feature_names": list(TAKER_FLOW_FEATURE_NAMES),
            "v14_taker_flow_features": list(values),
            "v14_flow_source": "SPLIT_PUBLIC_KLINE_VOLUME_AND_TAKER_BUY_BASE",
            "selection_effect": "NONE",
            "exchange_authority": False,
            "exchange_calls": 0,
            "exchange_mutations": 0,
        }
        prepared = prepare_row(enriched, funding_rate=None, funding_age_ms=None)
        groups.setdefault(timestamp, []).append(prepared)
        population.append(profile_record(prepared, "CURRENT_TS"))

    raw_candidates: list[Mapping[str, Any]] = []
    prepared_lookup: dict[tuple[str, str], Mapping[str, Any]] = {}
    for timestamp, rows in sorted(groups.items()):
        if len(rows) != 11:
            continue
        classifications = classify_timestamp(rows, v21_config)
        for index, strategies in classifications.items():
            if AlphaStrategy.EXTREME_REVERSAL in strategies:
                prepared = rows[index]
                record = profile_record(prepared, "CURRENT_TS")
                raw_candidates.append(record)
                prepared_lookup[(str(record["timestamp"]), str(record["symbol"]))] = prepared
    spacing = int(period["minimum_event_spacing_minutes_per_symbol"])
    spaced = apply_event_spacing(raw_candidates, spacing)
    identities = sorted(
        (str(row["timestamp"]), str(row["symbol"])) for row in spaced
    )
    candidate = [profile_record(prepared_lookup[key], "CURRENT_TS") for key in identities]
    profiles = {
        name: [profile_record(prepared_lookup[key], name) for key in identities]
        for name in ("LOCK_AT_5_ROE", "LOCK_AT_10_ROE", "LOCK_AT_20_ROE")
    }
    random_control = random.Random(int(config["controls"]["random_seed"])).sample(
        population, len(candidate)
    )
    result = assessment(
        candidate=candidate,
        v21_exit=profiles["LOCK_AT_5_ROE"],
        random_control=random_control,
        diagnostic_profiles={
            "LOCK_AT_10_ROE": profiles["LOCK_AT_10_ROE"],
            "LOCK_AT_20_ROE": profiles["LOCK_AT_20_ROE"],
        },
        config=config,
    )

    output_root.mkdir(parents=True, exist_ok=True)
    dataset_path = output_root / "short_reversal_candidates.jsonl.gz"
    temporary = dataset_path.with_suffix(".jsonl.gz.tmp")
    with _deterministic_gzip_text(temporary) as target:
        for row in candidate:
            target.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")
    os.replace(temporary, dataset_path)
    os.chmod(dataset_path, 0o600)
    report = {
        "schema_id": "aegis-short-reversal-exit-compatibility-x1a-result-v1",
        "experiment_id": str(config["experiment_id"]),
        "config_content_sha256": Sha256HashProvider().digest_value(config),
        "source_dataset_sha256": sha256_file(source_path),
        "public_candle_delta_sha256": sha256_file(root / str(authority["public_candle_delta"])),
        "public_microstructure_database_sha256": sha256_file(root / str(authority["public_microstructure_database"])),
        "future_evidence_start": start,
        "future_evidence_end": end,
        "future_source_rows": len(source_rows),
        "future_short_rows_with_flow": len(population),
        "omitted_short_rows_without_flow": omitted_without_flow,
        "complete_timestamp_groups": sum(len(rows) == 11 for rows in groups.values()),
        "raw_candidate_events": len(raw_candidates),
        "spaced_candidate_events": len(candidate),
        "candidate_dataset": str(dataset_path),
        "candidate_dataset_sha256": sha256_file(dataset_path),
        "flow_inventory": flow_inventory,
        "assessment": result,
        "candidate_exit_reason_counts": dict(
            sorted(Counter(str(row["protected_exit_reason"]) for row in candidate).items())
        ),
        "unconditional_short_future_current_ts": economic_summary(population),
        "entry_rule_modifications": "NONE",
        "physical_flow_source_change": "EXPLICIT",
        "holdout_opened_once": bool(population),
        "holdout_reusable_for_x1a_tuning": False,
        "X1A_READY_FOR_SHADOW": bool(result["passed"]),
        "X1A_READY_FOR_LIVE": False,
        "model_training_executed": False,
        "model_exported": False,
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
        "schema_id": "aegis-short-reversal-exit-compatibility-x1a-manifest-v1",
        "candidate_dataset": str(dataset_path),
        "candidate_dataset_sha256": report["candidate_dataset_sha256"],
        "result": str(report_path),
        "result_sha256": sha256_file(report_path),
        "candidate_events": len(candidate),
        "exchange_calls": 0,
        "exchange_mutations": 0,
    }
    manifest_path = output_root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.chmod(manifest_path, 0o600)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("config/experiments/aegis_short_reversal_exit_compatibility_x1a.yaml"))
    parser.add_argument("--output-root", type=Path, default=Path("data/short_reversal_exit_compatibility_x1a"))
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    config_path = args.config if args.config.is_absolute() else root / args.config
    output_root = args.output_root if args.output_root.is_absolute() else root / args.output_root
    config = _mapping(yaml.safe_load(config_path.read_text(encoding="utf-8")), "config")
    report = run(root, config, output_root)
    print(json.dumps({
        "candidate_events": report["spaced_candidate_events"],
        "passed": report["assessment"]["passed"],
        "X1A_READY_FOR_SHADOW": report["X1A_READY_FOR_SHADOW"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

