#!/usr/bin/env python3
"""Build immutable decomposed-direction/timing/trajectory V9 evidence."""

from __future__ import annotations

import argparse
import gzip
import json
import os
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Iterator, Mapping

import yaml

from aegis.research.decomposed_entry_v9 import (
    DirectionLabelContract,
    TimingFailure,
    TimingLabelContract,
    direction_label,
    rolling_four_hour_context,
    timing_labels,
    trajectory_targets,
    v9_feature_vectors,
)
from aegis.utils import Sha256HashProvider, sha256_file
from training.train_long_entry_v21_shadow import _mapping, _source_series


def _pairs(handle: Any) -> Iterator[tuple[dict[str, Any], dict[str, Any]]]:
    pending: dict[str, Any] | None = None
    for line_number, line in enumerate(handle, start=1):
        if not line.strip():
            continue
        row = dict(_mapping(json.loads(line), f"source:{line_number}"))
        if pending is None:
            pending = row
            continue
        identity = (str(row["timestamp"]), str(row["symbol"]))
        pending_identity = (str(pending["timestamp"]), str(pending["symbol"]))
        if identity != pending_identity or {str(pending["side"]), str(row["side"])} != {
            "LONG",
            "SHORT",
        }:
            raise ValueError("V9 source rows are not complete adjacent direction pairs")
        yield (pending, row) if pending["side"] == "LONG" else (row, pending)
        pending = None
    if pending is not None:
        raise ValueError("V9 source ended with an incomplete direction pair")


def _direction_contract(config: Mapping[str, Any]) -> DirectionLabelContract:
    raw = _mapping(config["direction"], "direction")
    costs = _mapping(config["costs"], "costs")
    return DirectionLabelContract(
        source_base_cost_fraction=float(costs["source_base_round_trip_fraction"]),
        stress_cost_fraction=float(raw["stress_round_trip_fraction"]),
        minimum_edge_fraction=float(raw["minimum_edge_over_opposite_fraction"]),
    )


def _timing_contract(config: Mapping[str, Any]) -> TimingLabelContract:
    raw = _mapping(config["timing"], "timing")
    return TimingLabelContract(
        clean_mae_fraction=float(raw["clean_mae_fraction"]),
        clean_positive_bar=int(raw["clean_positive_bar"]),
        overextension_atr=float(raw["overextension_atr"]),
        exhaustion_atr=float(raw["exhaustion_atr"]),
        weak_volume_ratio=float(raw["weak_volume_ratio"]),
    )


def build_dataset(
    *,
    root: Path,
    config: Mapping[str, Any],
    output: Path,
    manifest_path: Path,
    maximum_pairs: int | None = None,
) -> Mapping[str, Any]:
    authority = _mapping(config["authority"], "authority")
    source_dataset = root / str(authority["source_dataset"])
    source_manifest_path = root / str(authority["source_manifest"])
    source_validation_path = root / str(authority["source_validation"])
    expected = {
        source_dataset: str(authority["source_dataset_sha256"]),
        source_manifest_path: str(authority["source_manifest_sha256"]),
        source_validation_path: str(authority["source_validation_sha256"]),
    }
    for path, digest in expected.items():
        if sha256_file(path) != digest:
            raise ValueError(f"V9 authority hash mismatch: {path.name}")
    source_manifest = _mapping(json.loads(source_manifest_path.read_text()), "manifest")
    history_bars = int(authority["history_bars"])
    candles, common, candle_inventory = _source_series(
        root / str(authority["candle_database"]),
        root / str(authority["public_candle_delta"]),
        lookback_days=int(authority["lookback_days"]),
        history_bars=history_bars,
        horizon_bars=24,
    )
    index_by_open = {timestamp: index for index, timestamp in enumerate(common)}
    direction_contract = _direction_contract(config)
    timing_contract = _timing_contract(config)
    catastrophic = float(config["trajectory"]["catastrophic_stress_fraction"])
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    rows = pairs = 0
    first_timestamp: str | None = None
    last_timestamp: str | None = None
    direction_counts: Counter[str] = Counter()
    timing_counts: Counter[str] = Counter()
    trajectory_counts: Counter[str] = Counter()
    with gzip.open(source_dataset, "rt", encoding="utf-8") as source, gzip.open(
        temporary, "wt", encoding="utf-8", newline="\n"
    ) as target:
        for long_row, short_row in _pairs(source):
            if maximum_pairs is not None and pairs >= maximum_pairs:
                break
            timestamp = datetime.fromisoformat(str(long_row["timestamp"]))
            symbol = str(long_row["symbol"])
            index = index_by_open.get(timestamp)
            if index is None:
                raise ValueError("V9 candle source does not cover source pair")
            history = candles[symbol][index - history_bars : index]
            if len(history) != history_bars:
                raise ValueError("V9 candle history is incomplete")
            rolling = rolling_four_hour_context(history)
            label = direction_label(long_row, short_row, direction_contract)
            enriched_pair = []
            direction_vectors = []
            for row in (long_row, short_row):
                features, direction_features, context = v9_feature_vectors(row, rolling)
                timing = timing_labels(row, context, timing_contract)
                trajectory = trajectory_targets(row, catastrophic)
                enriched_pair.append(
                    {
                        **row,
                        "v9_features": features,
                        "v9_direction_features": direction_features,
                        "v9_causal_context": context,
                        "v9_direction_label": label,
                        "v9_timing_labels": timing,
                        "v9_trajectory_targets": trajectory,
                        "selection_effect": "NONE",
                        "exchange_authority": False,
                        "exchange_mutations": 0,
                    }
                )
                direction_vectors.append(direction_features)
                for name in TimingFailure:
                    timing_counts[f"{row['side']}::{name.value}"] += int(
                        bool(timing[name.value])
                    )
                timing_counts[f"{row['side']}::CLEAN_TIMING"] += int(
                    bool(timing["CLEAN_TIMING"])
                )
                trajectory_counts[f"{row['side']}::POSITIVE_CURRENT_TS_STRESS"] += int(
                    bool(trajectory["positive_current_ts_stress"])
                )
                trajectory_counts[
                    f"{row['side']}::CATASTROPHIC_CURRENT_TS_STRESS"
                ] += int(bool(trajectory["catastrophic_current_ts_stress"]))
            if direction_vectors[0] != direction_vectors[1]:
                raise ValueError(
                    "V9 direction-neutral vectors differ by candidate side"
                )
            for enriched in enriched_pair:
                target.write(
                    json.dumps(enriched, sort_keys=True, separators=(",", ":"))
                )
                target.write("\n")
                rows += 1
            pairs += 1
            direction_counts[str(label["label"])] += 1
            first_timestamp = first_timestamp or str(long_row["timestamp"])
            last_timestamp = str(long_row["timestamp"])
            if pairs % 10000 == 0:
                print(
                    json.dumps({"pairs": pairs, "rows": rows}, sort_keys=True),
                    flush=True,
                )
    if not rows:
        raise ValueError("V9 produced no rows")
    os.replace(temporary, output)
    os.chmod(output, 0o600)
    manifest = {
        "schema_id": "aegis-decomposed-entry-v9-dataset-manifest-v1",
        "config_content_sha256": Sha256HashProvider().digest_value(config),
        "source_dataset": str(source_dataset.resolve()),
        "source_dataset_sha256": sha256_file(source_dataset),
        "source_manifest": str(source_manifest_path.resolve()),
        "source_manifest_sha256": sha256_file(source_manifest_path),
        "source_validation_sha256": sha256_file(source_validation_path),
        "dataset": str(output.resolve()),
        "dataset_sha256": sha256_file(output),
        "rows": rows,
        "pairs": pairs,
        "evidence_start": first_timestamp,
        "evidence_end": last_timestamp,
        "symbols": source_manifest["symbols"],
        "sides": source_manifest["sides"],
        "v9_feature_count": len(features),
        "v9_direction_feature_count": len(direction_features),
        "direction_label_counts": dict(sorted(direction_counts.items())),
        "timing_label_counts": dict(sorted(timing_counts.items())),
        "trajectory_label_counts": dict(sorted(trajectory_counts.items())),
        "direction_pair_mismatches": 0,
        "candle_source": candle_inventory,
        "exchange_calls": 0,
        "exchange_mutations": 0,
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_manifest = manifest_path.with_suffix(".json.tmp")
    temporary_manifest.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    os.replace(temporary_manifest, manifest_path)
    os.chmod(manifest_path, 0o600)
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config/experiments/aegis_decomposed_entry_v9_research.yaml"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/decomposed_entry_v9/canonical_dataset.jsonl.gz"),
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/decomposed_entry_v9/dataset_manifest.json"),
    )
    parser.add_argument("--maximum-pairs", type=int)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    resolve = lambda path: path if path.is_absolute() else root / path
    result = build_dataset(
        root=root,
        config=_mapping(yaml.safe_load(resolve(args.config).read_text()), "config"),
        output=resolve(args.output),
        manifest_path=resolve(args.manifest),
        maximum_pairs=args.maximum_pairs,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
