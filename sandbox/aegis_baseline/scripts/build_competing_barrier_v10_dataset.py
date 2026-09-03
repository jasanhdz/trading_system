#!/usr/bin/env python3
"""Build immutable outcome-only competing-barrier V10 evidence."""

from __future__ import annotations

import argparse
import gzip
import json
import os
from collections import Counter
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Mapping

import yaml

from aegis.research.competing_barrier_v10 import (
    contracts_from_config,
    evaluate_barrier_path,
    primary_direction_label,
)
from aegis.utils import Sha256HashProvider, sha256_file
from build_decomposed_entry_v9_dataset import _pairs
from train_long_entry_v21_shadow import _mapping, _source_series


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
    source_manifest = root / str(authority["source_manifest"])
    source_validation = root / str(authority["source_validation"])
    expected = {
        source_dataset: str(authority["source_dataset_sha256"]),
        source_manifest: str(authority["source_manifest_sha256"]),
        source_validation: str(authority["source_validation_sha256"]),
    }
    for path, digest in expected.items():
        if sha256_file(path) != digest:
            raise ValueError(f"V10 authority hash mismatch: {path.name}")
    source_inventory = _mapping(json.loads(source_manifest.read_text()), "manifest")
    contracts = contracts_from_config(config)
    primary_name = str(config["models"]["direction"]["source_contract"])
    if primary_name not in {contract.name for contract in contracts}:
        raise ValueError("V10 primary direction contract is missing")
    history_bars = int(authority["history_bars"])
    horizon_bars = max(contract.horizon_bars for contract in contracts)
    candles, common, candle_inventory = _source_series(
        root / str(authority["candle_database"]),
        root / str(authority["public_candle_delta"]),
        lookback_days=int(authority["lookback_days"]),
        history_bars=history_bars,
        horizon_bars=horizon_bars,
    )
    index_by_open = {timestamp: index for index, timestamp in enumerate(common)}
    spacing = timedelta(minutes=int(config["episodes"]["spacing_minutes"]))
    last_episode: dict[str, datetime] = {}
    counts: Counter[str] = Counter()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    rows = episodes = source_pairs = 0
    first_timestamp: str | None = None
    last_timestamp: str | None = None
    with gzip.open(source_dataset, "rt", encoding="utf-8") as source, gzip.open(
        temporary, "wt", encoding="utf-8", newline="\n"
    ) as target:
        for long_row, short_row in _pairs(source):
            source_pairs += 1
            timestamp = datetime.fromisoformat(str(long_row["timestamp"]))
            symbol = str(long_row["symbol"])
            previous = last_episode.get(symbol)
            if previous is not None and timestamp - previous < spacing:
                continue
            if maximum_pairs is not None and episodes >= maximum_pairs:
                break
            index = index_by_open.get(timestamp)
            if index is None:
                raise ValueError("V10 candle source does not cover source pair")
            future = candles[symbol][index : index + horizon_bars]
            if len(future) != horizon_bars:
                raise ValueError("V10 future candle path is incomplete")
            if not all(
                abs(float(row["entry_price"]) - float(future[0].open)) <= max(
                    1e-10, abs(float(future[0].open)) * 1e-10
                )
                for row in (long_row, short_row)
            ):
                raise ValueError("V10 next-bar entry authority mismatch")
            evaluated: dict[str, dict[str, Mapping[str, Any]]] = {}
            for row in (long_row, short_row):
                side = str(row["side"])
                evaluated[side] = {
                    contract.name: evaluate_barrier_path(
                        side=side,
                        entry_price=float(row["entry_price"]),
                        future_bars=future,
                        contract=contract,
                    )
                    for contract in contracts
                }
            direction = primary_direction_label(
                evaluated["LONG"][primary_name], evaluated["SHORT"][primary_name]
            )
            for row in (long_row, short_row):
                side = str(row["side"])
                enriched = {
                    **row,
                    "v10_episode": True,
                    "v10_episode_spacing_minutes": int(
                        config["episodes"]["spacing_minutes"]
                    ),
                    "v10_direction_label": direction,
                    "v10_primary_contract": primary_name,
                    "v10_contract_outcomes": evaluated[side],
                    "selection_effect": "NONE",
                    "exchange_authority": False,
                    "exchange_calls": 0,
                    "exchange_mutations": 0,
                }
                target.write(json.dumps(enriched, sort_keys=True, separators=(",", ":")))
                target.write("\n")
                rows += 1
                for name, result in evaluated[side].items():
                    counts[f"{side}::{name}::{result['outcome']}"] += 1
            counts[f"DIRECTION::{direction}"] += 1
            last_episode[symbol] = timestamp
            episodes += 1
            first_timestamp = first_timestamp or str(long_row["timestamp"])
            last_timestamp = str(long_row["timestamp"])
            if episodes % 10000 == 0:
                print(json.dumps({"episodes": episodes, "rows": rows}), flush=True)
    if not rows:
        raise ValueError("V10 produced no rows")
    os.replace(temporary, output)
    os.chmod(output, 0o600)
    manifest = {
        "schema_id": "aegis-competing-barrier-v10-dataset-manifest-v1",
        "config_content_sha256": Sha256HashProvider().digest_value(config),
        "source_dataset": str(source_dataset.resolve()),
        "source_dataset_sha256": sha256_file(source_dataset),
        "source_manifest_sha256": sha256_file(source_manifest),
        "source_validation_sha256": sha256_file(source_validation),
        "dataset": str(output.resolve()),
        "dataset_sha256": sha256_file(output),
        "rows": rows,
        "episodes": episodes,
        "source_pairs_scanned": source_pairs,
        "evidence_start": first_timestamp,
        "evidence_end": last_timestamp,
        "symbols": source_inventory["symbols"],
        "sides": source_inventory["sides"],
        "contracts": [contract.__dict__ for contract in contracts],
        "outcome_counts": dict(sorted(counts.items())),
        "episode_rule": "FIRST_ELIGIBLE_THEN_FIXED_SPACING",
        "episode_spacing_minutes": int(config["episodes"]["spacing_minutes"]),
        "labels_use_only_future_ohlc": True,
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
        default=Path("config/experiments/aegis_competing_barrier_v10_research.yaml"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/competing_barrier_v10/canonical_dataset.jsonl.gz"),
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/competing_barrier_v10/dataset_manifest.json"),
    )
    parser.add_argument("--maximum-pairs", type=int)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    resolve = lambda path: path if path.is_absolute() else root / path
    manifest = build_dataset(
        root=root,
        config=_mapping(yaml.safe_load(resolve(args.config).read_text()), "config"),
        output=resolve(args.output),
        manifest_path=resolve(args.manifest),
        maximum_pairs=args.maximum_pairs,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
