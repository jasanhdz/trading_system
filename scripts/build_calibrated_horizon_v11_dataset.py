#!/usr/bin/env python3
"""Build immutable causal-regime and clean-entry V11 evidence."""

from __future__ import annotations

import argparse
import gzip
import json
import os
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping

import yaml

from aegis.research.calibrated_horizon_v11 import (
    causal_regime,
    clean_entry_diagnostics,
)
from aegis.utils import Sha256HashProvider, sha256_file
from train_long_entry_v21_shadow import _mapping, _source_series


def build_dataset(
    *,
    root: Path,
    config: Mapping[str, Any],
    output: Path,
    manifest_path: Path,
    maximum_rows: int | None = None,
) -> Mapping[str, Any]:
    authority = _mapping(config["authority"], "authority")
    paths = {
        root / str(authority["source_dataset"]): str(authority["source_dataset_sha256"]),
        root / str(authority["source_manifest"]): str(authority["source_manifest_sha256"]),
        root / str(authority["source_validation"]): str(authority["source_validation_sha256"]),
        root / str(authority["source_config"]): str(authority["source_config_sha256"]),
    }
    for path, expected in paths.items():
        if sha256_file(path) != expected:
            raise ValueError(f"V11 authority hash mismatch: {path.name}")
    source_dataset = root / str(authority["source_dataset"])
    source_manifest_path = root / str(authority["source_manifest"])
    source_manifest = _mapping(json.loads(source_manifest_path.read_text()), "manifest")
    horizon = int(authority["horizon_bars"])
    candles, common, candle_inventory = _source_series(
        root / str(authority["candle_database"]),
        root / str(authority["public_candle_delta"]),
        lookback_days=int(authority["lookback_days"]),
        history_bars=int(authority["history_bars"]),
        horizon_bars=horizon,
    )
    index_by_open = {timestamp: index for index, timestamp in enumerate(common)}
    primary = str(config["labels"]["clean_entry"]["source_contract"])
    clean_config = _mapping(config["labels"]["clean_entry"], "clean_entry")
    regime_config = _mapping(config["causal_regime"], "causal_regime")
    severe_cost = float(config["utility"]["severe_round_trip_fraction"])
    counts: Counter[str] = Counter()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    rows = 0
    first_timestamp: str | None = None
    last_timestamp: str | None = None
    with gzip.open(source_dataset, "rt", encoding="utf-8") as source, gzip.open(
        temporary, "wt", encoding="utf-8", newline="\n"
    ) as target:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            if maximum_rows is not None and rows >= maximum_rows:
                break
            row = dict(_mapping(json.loads(line), f"source:{line_number}"))
            timestamp = datetime.fromisoformat(str(row["timestamp"]))
            symbol = str(row["symbol"])
            side = str(row["side"])
            index = index_by_open.get(timestamp)
            if index is None:
                raise ValueError("V11 candle source does not cover episode")
            future = candles[symbol][index : index + horizon]
            if len(future) != horizon:
                raise ValueError("V11 future path is incomplete")
            outcomes = _mapping(row["v10_contract_outcomes"], "outcomes")
            primary_outcome = _mapping(outcomes[primary], "primary_outcome")
            diagnostics = clean_entry_diagnostics(
                side=side,
                entry_price=float(row["entry_price"]),
                future_bars=future,
                primary_outcome=primary_outcome,
                maximum_mae_fraction_of_barrier=float(
                    clean_config["maximum_pre_event_mae_fraction_of_adverse_barrier"]
                ),
                maximum_event_bar=int(clean_config["maximum_favorable_event_bar"]),
                severe_cost_fraction=severe_cost,
            )
            regime = causal_regime(
                _mapping(row["v9_causal_context"], "causal_context"), regime_config
            )
            enriched = {
                **row,
                "v11_causal_regime": regime,
                "v11_clean_entry_label": bool(diagnostics["clean_entry"]),
                "v11_path_diagnostics": diagnostics,
                "selection_effect": "NONE",
                "exchange_authority": False,
                "exchange_calls": 0,
                "exchange_mutations": 0,
            }
            target.write(json.dumps(enriched, sort_keys=True, separators=(",", ":")))
            target.write("\n")
            counts[f"{side}::REGIME::{regime}"] += 1
            counts[f"{side}::CLEAN::{bool(diagnostics['clean_entry'])}"] += 1
            rows += 1
            first_timestamp = first_timestamp or str(row["timestamp"])
            last_timestamp = str(row["timestamp"])
            if rows % 20000 == 0:
                print(json.dumps({"rows": rows}, sort_keys=True), flush=True)
    if not rows:
        raise ValueError("V11 produced no rows")
    os.replace(temporary, output)
    os.chmod(output, 0o600)
    manifest = {
        "schema_id": "aegis-calibrated-horizon-v11-dataset-manifest-v1",
        "config_content_sha256": Sha256HashProvider().digest_value(config),
        "source_dataset": str(source_dataset.resolve()),
        "source_dataset_sha256": sha256_file(source_dataset),
        "source_manifest_sha256": sha256_file(source_manifest_path),
        "source_validation_sha256": sha256_file(root / str(authority["source_validation"])),
        "dataset": str(output.resolve()),
        "dataset_sha256": sha256_file(output),
        "rows": rows,
        "episodes": rows // 2,
        "evidence_start": first_timestamp,
        "evidence_end": last_timestamp,
        "symbols": source_manifest["symbols"],
        "sides": source_manifest["sides"],
        "primary_contract": primary,
        "label_and_regime_counts": dict(sorted(counts.items())),
        "barrier_outcomes_reused_unchanged": True,
        "clean_label_uses_only_future_ohlc": True,
        "causal_regime_uses_only_pre_entry_context": True,
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
        default=Path("config/experiments/aegis_calibrated_horizon_v11_research.yaml"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/calibrated_horizon_v11/canonical_dataset.jsonl.gz"),
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/calibrated_horizon_v11/dataset_manifest.json"),
    )
    parser.add_argument("--maximum-rows", type=int)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    resolve = lambda path: path if path.is_absolute() else root / path
    result = build_dataset(
        root=root,
        config=_mapping(yaml.safe_load(resolve(args.config).read_text()), "config"),
        output=resolve(args.output),
        manifest_path=resolve(args.manifest),
        maximum_rows=args.maximum_rows,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
