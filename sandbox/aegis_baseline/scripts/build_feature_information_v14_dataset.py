#!/usr/bin/env python3
"""Build V14 evidence with strictly pre-entry historical taker flow."""

from __future__ import annotations

import argparse
import gzip
import io
import json
import os
import sqlite3
from collections import defaultdict, deque
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Mapping

import yaml

from aegis.features import CANONICAL_SYMBOLS
from aegis.research.feature_information_v14 import (
    TAKER_FLOW_FEATURE_NAMES,
    local_taker_flow,
    market_taker_flow,
    taker_imbalance,
)
from aegis.utils import Sha256HashProvider, sha256_file
from training.train_long_entry_v21_shadow import _mapping


def _timestamp(value: str) -> datetime:
    parsed = datetime.fromisoformat(value)
    return parsed.replace(tzinfo=timezone.utc) if parsed.tzinfo is None else parsed


def _database_symbol(symbol: str) -> str:
    return symbol.removesuffix("USDT") + "/USDT"


def _required_timestamps(source: Path) -> tuple[set[datetime], datetime, datetime]:
    values: set[datetime] = set()
    with gzip.open(source, "rt", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                values.add(_timestamp(str(json.loads(line)["timestamp"])))
    if not values:
        raise ValueError("V14 source dataset is empty")
    return values, min(values), max(values)


def _flow_lookup(
    database: Path,
    required: set[datetime],
    start: datetime,
    end: datetime,
) -> tuple[Mapping[tuple[datetime, str], tuple[float, ...]], Mapping[str, Any]]:
    local_by_time: dict[datetime, dict[str, Mapping[str, float]]] = defaultdict(dict)
    connection = sqlite3.connect(f"file:{database}?mode=ro", uri=True)
    try:
        market_rows = int(
            connection.execute("select count(*) from market_data").fetchone()[0]
        )
        for symbol in CANONICAL_SYMBOLS:
            history: deque[float] = deque(maxlen=24)
            rows = connection.execute(
                """
                select timestamp, volume, buy_volume
                from ohlcv_data
                where symbol = ? and timeframe = '5m'
                  and timestamp >= ? and timestamp < ?
                order by timestamp
                """,
                (
                    _database_symbol(symbol),
                    (start - timedelta(hours=3)).strftime("%Y-%m-%d %H:%M:%S.%f"),
                    end.strftime("%Y-%m-%d %H:%M:%S.%f"),
                ),
            )
            for raw_timestamp, volume, buy_volume in rows:
                bar_time = _timestamp(str(raw_timestamp))
                history.append(taker_imbalance(float(volume), float(buy_volume)))
                entry_time = bar_time + timedelta(minutes=5)
                if entry_time in required and len(history) == 24:
                    local_by_time[entry_time][symbol] = local_taker_flow(tuple(history))
    finally:
        connection.close()
    lookup = {}
    complete_times = 0
    for timestamp, local in local_by_time.items():
        if set(local) != set(CANONICAL_SYMBOLS):
            continue
        complete_times += 1
        for symbol in CANONICAL_SYMBOLS:
            values = {**local[symbol], **market_taker_flow(local, symbol=symbol)}
            lookup[(timestamp, symbol)] = tuple(
                float(values[name]) for name in TAKER_FLOW_FEATURE_NAMES
            )
    return lookup, {
        "database": str(database.resolve()),
        "database_sha256": sha256_file(database),
        "open_mode": "READ_ONLY",
        "market_data_rows": market_rows,
        "funding_and_open_interest_available": market_rows > 0,
        "required_timestamps": len(required),
        "complete_eleven_symbol_timestamps": complete_times,
        "flow_keys": len(lookup),
    }


def build_dataset(
    *,
    root: Path,
    config: Mapping[str, Any],
    output: Path,
    manifest_path: Path,
) -> Mapping[str, Any]:
    authority = _mapping(config["authority"], "authority")
    paths = {
        root
        / str(authority["source_dataset"]): str(authority["source_dataset_sha256"]),
        root
        / str(authority["source_manifest"]): str(authority["source_manifest_sha256"]),
        root
        / str(authority["source_v13_validation"]): str(
            authority["source_v13_validation_sha256"]
        ),
        root
        / str(authority["source_v13_config"]): str(
            authority["source_v13_config_sha256"]
        ),
        root
        / str(authority["candle_database"]): str(authority["candle_database_sha256"]),
    }
    for path, expected in paths.items():
        if sha256_file(path) != expected:
            raise ValueError(f"V14 authority hash mismatch: {path.name}")
    source = root / str(authority["source_dataset"])
    database = root / str(authority["candle_database"])
    required, start, end = _required_timestamps(source)
    flow, inventory = _flow_lookup(database, required, start, end)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    rows = 0
    omitted = 0
    timestamps: set[str] = set()
    symbols: set[str] = set()
    with gzip.open(source, "rt", encoding="utf-8") as source_handle, temporary.open(
        "wb"
    ) as raw_target, gzip.GzipFile(
        fileobj=raw_target, mode="wb", mtime=0
    ) as compressed_target, io.TextIOWrapper(
        compressed_target, encoding="utf-8", newline="\n"
    ) as target:
        for line_number, line in enumerate(source_handle, start=1):
            if not line.strip():
                continue
            row = dict(_mapping(json.loads(line), f"source:{line_number}"))
            timestamp = _timestamp(str(row["timestamp"]))
            symbol = str(row["symbol"])
            values = flow.get((timestamp, symbol))
            if values is None:
                omitted += 1
                continue
            enriched = {
                **row,
                "v14_taker_flow_feature_names": list(TAKER_FLOW_FEATURE_NAMES),
                "v14_taker_flow_features": list(values),
                "v14_flow_source": "CLOSED_5M_OHLCV_TAKER_BUY_VOLUME",
                "v14_flow_latest_bar": (timestamp - timedelta(minutes=5)).isoformat(),
                "selection_effect": "NONE",
                "exchange_authority": False,
                "exchange_calls": 0,
                "exchange_mutations": 0,
            }
            target.write(
                json.dumps(enriched, sort_keys=True, separators=(",", ":")) + "\n"
            )
            rows += 1
            timestamps.add(str(row["timestamp"]))
            symbols.add(symbol)
    if not rows:
        raise ValueError("V14 has no complete causal flow rows")
    os.replace(temporary, output)
    os.chmod(output, 0o600)
    manifest = {
        "schema_id": "aegis-feature-information-v14-dataset-manifest-v1",
        "config_content_sha256": Sha256HashProvider().digest_value(config),
        "source_dataset": str(source.resolve()),
        "source_dataset_sha256": sha256_file(source),
        "dataset": str(output.resolve()),
        "dataset_sha256": sha256_file(output),
        "rows": rows,
        "episodes": rows // 2,
        "omitted_source_rows_without_complete_flow": omitted,
        "coverage_fraction": rows / (rows + omitted),
        "evidence_start": min(timestamps),
        "evidence_end": max(timestamps),
        "symbols": sorted(symbols),
        "taker_flow_features": list(TAKER_FLOW_FEATURE_NAMES),
        "strict_pre_entry_only": True,
        "complete_eleven_symbol_timestamp_required": True,
        "source_inventory": inventory,
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
        default=Path("config/experiments/aegis_feature_information_v14_research.yaml"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/feature_information_v14/canonical_dataset.jsonl.gz"),
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/feature_information_v14/dataset_manifest.json"),
    )
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    resolve = lambda path: path if path.is_absolute() else root / path
    result = build_dataset(
        root=root,
        config=_mapping(yaml.safe_load(resolve(args.config).read_text()), "config"),
        output=resolve(args.output),
        manifest_path=resolve(args.manifest),
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
