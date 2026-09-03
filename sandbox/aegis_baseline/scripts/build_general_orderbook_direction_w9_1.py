#!/usr/bin/env python3
"""Build resumable causal W9.1 episodes from public Tardis files."""

from __future__ import annotations

import argparse
import hashlib
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
import yaml

from aegis.research.general_orderbook_direction_w9_1 import build_episodes


def file_path(raw: Path, kind: str, date: str, symbol: str) -> Path:
    return raw / f"binance-futures_{kind}_{date}_{symbol}.csv.gz"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def process_part(task: tuple[str, str, list[Path], Path, Path]) -> dict:
    symbol, date, source_paths, destination, audit_path = task
    episodes, audit = build_episodes(
        symbol=symbol,
        date=date,
        l2_path=source_paths[0],
        quotes_path=source_paths[1],
        trades_path=source_paths[2],
    )
    audit["sources"] = {
        path.name: {"bytes": path.stat().st_size, "sha256": sha256(path)}
        for path in source_paths
    }
    episodes.to_parquet(destination, index=False)
    audit_path.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")
    return audit


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--available-only", action="store_true")
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()
    root = args.root.resolve()
    config = yaml.safe_load((root / "config/experiments/aegis_general_orderbook_direction_w9_1.yaml").read_text())
    raw = root / "data/historical_orderbook_direction_w9_1/raw"
    output = root / "data/historical_orderbook_direction_w9_1/episodes"
    output.mkdir(parents=True, exist_ok=True)
    months = config["partitions"]["train_months"] + config["partitions"]["validation_months"]
    tasks: list[tuple[str, str, list[Path], Path, Path]] = []
    for month in months:
        date = f"{month}-01"
        for symbol in config["data"]["symbols"]:
            source_paths = [
                file_path(raw, "incremental_book_L2", date, symbol),
                file_path(raw, "quotes", date, symbol),
                file_path(raw, "trades", date, symbol),
            ]
            if not all(path.exists() for path in source_paths):
                if args.available_only:
                    continue
                missing = [str(path) for path in source_paths if not path.exists()]
                raise FileNotFoundError("AEGIS_W9_1_SOURCE_MISSING:" + ",".join(missing))
            destination = output / f"episodes_{date}_{symbol}.parquet"
            audit_path = output / f"audit_{date}_{symbol}.json"
            if destination.exists() and audit_path.exists() and not args.force:
                continue
            tasks.append((symbol, date, source_paths, destination, audit_path))
    if tasks:
        with ProcessPoolExecutor(max_workers=max(1, args.workers)) as executor:
            futures = [executor.submit(process_part, task) for task in tasks]
            for future in as_completed(futures):
                print(json.dumps(future.result(), sort_keys=True), flush=True)
    audits = []
    parts = []
    for audit_path in sorted(output.glob("audit_*.json")):
        destination = output / audit_path.name.replace("audit_", "episodes_").replace(".json", ".parquet")
        if destination.exists():
            audits.append(json.loads(audit_path.read_text()))
            parts.append(pd.read_parquet(destination))
    if not parts:
        raise RuntimeError("AEGIS_W9_1_NO_COMPLETE_SOURCE_PARTS")
    combined = pd.concat(parts, ignore_index=True)
    combined.to_parquet(output / "w9_1_episodes.parquet", index=False)
    manifest = {
        "schema_version": "aegis-general-orderbook-direction-w9-1-dataset-v1",
        "rows": len(combined),
        "symbols": sorted(combined["symbol"].unique().tolist()),
        "dates": sorted(combined["date"].unique().tolist()),
        "all_partitions_pass": all(item["passes"] for item in audits),
        "available_only": args.available_only,
        "expected_parts": len(months) * len(config["data"]["symbols"]),
        "completed_parts": len(audits),
        "parts": audits,
    }
    (output / "w9_1_dataset_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
