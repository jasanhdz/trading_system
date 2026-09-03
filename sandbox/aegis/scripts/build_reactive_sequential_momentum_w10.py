#!/usr/bin/env python3
"""Build resumable W10 sequential state datasets from validated W9.1 inputs."""

from __future__ import annotations

import argparse
import hashlib
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
import yaml

from aegis.research.reactive_sequential_momentum_w10 import build_sequential_states


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def raw_path(raw: Path, kind: str, date: str, symbol: str) -> Path:
    return raw / f"binance-futures_{kind}_{date}_{symbol}.csv.gz"


def process_part(task: tuple[Path, str, str, list[Path], Path, Path]) -> dict:
    root, symbol, date, sources, output_path, audit_path = task
    frame, audit = build_sequential_states(
        root=root,
        symbol=symbol,
        date=date,
        l2_path=sources[0],
        quotes_path=sources[1],
        trades_path=sources[2],
    )
    audit["sources"] = {
        path.name: {"bytes": path.stat().st_size, "sha256": sha256(path)}
        for path in sources
    }
    frame.to_parquet(output_path, index=False)
    audit_path.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")
    return audit


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    root = args.root.resolve()
    config = yaml.safe_load((root / "config/experiments/aegis_reactive_sequential_momentum_w10.yaml").read_text())
    raw = root / config["data"]["source"]
    output = root / "data/reactive_sequential_momentum_w10/states"
    output.mkdir(parents=True, exist_ok=True)
    months = config["partitions"]["train_months"] + config["partitions"]["validation_months"]
    tasks = []
    for month in months:
        date = f"{month}-01"
        for symbol in config["data"]["symbols"]:
            sources = [raw_path(raw, kind, date, symbol) for kind in ("incremental_book_L2", "quotes", "trades")]
            if not all(path.exists() for path in sources):
                raise FileNotFoundError("AEGIS_W10_SOURCE_MISSING:" + ",".join(str(path) for path in sources if not path.exists()))
            destination = output / f"states_{date}_{symbol}.parquet"
            audit_path = output / f"audit_{date}_{symbol}.json"
            if destination.exists() and audit_path.exists() and not args.force:
                continue
            tasks.append((root, symbol, date, sources, destination, audit_path))
    if tasks:
        with ProcessPoolExecutor(max_workers=max(1, args.workers)) as executor:
            futures = [executor.submit(process_part, task) for task in tasks]
            for future in as_completed(futures):
                print(json.dumps(future.result(), sort_keys=True), flush=True)
    audits = [json.loads(path.read_text()) for path in sorted(output.glob("audit_*.json"))]
    parts = [pd.read_parquet(path) for path in sorted(output.glob("states_*.parquet"))]
    if not parts:
        raise RuntimeError("AEGIS_W10_NO_DATASET_PARTS")
    combined = pd.concat(parts, ignore_index=True)
    combined_path = output / "w10_states.parquet"
    combined.to_parquet(combined_path, index=False)
    manifest = {
        "schema_version": "aegis-reactive-sequential-momentum-w10-dataset-v1",
        "rows": len(combined),
        "episodes": int(combined["momentum_episode_id"].nunique()),
        "symbols": sorted(combined["symbol"].unique()),
        "dates": sorted(combined["date"].unique()),
        "expected_parts": len(months) * len(config["data"]["symbols"]),
        "completed_parts": len(parts),
        "all_partitions_pass": len(parts) == len(months) * len(config["data"]["symbols"]) and all(item["passes"] for item in audits),
        "w7_covered_episodes": int(combined.groupby("momentum_episode_id")["w7_opportunity_probability"].first().notna().sum()),
        "dataset_sha256": sha256(combined_path),
        "parts": audits,
    }
    (output / "w10_dataset_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps({key: value for key, value in manifest.items() if key != "parts"}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
