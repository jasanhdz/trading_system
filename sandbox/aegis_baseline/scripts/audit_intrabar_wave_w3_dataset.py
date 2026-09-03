#!/usr/bin/env python3
"""Audit W3 dataset causality, identities, coverage, and sealed-holdout state."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from aegis.research.intrabar_wave_w3 import W3_FEATURE_COLUMNS, stable_wave_episode_id
from aegis.utils import sha256_file


def _partition_bounds(config: dict[str, Any], name: str) -> tuple[int, int]:
    purge = pd.Timedelta(minutes=int(config["partitions"]["purge_minutes"]))
    start, end = (pd.Timestamp(value) for value in config["partitions"][name.lower()])
    if name == "VALIDATION":
        start += purge
    end -= purge
    return int(start.timestamp() * 1000), int(end.timestamp() * 1000)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("config/experiments/aegis_intrabar_wave_w3.yaml"))
    parser.add_argument("--dataset-root", type=Path, default=Path("data/intrabar_wave_w3/dataset_train_validation_01"))
    parser.add_argument("--output", type=Path, default=Path("data/intrabar_wave_w3/dataset_audit_01.json"))
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    resolve = lambda value: value if value.is_absolute() else root / value
    config_path, dataset_root, output = map(resolve, (args.config, args.dataset_root, args.output))
    config = yaml.safe_load(config_path.read_text())
    manifest_path = dataset_root / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    failures: list[str] = []
    summaries = []
    for symbol in config["universe"]["symbols"]:
        entry = pd.read_parquet(dataset_root / f"{symbol}_entry.parquet")
        exits = pd.read_parquet(dataset_root / f"{symbol}_exit.parquet")
        episodes = pd.read_parquet(dataset_root / f"{symbol}_episodes.parquet")
        if entry.duplicated(["wave_episode_id", "offset_minutes"]).any():
            failures.append(f"{symbol}:DUPLICATE_ENTRY_DECISION")
        if exits.duplicated(["wave_episode_id", "episode_minute"]).any():
            failures.append(f"{symbol}:DUPLICATE_EXIT_DECISION")
        if episodes.wave_episode_id.duplicated().any():
            failures.append(f"{symbol}:DUPLICATE_EPISODE")
        if not (entry.entry_time_ms - entry.decision_time_ms).eq(60_000).all():
            failures.append(f"{symbol}:ENTRY_EXECUTION_DELAY_INVALID")
        if not np.isfinite(entry[list(W3_FEATURE_COLUMNS)].to_numpy(dtype=float)).all():
            failures.append(f"{symbol}:ENTRY_FEATURE_NONFINITE")
        if not np.isfinite(exits[list(W3_FEATURE_COLUMNS)].to_numpy(dtype=float)).all():
            failures.append(f"{symbol}:EXIT_FEATURE_NONFINITE")
        expected_ids = episodes.apply(
            lambda row: stable_wave_episode_id(row.symbol, row.side, int(row.impulse_close_time_ms)), axis=1
        )
        if not expected_ids.eq(episodes.wave_episode_id).all():
            failures.append(f"{symbol}:EPISODE_IDENTITY_INVALID")
        for partition in ("TRAIN", "VALIDATION"):
            lower, upper = _partition_bounds(config, partition)
            subset = episodes.loc[episodes.partition.eq(partition), "impulse_close_time_ms"]
            if not subset.between(lower, upper, inclusive="left").all():
                failures.append(f"{symbol}:{partition}_PURGE_INVALID")
        summaries.append({
            "symbol": symbol, "entry_rows": len(entry), "exit_rows": len(exits),
            "episodes": len(episodes), "gate_reached_episodes": int(episodes.gate_reached.sum()),
            "train_episodes": int(episodes.partition.eq("TRAIN").sum()),
            "validation_episodes": int(episodes.partition.eq("VALIDATION").sum()),
            "minimum_timestamp_ms": int(episodes.impulse_close_time_ms.min()),
            "maximum_timestamp_ms": int(episodes.impulse_close_time_ms.max()),
        })
    if manifest["final_holdout_state"] != "SEALED" or manifest["final_holdout_outcomes_read"]:
        failures.append("FINAL_HOLDOUT_CONTRACT_INVALID")
    result = {
        "schema_version": "aegis-intrabar-wave-w3-dataset-audit-v1",
        "status": "PASS" if not failures else "FAIL",
        "failures": failures,
        "config_sha256": sha256_file(config_path),
        "dataset_manifest_sha256": sha256_file(manifest_path),
        "symbols": summaries,
        "total_episodes": sum(item["episodes"] for item in summaries),
        "final_holdout_state": "SEALED",
        "final_holdout_outcomes_read": False,
        "aggregate_trade_ticks_used": False,
        "order_book_reconstructed": False,
        "authenticated_requests": 0,
        "exchange_mutations": 0,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    os.chmod(output, 0o600)
    print(json.dumps({"output": str(output), "status": result["status"]}))
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
