#!/usr/bin/env python3
"""Build the sealed-holdout W11 causal path and confirmation dataset."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd
import yaml

from aegis.research.entry_safety_gate_w11 import build_candidate_dataset, path_order_summary
from aegis.research.live_entry_quality_audit import read_trade_pairs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("config/experiments/aegis_entry_safety_gate_w11.yaml"))
    parser.add_argument("--out-dir", type=Path, default=Path("data/entry_safety_gate_w11/run_01"))
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    entries = pd.read_csv(config["sources"]["entry_csv"])
    pairs, log_audit = read_trade_pairs(Path(config["sources"]["logs_dir"]), {
        "trade_glob": "turbo_trades_*.jsonl",
        "strategy": "AEGIS_TURBO",
        "mode": "AEGIS_TURBO_MICRO_LIVE",
    })
    entry_prices = {}
    for pair in pairs:
        opened = pair.get("open") or {}
        closed = pair.get("close") or {}
        value = opened.get("entry_price", closed.get("entry_price"))
        if value is not None:
            entry_prices[str(closed.get("trade_id"))] = float(value)
    candle_dir = Path(config["sources"]["candle_dir"])
    candles = {
        symbol: pd.read_parquet(candle_dir / f"{symbol}_1m.parquet")
        for symbol in sorted(entries["symbol"].unique())
    }
    dataset, audit = build_candidate_dataset(entries, candles, entry_prices, config, include_holdout=False)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    dataset_path = args.out_dir / "w11_candidates_train_validation.parquet"
    dataset.to_parquet(dataset_path, index=False)
    digest = hashlib.sha256(dataset_path.read_bytes()).hexdigest()
    manifest = {
        "schema_version": "aegis-entry-safety-gate-w11-dataset-v1",
        "dataset_path": str(dataset_path),
        "dataset_sha256": digest,
        "audit": audit,
        "log_audit": log_audit,
        "entry_prices_found": len(entry_prices),
        "path_order_summary": path_order_summary(dataset),
        "holdout_status": "SEALED_NOT_OPENED",
        "restrictions_verified": config["restrictions"],
    }
    (args.out_dir / "w11_dataset_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
