#!/usr/bin/env python3
"""Build sealed W2 TRAIN/VALIDATION position episodes and decision rows."""

from __future__ import annotations

import argparse
import glob
import json
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from aegis.config import CANONICAL_SYMBOLS
from aegis.research.momentum_exhaustion_w2 import (
    build_episode_tables,
    select_nonoverlapping_candidates,
)
from aegis.research.volume_wave_w1 import build_causal_feature_frame
from aegis.utils import sha256_file


def _minutes(path: Path, start_ms: int, end_ms: int) -> pd.DataFrame:
    columns = [
        "open_time", "open", "high", "low", "close", "quote_volume",
        "trade_count", "taker_buy_quote", "symbol",
    ]
    frame = pd.read_parquet(path, columns=columns)
    frame = frame.loc[
        frame["open_time"].ge(start_ms) & frame["open_time"].lt(end_ms)
    ].copy()
    frame.rename(columns={
        "open_time": "open_time_ms",
        "trade_count": "agg_trade_count",
    }, inplace=True)
    frame["taker_sell_quote"] = (
        frame["quote_volume"] - frame["taker_buy_quote"]
    ).clip(lower=0.0)
    return frame[[
        "symbol", "open_time_ms", "open", "high", "low", "close",
        "quote_volume", "taker_buy_quote", "taker_sell_quote",
        "agg_trade_count",
    ]]


def _build_symbol(
    symbol: str,
    *,
    cache_root: str,
    output_root: str,
    candidates_path: str,
    config_path: str,
    w1_config_path: str,
    start_ms: int,
    end_ms: int,
) -> dict[str, Any]:
    cache = Path(cache_root)
    output = Path(output_root)
    config = yaml.safe_load(Path(config_path).read_text())
    w1_config = yaml.safe_load(Path(w1_config_path).read_text())
    all_candidates = pd.read_parquet(candidates_path)
    selected = select_nonoverlapping_candidates(all_candidates, config)
    minutes = _minutes(cache / f"{symbol}.parquet", start_ms, end_ms)
    btc = _minutes(cache / "BTCUSDT.parquet", start_ms, end_ms)
    features = build_causal_feature_frame(minutes, btc, w1_config)
    episodes, decisions = build_episode_tables(symbol, features, selected, config)
    if episodes.empty or decisions.empty:
        raise RuntimeError(f"AEGIS_W2_SYMBOL_DATASET_EMPTY:{symbol}")
    episode_path = output / f"{symbol}_episodes.parquet"
    decision_path = output / f"{symbol}_decisions.parquet"
    episodes.to_parquet(episode_path, index=False, compression="zstd")
    decisions.to_parquet(decision_path, index=False, compression="zstd")
    os.chmod(episode_path, 0o600)
    os.chmod(decision_path, 0o600)
    return {
        "symbol": symbol,
        "episodes": int(len(episodes)),
        "decisions": int(len(decisions)),
        "side_episodes": {
            str(key): int(value)
            for key, value in episodes["side"].value_counts().items()
        },
        "partition_episodes": {
            str(key): int(value)
            for key, value in episodes["partition"].value_counts().items()
        },
        "episode_sha256": sha256_file(episode_path),
        "decision_sha256": sha256_file(decision_path),
        "source_sha256": sha256_file(cache / f"{symbol}.parquet"),
    }


def _actual_inventory(pattern: str) -> dict[str, Any]:
    status_by_id: dict[str, set[str]] = {}
    source_hashes = []
    for raw_path in sorted(glob.glob(pattern)):
        path = Path(raw_path)
        source_hashes.append({"path": str(path.resolve()), "sha256": sha256_file(path)})
        for line in path.read_text(errors="replace").splitlines():
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            trade_id = str(row.get("trade_id", ""))
            status = str(row.get("status", ""))
            if trade_id and status:
                status_by_id.setdefault(trade_id, set()).add(status)
    return {
        "outcome_source": "ACTUAL",
        "role": "AUDIT_ONLY_FINAL_HOLDOUT",
        "complete_episode_count": sum(
            {"OPEN", "CLOSED"}.issubset(statuses)
            for statuses in status_by_id.values()
        ),
        "outcomes_read_by_builder": False,
        "sources": source_hashes,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path,
        default=Path("config/experiments/aegis_momentum_exhaustion_w2.yaml"),
    )
    parser.add_argument(
        "--w1-feature-config", type=Path,
        default=Path("config/experiments/aegis_volume_wave_w1.yaml"),
    )
    parser.add_argument(
        "--cache-root", type=Path,
        default=Path("data/market_event_fast_track_m1a/full_run_01/cache"),
    )
    parser.add_argument(
        "--candidates", type=Path,
        default=Path("data/market_event_fast_track_m1a/full_run_01/independent_candidates.parquet"),
    )
    parser.add_argument(
        "--output-root", type=Path,
        default=Path("data/momentum_exhaustion_w2/dataset_train_validation_01"),
    )
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--symbols", nargs="+", default=list(CANONICAL_SYMBOLS))
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    resolve = lambda value: value if value.is_absolute() else root / value
    config_path = resolve(args.config)
    w1_config_path = resolve(args.w1_feature_config)
    cache_root = resolve(args.cache_root)
    candidates_path = resolve(args.candidates)
    output_root = resolve(args.output_root)
    config = yaml.safe_load(config_path.read_text())
    symbols = tuple(args.symbols)
    if any(symbol not in CANONICAL_SYMBOLS for symbol in symbols):
        raise ValueError("AEGIS_W2_SYMBOL_INVALID")
    output_root.mkdir(parents=True, exist_ok=True)
    os.chmod(output_root, 0o700)
    train_start = int(pd.Timestamp(config["partitions"]["train"][0]).timestamp() * 1_000)
    validation_end = int(pd.Timestamp(config["partitions"]["validation"][1]).timestamp() * 1_000)
    start_ms = train_start - 7 * 86_400_000
    end_ms = validation_end
    kwargs = {
        "cache_root": str(cache_root),
        "output_root": str(output_root),
        "candidates_path": str(candidates_path),
        "config_path": str(config_path),
        "w1_config_path": str(w1_config_path),
        "start_ms": start_ms,
        "end_ms": end_ms,
    }
    results = []
    with ProcessPoolExecutor(max_workers=max(1, args.workers)) as pool:
        futures = {
            pool.submit(_build_symbol, symbol, **kwargs): symbol
            for symbol in symbols
        }
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            print(json.dumps({"symbol": result["symbol"], "state": "BUILT"}), flush=True)
    all_candidates = pd.read_parquet(candidates_path)
    holdout = select_nonoverlapping_candidates(
        all_candidates, config, include_partitions=("FINAL_HOLDOUT",)
    )
    manifest = {
        "schema_version": "aegis-momentum-exhaustion-w2-manifest-v1",
        "config_sha256": sha256_file(config_path),
        "w1_feature_implementation_reuse_only": True,
        "w1_config_sha256": sha256_file(w1_config_path),
        "candidate_source_sha256": sha256_file(candidates_path),
        "outcome_source": "SIMULATED",
        "results": sorted(results, key=lambda row: row["symbol"]),
        "total_episodes": sum(row["episodes"] for row in results),
        "total_decisions": sum(row["decisions"] for row in results),
        "final_holdout_state": "SEALED",
        "final_holdout_identity_count": int(len(holdout)),
        "final_holdout_outcomes_read": False,
        "actual_inventory": _actual_inventory(
            str(root / "binance-futures-bot-ts/logs/aegis/turbo_trades_*.jsonl")
        ),
        "authenticated_requests": 0,
        "exchange_mutations": 0,
    }
    manifest_path = output_root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    os.chmod(manifest_path, 0o600)
    print(json.dumps({
        "manifest": str(manifest_path),
        "episodes": manifest["total_episodes"],
        "decisions": manifest["total_decisions"],
        "holdout": "SEALED",
    }))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
