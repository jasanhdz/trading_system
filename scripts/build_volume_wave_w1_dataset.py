#!/usr/bin/env python3
"""Build the preregistered W1 volume-wave dataset from verified C2A sources."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import multiprocessing
import os
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from aegis.config import CANONICAL_SYMBOLS
from aegis.research.volume_wave_w1 import (
    SOURCE_COLUMNS,
    build_causal_feature_frame,
    build_wave_events,
    collapse_event_cooldown,
    deterministic_matched_controls,
)
from aegis.utils import sha256_file


def _read_minutes(path: Path) -> pd.DataFrame:
    frame = pd.read_parquet(
        path, columns=[*SOURCE_COLUMNS, "side"], filters=[("side", "==", "LONG")]
    ).drop(columns="side")
    if frame.empty or frame["open_time_ms"].duplicated().any():
        raise RuntimeError(f"AEGIS_W1_SOURCE_INVALID:{path.name}")
    return frame


def _build_symbol(
    symbol: str,
    source_root: Path,
    output_root: Path,
    config: dict[str, Any],
) -> dict[str, Any]:
    minutes = _read_minutes(source_root / f"{symbol}.parquet")
    btc = _read_minutes(source_root / "BTCUSDT.parquet")
    features = build_causal_feature_frame(minutes, btc, config)
    minimum_volume = float(config["candidate_population"]["minimum_volume_ratio_20"])
    cooldown = int(config["candidate_population"]["cooldown_bars_by_symbol_side"])
    wave = collapse_event_cooldown(
        build_wave_events(features, config), cooldown
    )
    broad_controls = build_wave_events(
        features, config, minimum_volume_ratio=0.0,
    )
    non_wave = collapse_event_cooldown(
        broad_controls.loc[
            broad_controls["volume_ratio_20"].lt(minimum_volume)
        ].copy(), cooldown
    )
    controls = deterministic_matched_controls(
        pd.concat([wave, non_wave], ignore_index=True), wave,
        minimum_volume_ratio=minimum_volume,
    )
    wave["sample_source"] = "WAVE_CANDIDATE"
    combined = pd.concat([wave, controls], ignore_index=True)
    combined.sort_values(
        ["event_timestamp_ms", "sample_source", "side", "entry_variant"],
        inplace=True,
    )
    path = output_root / f"{symbol}.parquet"
    combined.to_parquet(path, index=False, compression="zstd")
    counts = {
        f"{source}:{side}:{variant}": int(len(rows))
        for (source, side, variant), rows in combined.groupby(
            ["sample_source", "side", "entry_variant"], sort=True
        )
    }
    return {
        "symbol": symbol,
        "rows": int(len(combined)),
        "wave_rows": int(combined["sample_source"].eq("WAVE_CANDIDATE").sum()),
        "control_rows": int(combined["sample_source"].eq("MATCHED_PRICE_ONLY_CONTROL").sum()),
        "counts": counts,
        "dataset_sha256": sha256_file(path),
        "source_sha256": sha256_file(source_root / f"{symbol}.parquet"),
        "minimum_event_timestamp_ms": int(combined["event_timestamp_ms"].min()),
        "maximum_event_timestamp_ms": int(combined["event_timestamp_ms"].max()),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-root", type=Path,
        default=Path("data/event_path_quality_c2a/dataset_2025_08_2026_07"),
    )
    parser.add_argument(
        "--output-root", type=Path,
        default=Path("data/volume_wave_w1/dataset_2025_08_2026_07"),
    )
    parser.add_argument(
        "--config", type=Path,
        default=Path("config/experiments/aegis_volume_wave_w1.yaml"),
    )
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--symbols", nargs="*", default=list(CANONICAL_SYMBOLS))
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    source_root = args.source_root if args.source_root.is_absolute() else root / args.source_root
    output_root = args.output_root if args.output_root.is_absolute() else root / args.output_root
    config_path = args.config if args.config.is_absolute() else root / args.config
    symbols = tuple(args.symbols)
    if (
        args.workers < 1 or not symbols
        or not set(symbols).issubset(CANONICAL_SYMBOLS)
        or len(set(symbols)) != len(symbols)
        or not (source_root / "manifest.json").is_file()
    ):
        raise RuntimeError("AEGIS_W1_BUILD_INPUT_INVALID")
    config = yaml.safe_load(config_path.read_text())
    source_manifest = json.loads((source_root / "manifest.json").read_text())
    if source_manifest.get("schema_version") != "aegis-event-path-quality-c2a-manifest-v1":
        raise RuntimeError("AEGIS_W1_SOURCE_MANIFEST_INVALID")
    output_root.mkdir(parents=True, exist_ok=True)
    os.chmod(output_root, 0o700)
    context = multiprocessing.get_context("spawn")
    results = []
    with concurrent.futures.ProcessPoolExecutor(
        max_workers=args.workers, mp_context=context
    ) as executor:
        futures = {
            executor.submit(_build_symbol, symbol, source_root, output_root, config): symbol
            for symbol in symbols
        }
        for future in concurrent.futures.as_completed(futures):
            result = future.result()
            results.append(result)
            print(json.dumps({
                "symbol": result["symbol"], "rows": result["rows"],
                "state": "BUILT",
            }), flush=True)
    results.sort(key=lambda item: CANONICAL_SYMBOLS.index(item["symbol"]))
    manifest = {
        "schema_version": "aegis-volume-wave-w1-manifest-v1",
        "config_sha256": sha256_file(config_path),
        "source_manifest_sha256": sha256_file(source_root / "manifest.json"),
        "source_projection": [*SOURCE_COLUMNS, "side=LONG_identity_deduplication"],
        "symbols": list(symbols),
        "results": results,
        "total_rows": sum(item["rows"] for item in results),
        "wave_rows": sum(item["wave_rows"] for item in results),
        "control_rows": sum(item["control_rows"] for item in results),
        "final_holdout_state": "SEALED",
        "W1_DATASET_READY": True,
        "W1_READY_FOR_SHADOW": False,
        "W1_READY_FOR_LIVE": False,
        "authenticated_requests": 0,
        "exchange_mutations": 0,
    }
    manifest_path = output_root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    os.chmod(manifest_path, 0o600)
    print(json.dumps({
        "manifest": str(manifest_path), "total_rows": manifest["total_rows"],
        "holdout": manifest["final_holdout_state"],
    }))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
