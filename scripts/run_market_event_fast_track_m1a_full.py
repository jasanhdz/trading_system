#!/usr/bin/env python3
"""Run frozen M1A patterns across all symbols and retrospective partitions."""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import subprocess
from dataclasses import asdict
from pathlib import Path

import pandas as pd

from aegis.config import CANONICAL_SYMBOLS
from aegis.research.market_event_fast_track_batch import (
    DISCOVERY_END_MS,
    attach_regime,
    bootstrap_expectancy,
    build_causal_features,
    build_hourly_regime,
    classify_hourly_regime,
    collapse_events,
    detect_candidates,
    evaluate_events,
    fit_global_pattern_thresholds,
    fit_global_regime_thresholds,
    load_symbol_frame,
    matched_random_control,
    summaries,
)
from aegis.utils import sha256_file


def _safe(value):
    if isinstance(value, float) and not math.isfinite(value):
        return "INF" if value > 0 else "-INF"
    if isinstance(value, dict):
        return {str(key): _safe(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_safe(item) for item in value]
    return value


def _commit(root: Path) -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--archive-root",
        type=Path,
        default=Path("data/market_event_fast_track_m1a/raw"),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("data/market_event_fast_track_m1a/full_run_01"),
    )
    parser.add_argument("--horizon-minutes", type=int, default=60)
    parser.add_argument("--symbols", nargs="+", default=list(CANONICAL_SYMBOLS))
    parser.add_argument("--rebuild-cache", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    archive_root = (
        args.archive_root
        if args.archive_root.is_absolute()
        else root / args.archive_root
    )
    output_root = (
        args.output_root if args.output_root.is_absolute() else root / args.output_root
    )
    cache_root = output_root / "cache"
    cache_root.mkdir(parents=True, exist_ok=True)
    os.chmod(output_root, 0o700)
    os.chmod(cache_root, 0o700)

    config = root / "config/experiments/aegis_market_event_fast_track_m1a.yaml"
    manifest = root / "data/market_event_fast_track_m1a/archive_manifest.jsonl"
    module = root / "src/aegis/research/market_event_fast_track_batch.py"
    cache_identity = {
        "config_sha256": sha256_file(config),
        "archive_manifest_sha256": sha256_file(manifest),
        "batch_module_sha256": sha256_file(module),
    }
    cache_identity_path = cache_root / "identity.json"
    if cache_identity_path.exists() and not args.rebuild_cache:
        existing = json.loads(cache_identity_path.read_text())
        if existing != cache_identity:
            raise RuntimeError("AEGIS_M1A_CACHE_IDENTITY_MISMATCH_USE_REBUILD_CACHE")

    discovery_frames = []
    hourly_frames = {}
    source_counts = {}
    for symbol in args.symbols:
        cache = cache_root / f"{symbol}.parquet"
        if cache.exists() and not args.rebuild_cache:
            frame = pd.read_parquet(cache)
        else:
            frame = build_causal_features(load_symbol_frame(archive_root, symbol))
            frame.to_parquet(cache, compression="zstd", index=False)
            os.chmod(cache, 0o600)
        discovery_frames.append(frame.loc[frame["open_time"] < DISCOVERY_END_MS].copy())
        hourly_frames[symbol] = build_hourly_regime(frame)
        source_counts[symbol] = len(frame)
        print(f"cache_ready symbol={symbol} rows={len(frame)}", flush=True)
        del frame
        gc.collect()

    cache_identity_path.write_text(
        json.dumps(cache_identity, indent=2, sort_keys=True) + "\n"
    )
    os.chmod(cache_identity_path, 0o600)

    pattern_thresholds = fit_global_pattern_thresholds(tuple(discovery_frames))
    regime_thresholds = fit_global_regime_thresholds(tuple(hourly_frames.values()))
    print("thresholds_fitted", flush=True)
    del discovery_frames
    gc.collect()
    candidates = []
    for symbol in args.symbols:
        frame = pd.read_parquet(cache_root / f"{symbol}.parquet")
        classified = classify_hourly_regime(hourly_frames[symbol], regime_thresholds)
        attached = attach_regime(frame, classified)
        detected = detect_candidates(attached, pattern_thresholds)
        if not detected.empty:
            candidates.append(detected)
        print(f"candidates_ready symbol={symbol} rows={len(detected)}", flush=True)
        del frame, attached
        gc.collect()
    raw_candidates = pd.concat(candidates, ignore_index=True)
    independent = collapse_events(raw_candidates)
    print(
        f"events_collapsed raw={len(raw_candidates)} independent={len(independent)}",
        flush=True,
    )
    evaluated_parts = []
    control_parts = []
    for symbol in args.symbols:
        symbol_events = independent.loc[independent["symbol"].eq(symbol)]
        if symbol_events.empty:
            continue
        frame = pd.read_parquet(cache_root / f"{symbol}.parquet")
        classified = classify_hourly_regime(hourly_frames[symbol], regime_thresholds)
        attached = attach_regime(frame, classified)
        evaluated_symbol = evaluate_events(
            symbol_events, {symbol: attached}, args.horizon_minutes
        )
        evaluated_parts.append(evaluated_symbol)
        control_candidates = matched_random_control(
            evaluated_symbol, {symbol: attached}, args.horizon_minutes
        )
        control_parts.append(
            evaluate_events(
                control_candidates, {symbol: attached}, args.horizon_minutes
            )
        )
        print(
            f"events_evaluated symbol={symbol} real={len(evaluated_symbol)} "
            f"control={len(control_parts[-1])}",
            flush=True,
        )
        del frame, attached, evaluated_symbol, control_candidates
        gc.collect()
    evaluated = pd.concat(evaluated_parts, ignore_index=True)
    controls = pd.concat(control_parts, ignore_index=True)

    raw_path = output_root / "raw_candidates.parquet"
    independent_path = output_root / "independent_candidates.parquet"
    evaluated_path = output_root / "evaluated_events.parquet"
    control_path = output_root / "matched_random_events.parquet"
    raw_candidates.to_parquet(raw_path, compression="zstd", index=False)
    independent.to_parquet(independent_path, compression="zstd", index=False)
    evaluated.to_parquet(evaluated_path, compression="zstd", index=False)
    controls.to_parquet(control_path, compression="zstd", index=False)
    for path in (raw_path, independent_path, evaluated_path, control_path):
        os.chmod(path, 0o600)

    real_summaries = summaries(evaluated)
    control_summaries = summaries(controls)
    comparisons = {}
    for key, real in real_summaries.items():
        control = control_summaries.get(key)
        partition, pattern, side = key.split(":")
        group = evaluated.loc[
            evaluated["partition"].eq(partition)
            & evaluated["pattern"].eq(pattern)
            & evaluated["side"].eq(side)
        ]
        interval = bootstrap_expectancy(group) if len(group) >= 2 else None
        stress_expectancy = float(real["expectancy"]) - 0.0006
        thirds = []
        if not group.empty:
            ordered = group.sort_values("timestamp_ms")
            for third in range(3):
                sample = ordered.iloc[
                    third * len(ordered) // 3 : (third + 1) * len(ordered) // 3
                ]
                thirds.append(
                    float(sample["net_return_fraction"].mean())
                    if not sample.empty
                    else None
                )
        gate = {
            "minimum_events": len(group) >= 100,
            "positive_expectancy": float(real["expectancy"]) > 0.0,
            "expectancy_ci_lower_positive": bool(
                interval and interval["expectancy_lower_95"] > 0.0
            ),
            "profit_factor_ci_lower_above_one": bool(
                interval and interval["profit_factor_lower_95"] > 1.0
            ),
            "positive_temporal_thirds": sum(
                value is not None and value > 0.0 for value in thirds
            )
            >= 2,
            "maximum_symbol_share": float(real["symbol_share_maximum"]) <= 0.25,
            "outperforms_matched_random": bool(
                control and float(real["expectancy"]) > float(control["expectancy"])
            ),
            "positive_first_stress_cost": stress_expectancy > 0.0,
        }
        comparisons[key] = {
            "real": real,
            "matched_random": control,
            "bootstrap_by_utc_day": interval,
            "temporal_thirds_expectancy": thirds,
            "first_stress_expectancy": stress_expectancy,
            "expectancy_increment": (
                float(real["expectancy"]) - float(control["expectancy"])
                if control is not None
                else None
            ),
            "gate": gate,
            "gate_pass": all(gate.values()),
        }
    validation_passes = [
        key
        for key, value in comparisons.items()
        if key.startswith("VALIDATION:") and value["gate_pass"]
    ]
    report = {
        "schema_version": "aegis-m1a-full-retrospective-report-v1",
        "experiment_id": "aegis-market-event-fast-track-m1a-01",
        "code_commit": _commit(root),
        "config_sha256": sha256_file(config),
        "symbols": list(args.symbols),
        "source_minutes": source_counts,
        "horizon_minutes": args.horizon_minutes,
        "pattern_thresholds": asdict(pattern_thresholds),
        "regime_thresholds": asdict(regime_thresholds),
        "raw_candidate_count": len(raw_candidates),
        "independent_candidate_count": len(independent),
        "evaluated_event_count": len(evaluated),
        "matched_random_event_count": len(controls),
        "comparisons": comparisons,
        "validation_gate_passes": validation_passes,
        "artifacts": {
            path.name: {"path": str(path), "sha256": sha256_file(path)}
            for path in (raw_path, independent_path, evaluated_path, control_path)
        },
        "retrospective_pseudo_holdout_has_promotion_authority": False,
        "M1A_READY_FOR_FORWARD_SHADOW": bool(validation_passes),
        "M1A_READY_FOR_LIVE": False,
        "authenticated_requests": 0,
        "exchange_mutations": 0,
        "runtime_changes": "NONE",
    }
    report_path = output_root / "full_report.json"
    report_path.write_text(json.dumps(_safe(report), indent=2, sort_keys=True) + "\n")
    os.chmod(report_path, 0o600)
    print(
        json.dumps(
            {
                "report": str(report_path),
                "raw_candidates": len(raw_candidates),
                "independent_candidates": len(independent),
                "evaluated": len(evaluated),
                "groups": len(comparisons),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
