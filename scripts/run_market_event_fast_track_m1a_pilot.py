#!/usr/bin/env python3
"""Run one checksum-backed M1A archive through TRAIN and validation."""

from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import asdict
from pathlib import Path

from aegis.research.market_event_fast_track_evaluation import (
    evaluate_candidate,
    summarize_by_pattern_side,
)
from aegis.research.market_event_fast_track_m1a import (
    CausalRegimeClassifier,
    extract_pattern_features,
    fit_pattern_thresholds_from_train,
    fit_regime_thresholds_from_train,
    read_agg_trade_archive,
    read_kline_archive,
    resample_closed_bars,
    detect_micro_patterns,
)
from aegis.utils import sha256_file


def _safe(value):
    if isinstance(value, float) and not math.isfinite(value):
        return "INF" if value > 0 else "-INF"
    if isinstance(value, dict):
        return {str(key): _safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_safe(item) for item in value]
    if hasattr(value, "value"):
        return value.value
    return value


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--symbol", required=True)
    parser.add_argument("--month", required=True)
    parser.add_argument("--horizon-minutes", type=int, default=60)
    parser.add_argument(
        "--archive-root", type=Path, default=Path("data/market_event_fast_track_m1a/raw")
    )
    parser.add_argument(
        "--output", type=Path,
        default=Path("data/market_event_fast_track_m1a/pilot/pilot_report.json"),
    )
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    archive_root = args.archive_root if args.archive_root.is_absolute() else root / args.archive_root
    output = args.output if args.output.is_absolute() else root / args.output
    paths = {
        "futures_klines": archive_root / "futures/um/monthly/klines" / args.symbol / "1m" / f"{args.symbol}-1m-{args.month}.zip",
        "futures_aggTrades": archive_root / "futures/um/monthly/aggTrades" / args.symbol / f"{args.symbol}-aggTrades-{args.month}.zip",
        "spot_klines": archive_root / "spot/monthly/klines" / args.symbol / "1m" / f"{args.symbol}-1m-{args.month}.zip",
    }
    if any(not path.is_file() for path in paths.values()):
        raise SystemExit("AEGIS_M1A_PILOT_ARCHIVE_MISSING")
    futures_rows = read_kline_archive(paths["futures_klines"], args.symbol)
    spot_by_time = {row.open_time_ms: row for row in read_kline_archive(paths["spot_klines"], args.symbol)}
    flow_by_time = {row.open_time_ms: row for row in read_agg_trade_archive(paths["futures_aggTrades"], args.symbol)}
    common = tuple(
        row for row in futures_rows
        if row.open_time_ms in spot_by_time and row.open_time_ms in flow_by_time
    )
    if len(common) < 10_000:
        raise SystemExit("AEGIS_M1A_PILOT_COMMON_COVERAGE_INSUFFICIENT")
    split = int(len(common) * 0.60)
    train_features = []
    for index in range(240, split):
        timestamp = common[index].open_time_ms
        train_features.append(
            extract_pattern_features(
                futures=common[index - 240:index + 1],
                spot=tuple(spot_by_time[row.open_time_ms] for row in common[index - 240:index + 1]),
                flow=tuple(flow_by_time[row.open_time_ms] for row in common[index - 240:index + 1]),
                funding_rate=None,
            )
        )
    pattern_thresholds = fit_pattern_thresholds_from_train(train_features)

    hourly = resample_closed_bars(common, 60)
    four_hourly = resample_closed_bars(common, 240)
    train_cutoff = common[split - 1].close_time_ms
    regime_samples = []
    for index in range(24, len(hourly)):
        if hourly[index].close_time_ms > train_cutoff:
            break
        available_four = tuple(row for row in four_hourly if row.close_time_ms <= hourly[index].close_time_ms)
        if len(available_four) >= 7:
            regime_samples.append((hourly[: index + 1], available_four))
    regime_thresholds = fit_regime_thresholds_from_train(regime_samples)
    classifier = CausalRegimeClassifier(regime_thresholds, minimum_state_bars=3)
    regimes = {}
    for index in range(24, len(hourly)):
        available_four = tuple(row for row in four_hourly if row.close_time_ms <= hourly[index].close_time_ms)
        if len(available_four) < 7:
            continue
        observation = classifier.observe(hourly[: index + 1], available_four)
        regimes[observation.timestamp_ms] = observation
    regime_times = sorted(regimes)

    evaluated = []
    raw_candidates = 0
    omitted_missing_regime = 0
    last_event: dict[tuple[str, str], int] = {}
    for index in range(split, len(common) - args.horizon_minutes):
        row = common[index]
        eligible_regimes = [timestamp for timestamp in regime_times if timestamp <= row.close_time_ms]
        if not eligible_regimes:
            omitted_missing_regime += 1
            continue
        regime = regimes[eligible_regimes[-1]]
        history = common[index - 240:index + 1]
        candidates = detect_micro_patterns(
            symbol=args.symbol,
            futures=history,
            spot=tuple(spot_by_time[item.open_time_ms] for item in history),
            flow=tuple(flow_by_time[item.open_time_ms] for item in history),
            regime=regime,
            thresholds=pattern_thresholds,
            funding_rate=None,
        )
        raw_candidates += len(candidates)
        for candidate in candidates:
            key = (candidate.pattern.value, candidate.side.value)
            previous = last_event.get(key)
            if previous is not None and candidate.timestamp_ms - previous <= 60 * 60_000:
                continue
            last_event[key] = candidate.timestamp_ms
            evaluated.append(
                evaluate_candidate(
                    candidate,
                    common[index + 1:index + 1 + args.horizon_minutes],
                    horizon_minutes=args.horizon_minutes,
                )
            )
    metrics = summarize_by_pattern_side(evaluated, bootstrap_repetitions=500) if evaluated else {}
    report = {
        "schema_version": "aegis-m1a-archive-pilot-report-v1",
        "symbol": args.symbol,
        "month": args.month,
        "archives": {name: {"path": str(path), "sha256": sha256_file(path)} for name, path in paths.items()},
        "common_minutes": len(common),
        "train_minutes": split,
        "validation_minutes": len(common) - split,
        "train_feature_snapshots": len(train_features),
        "regime_train_samples": len(regime_samples),
        "pattern_thresholds": asdict(pattern_thresholds),
        "regime_thresholds": asdict(regime_thresholds),
        "raw_candidates": raw_candidates,
        "independent_evaluated_candidates": len(evaluated),
        "omitted_missing_regime": omitted_missing_regime,
        "metrics": {name: asdict(value) for name, value in metrics.items()},
        "promotion_authority": False,
        "authenticated_requests": 0,
        "exchange_mutations": 0,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(_safe(report), indent=2, sort_keys=True) + "\n")
    os.chmod(output, 0o600)
    print(json.dumps({"output": str(output), "candidates": len(evaluated), "groups": len(metrics)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
