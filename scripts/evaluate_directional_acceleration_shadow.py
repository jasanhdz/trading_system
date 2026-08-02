"""Evaluate the preregistered acceleration observer on local 5m candles."""

from __future__ import annotations

import argparse
import json
import math
import os
import sqlite3
from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from aegis.config import CANONICAL_SYMBOLS
from aegis.domain import Candle
from aegis.features import DeterministicFeaturePipeline
from aegis.research.directional_acceleration_shadow import (
    AccelerationState,
    evaluate_directional_acceleration_shadow,
)
from aegis.research.entry_intelligence_shadow import (
    load_entry_intelligence_shadow_config,
)


def _db_symbol(symbol: str) -> str:
    return f"{symbol[:-4]}/USDT"


def _parse_timestamp(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _load_symbol(
    connection: sqlite3.Connection,
    symbol: str,
    start: datetime,
    end: datetime,
) -> dict[datetime, Candle]:
    rows = connection.execute(
        "SELECT timestamp, open, high, low, close, volume "
        "FROM ohlcv_data WHERE symbol=? AND timeframe='5m' "
        "AND timestamp>=? AND timestamp<=? ORDER BY timestamp",
        (
            _db_symbol(symbol),
            start.replace(tzinfo=None).isoformat(sep=" "),
            end.replace(tzinfo=None).isoformat(sep=" "),
        ),
    ).fetchall()
    result: dict[datetime, Candle] = {}
    for raw_timestamp, open_, high, low, close, volume in rows:
        timestamp = _parse_timestamp(str(raw_timestamp))
        result[timestamp] = Candle(
            open_time=timestamp,
            close_time=timestamp + timedelta(minutes=5),
            open=float(open_),
            high=float(high),
            low=float(low),
            close=float(close),
            volume=float(volume),
            is_closed=True,
            source="LOCAL_BINANCE_CANDLE_DB",
        )
    return result


def _ratio(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def evaluate(
    database: Path,
    config_path: Path,
    regime_config_path: Path,
    *,
    lookback_days: int,
    horizon_bars: int,
) -> dict[str, Any]:
    root = config_path.resolve().parents[1]
    config = load_entry_intelligence_shadow_config(
        config_path,
        repo_root=root,
        regime_config_path=regime_config_path,
    )
    connection = sqlite3.connect(f"file:{database.resolve()}?mode=ro", uri=True)
    try:
        latest_raw = connection.execute(
            "SELECT timestamp FROM ohlcv_data WHERE symbol=? AND timeframe='5m' "
            "ORDER BY timestamp DESC LIMIT 1",
            (_db_symbol("BTCUSDT"),),
        ).fetchone()
        if latest_raw is None:
            raise RuntimeError("AEGIS_ACCELERATION_REPLAY_NO_CANDLES")
        end = _parse_timestamp(str(latest_raw[0]))
        start = end - timedelta(days=lookback_days, minutes=5 * 96)
        series = {
            symbol: _load_symbol(connection, symbol, start, end)
            for symbol in CANONICAL_SYMBOLS
        }
    finally:
        connection.close()

    common_timestamps = sorted(
        set.intersection(*(set(rows) for rows in series.values()))
    )
    if len(common_timestamps) <= 96 + horizon_bars:
        raise RuntimeError("AEGIS_ACCELERATION_REPLAY_INSUFFICIENT_ALIGNED_CANDLES")
    candles = {
        symbol: [series[symbol][timestamp] for timestamp in common_timestamps]
        for symbol in CANONICAL_SYMBOLS
    }
    pipeline = DeterministicFeaturePipeline()
    counts: dict[str, Counter[str]] = defaultdict(Counter)
    sums: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    confusion: dict[str, Counter[str]] = defaultdict(Counter)
    evaluated = 0

    for index in range(96, len(common_timestamps) - horizon_bars):
        local = {
            symbol: pipeline._local_features(candles[symbol][index - 95 : index + 1])
            for symbol in CANONICAL_SYMBOLS
        }
        btc_return = local["BTCUSDT"]["ret_6"]
        for symbol in CANONICAL_SYMBOLS:
            features = local[symbol]
            features["btc_divergence_6"] = features["ret_6"] - btc_return
            observation = evaluate_directional_acceleration_shadow(
                features, config.acceleration_settings
            )
            future = candles[symbol][index + 1 : index + 1 + horizon_bars]
            entry = candles[symbol][index].close
            terminal_return = (future[-1].close - entry) / entry
            maximum_up = max(0.0, (max(row.high for row in future) - entry) / entry)
            maximum_down = max(0.0, (entry - min(row.low for row in future)) / entry)
            closes_above = sum(row.close > entry for row in future)
            closes_below = sum(row.close < entry for row in future)
            atr = max(float(features["atr_12"]), 1e-15)
            persistent_up = terminal_return >= atr * 2.0 and closes_above >= math.ceil(
                horizon_bars * 0.75
            )
            persistent_down = (
                terminal_return <= -atr * 2.0
                and closes_below >= math.ceil(horizon_bars * 0.75)
            )
            state = str(observation["state"])
            for group in ("ALL", symbol):
                counts[group][state] += 1
                counts[group]["observations"] += 1
                counts[group]["persistent_up"] += int(persistent_up)
                counts[group]["persistent_down"] += int(persistent_down)
                sums[group][f"return::{state}"] += terminal_return
                sums[group][f"up_excursion::{state}"] += maximum_up
                sums[group][f"down_excursion::{state}"] += maximum_down
                predicted_up = state in {
                    AccelerationState.UPWARD_PRESSURE.value,
                    AccelerationState.UPWARD_ACCELERATION.value,
                }
                predicted_down = state in {
                    AccelerationState.DOWNWARD_PRESSURE.value,
                    AccelerationState.DOWNWARD_ACCELERATION.value,
                }
                confusion[group]["up_tp"] += int(predicted_up and persistent_up)
                confusion[group]["up_fp"] += int(predicted_up and not persistent_up)
                confusion[group]["up_fn"] += int(not predicted_up and persistent_up)
                confusion[group]["down_tp"] += int(predicted_down and persistent_down)
                confusion[group]["down_fp"] += int(
                    predicted_down and not persistent_down
                )
                confusion[group]["down_fn"] += int(
                    not predicted_down and persistent_down
                )
            evaluated += 1

    summaries: dict[str, Any] = {}
    for group, group_counts in counts.items():
        states = {}
        for state in AccelerationState:
            count = group_counts[state.value]
            states[state.value] = {
                "count": count,
                "fraction": _ratio(count, group_counts["observations"]),
                "mean_terminal_return_fraction": (
                    sums[group][f"return::{state.value}"] / count if count else None
                ),
                "mean_maximum_up_excursion_fraction": (
                    sums[group][f"up_excursion::{state.value}"] / count
                    if count
                    else None
                ),
                "mean_maximum_down_excursion_fraction": (
                    sums[group][f"down_excursion::{state.value}"] / count
                    if count
                    else None
                ),
            }
        matrix = confusion[group]
        summaries[group] = {
            "observations": group_counts["observations"],
            "persistent_up": group_counts["persistent_up"],
            "persistent_down": group_counts["persistent_down"],
            "states": states,
            "upward_precision": _ratio(
                matrix["up_tp"], matrix["up_tp"] + matrix["up_fp"]
            ),
            "upward_recall": _ratio(matrix["up_tp"], matrix["up_tp"] + matrix["up_fn"]),
            "downward_precision": _ratio(
                matrix["down_tp"], matrix["down_tp"] + matrix["down_fp"]
            ),
            "downward_recall": _ratio(
                matrix["down_tp"], matrix["down_tp"] + matrix["down_fn"]
            ),
        }
    return {
        "schema_id": "aegis-directional-acceleration-shadow-replay-v1",
        "mode": "SHADOW",
        "database": str(database.resolve()),
        "database_open_mode": "READ_ONLY",
        "evidence_start": common_timestamps[96].isoformat().replace("+00:00", "Z"),
        "evidence_end": common_timestamps[-1 - horizon_bars]
        .isoformat()
        .replace("+00:00", "Z"),
        "lookback_days": lookback_days,
        "horizon_bars": horizon_bars,
        "aligned_candles": len(common_timestamps),
        "symbol_evaluations": evaluated,
        "feature_schema": pipeline.schema_version,
        "feature_count": len(pipeline.feature_names),
        "config_sha256": config.config_sha256,
        "thresholds_preregistered_before_replay": True,
        "threshold_tuning_from_this_replay": False,
        "summaries": summaries,
        "selection_effect": "NONE",
        "exchange_calls": 0,
        "exchange_mutations": 0,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--database", type=Path, default=Path("data/binance_candles.db")
    )
    parser.add_argument(
        "--config", type=Path, default=Path("config/entry_intelligence_shadow.yaml")
    )
    parser.add_argument(
        "--regime-config", type=Path, default=Path("config/entry_quality_v2.yaml")
    )
    parser.add_argument("--lookback-days", type=int, default=60)
    parser.add_argument("--horizon-bars", type=int, default=12)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/directional_acceleration_shadow/replay_v1.json"),
    )
    args = parser.parse_args()
    if args.lookback_days <= 0 or args.horizon_bars <= 0:
        parser.error("lookback and horizon must be positive")
    root = Path(__file__).resolve().parents[1]

    def resolve(value: Path) -> Path:
        return value if value.is_absolute() else root / value

    report = evaluate(
        resolve(args.database),
        resolve(args.config),
        resolve(args.regime_config),
        lookback_days=args.lookback_days,
        horizon_bars=args.horizon_bars,
    )
    output = resolve(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.chmod(temporary, 0o600)
    temporary.replace(output)
    print(
        json.dumps({"output": str(output), "evaluations": report["symbol_evaluations"]})
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
