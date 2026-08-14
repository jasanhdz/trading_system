#!/usr/bin/env python3
"""Build causal W3A/W3B minute-state datasets without reading W3 holdout outcomes."""

from __future__ import annotations

import argparse
import bisect
import concurrent.futures
import json
import multiprocessing
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from aegis.config import CANONICAL_SYMBOLS
from aegis.research.intrabar_wave_w3 import (
    W3_FEATURE_COLUMNS,
    barrier_return,
    directional_clv,
    future_giveback_before_new_extreme,
    safe_ratio,
    stable_wave_episode_id,
)
from aegis.utils import sha256_file


SOURCE_COLUMNS = (
    "open_time", "open", "high", "low", "close", "quote_volume",
    "trade_count", "taker_buy_quote", "symbol",
)


def _rsi(close: pd.Series, period: int) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0.0).ewm(alpha=1.0 / period, adjust=False).mean()
    loss = (-delta.clip(upper=0.0)).ewm(alpha=1.0 / period, adjust=False).mean()
    rs = gain / loss.replace(0.0, np.nan)
    return (100.0 - 100.0 / (1.0 + rs)).fillna(50.0)


def _read_minutes(path: Path, start_ms: int, end_ms: int) -> pd.DataFrame:
    frame = pd.read_parquet(
        path,
        columns=list(SOURCE_COLUMNS),
        filters=[("open_time", ">=", start_ms), ("open_time", "<", end_ms)],
    ).copy()
    frame.rename(columns={"open_time": "open_time_ms"}, inplace=True)
    frame.sort_values("open_time_ms", inplace=True)
    frame.reset_index(drop=True, inplace=True)
    if frame.empty or frame["open_time_ms"].duplicated().any():
        raise RuntimeError(f"AEGIS_W3_SOURCE_INVALID:{path.name}")
    delta = frame["open_time_ms"].diff().dropna()
    if not delta.eq(60_000).all():
        raise RuntimeError(f"AEGIS_W3_SOURCE_GAP:{path.name}")
    return frame


def _aggregate(frame: pd.DataFrame, minutes: int) -> pd.DataFrame:
    interval = minutes * 60_000
    work = frame.copy()
    work["bucket"] = work["open_time_ms"] // interval * interval
    grouped = work.groupby("bucket", sort=True)
    result = grouped.agg(
        open=("open", "first"), high=("high", "max"), low=("low", "min"),
        close=("close", "last"), quote_volume=("quote_volume", "sum"),
        taker_buy_quote=("taker_buy_quote", "sum"), trade_count=("trade_count", "sum"),
        minute_count=("open_time_ms", "size"),
    )
    result = result.loc[result["minute_count"].eq(minutes)].copy()
    result["close_time_ms"] = result.index.to_numpy(dtype=np.int64) + interval - 60_000
    previous = result["close"].shift(1)
    tr = pd.concat([
        result["high"] - result["low"],
        (result["high"] - previous).abs(),
        (result["low"] - previous).abs(),
    ], axis=1).max(axis=1)
    result["atr"] = tr.rolling(14, min_periods=14).mean()
    prior_volume = result["quote_volume"].shift(1)
    result["volume_ratio_20"] = result["quote_volume"] / prior_volume.rolling(20).median()
    log_volume = np.log(result["quote_volume"].clip(lower=1e-12))
    result["volume_zscore_20"] = (
        log_volume - log_volume.shift(1).rolling(20).mean()
    ) / log_volume.shift(1).rolling(20).std().replace(0.0, np.nan)
    result["body_ratio"] = (result["close"] - result["open"]).abs() / (
        result["high"] - result["low"]
    ).replace(0.0, np.nan)
    result["body_atr"] = (result["close"] - result["open"]).abs() / result["atr"]
    result["taker_imbalance"] = (
        2.0 * result["taker_buy_quote"] / result["quote_volume"].replace(0.0, np.nan) - 1.0
    )
    result["velocity"] = (result["close"] - previous) / result["atr"]
    result["ma7"] = result["close"].rolling(7).mean()
    result["ma25"] = result["close"].rolling(25).mean()
    result["trend"] = np.sign(result["ma7"] - result["ma25"])
    result["rsi"] = _rsi(result["close"], 12)
    return result.reset_index(drop=False)


def _prepare_minutes(frame: pd.DataFrame, five: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    previous = result["close"].shift(1)
    result["raw_return_1m"] = result["close"] / previous - 1.0
    result["raw_return_5m"] = result["close"] / result["close"].shift(5) - 1.0
    result["raw_return_15m"] = result["close"] / result["close"].shift(15) - 1.0
    result["range"] = result["high"] - result["low"]
    result["body"] = result["close"] - result["open"]
    result["body_ratio_1m"] = result["body"].abs() / result["range"].replace(0.0, np.nan)
    result["clv"] = (result["close"] - result["low"]) / result["range"].replace(0.0, np.nan)
    prior_volume = result["quote_volume"].shift(1)
    result["volume_ratio_1m"] = result["quote_volume"] / prior_volume.rolling(30).median()
    log_volume = np.log(result["quote_volume"].clip(lower=1e-12))
    result["volume_zscore_1m"] = (
        log_volume - log_volume.shift(1).rolling(30).mean()
    ) / log_volume.shift(1).rolling(30).std().replace(0.0, np.nan)
    result["taker_imbalance"] = (
        2.0 * result["taker_buy_quote"] / result["quote_volume"].replace(0.0, np.nan) - 1.0
    )
    result["trade_count_ratio_1m"] = result["trade_count"] / (
        result["trade_count"].shift(1).rolling(30).median().replace(0.0, np.nan)
    )
    atr_map = five.set_index("close_time_ms")["atr"]
    result["atr"] = result["open_time_ms"].map(atr_map).ffill()
    result["atr_fraction"] = result["atr"] / result["close"]
    result["velocity"] = result["raw_return_1m"] / result["atr_fraction"]
    result["velocity_mean_3"] = result["velocity"].rolling(3).mean()
    result["velocity_slope"] = result["velocity"] - result["velocity"].shift(2)
    result["velocity_acceleration"] = result["velocity"].diff()
    result["taker_slope"] = result["taker_imbalance"] - result["taker_imbalance"].shift(2)
    result["taker_acceleration"] = result["taker_imbalance"].diff().diff()
    result["prior_high_20"] = result["high"].shift(1).rolling(20).max()
    result["prior_low_20"] = result["low"].shift(1).rolling(20).min()
    result["prior_high_3"] = result["high"].shift(1).rolling(3).max()
    result["prior_low_3"] = result["low"].shift(1).rolling(3).min()
    return result


def _partition(timestamp: pd.Timestamp, config: dict[str, Any]) -> str | None:
    for name in ("train", "validation"):
        start, end = (pd.Timestamp(value) for value in config["partitions"][name])
        if start <= timestamp < end:
            return name.upper()
    return None


def _context_by_close(frame: pd.DataFrame) -> dict[int, dict[str, float]]:
    return {
        int(row.close_time_ms): {
            "trend": float(row.trend), "velocity": float(row.velocity),
            "rsi": float(row.rsi),
        }
        for row in frame.itertuples()
        if np.isfinite(row.trend) and np.isfinite(row.velocity) and np.isfinite(row.rsi)
    }


def _latest_context(
    mapping: dict[int, dict[str, float]], keys: list[int], timestamp: int
) -> dict[str, float]:
    index = bisect.bisect_right(keys, timestamp) - 1
    return mapping[keys[index]] if index >= 0 else {"trend": 0.0, "velocity": 0.0, "rsi": 50.0}


def _state_features(
    minutes: pd.DataFrame,
    btc_by_time: pd.DataFrame,
    decision_index: int,
    anchor_index: int,
    anchor: Any,
    direction: int,
    context_5m: dict[str, float],
    context_15m: dict[str, float],
) -> dict[str, float]:
    row = minutes.iloc[decision_index]
    start = max(anchor_index + 1, decision_index - 9)
    path = minutes.iloc[start:decision_index + 1]
    if path.empty:
        path = minutes.iloc[decision_index:decision_index + 1]
    directional_taker = direction * path["taker_imbalance"].to_numpy(dtype=float)
    current_favorable = direction * (float(row.close) / float(anchor.close) - 1.0)
    if direction > 0:
        favorable_path = path["high"].to_numpy(dtype=float) / float(anchor.close) - 1.0
    else:
        favorable_path = 1.0 - path["low"].to_numpy(dtype=float) / float(anchor.close)
    peak_index = int(np.argmax(favorable_path))
    peak = max(0.0, float(np.max(favorable_path)))
    pullback = max(0.0, peak - current_favorable)
    atr = float(anchor.atr)
    impulse = abs(float(anchor.close) - float(anchor.open))
    btc = btc_by_time.loc[int(row.open_time_ms)] if int(row.open_time_ms) in btc_by_time.index else None
    favorable_level = float(row.prior_high_20) if direction > 0 else float(row.prior_low_20)
    adverse_level = float(row.prior_low_20) if direction > 0 else float(row.prior_high_20)
    micro_aligned = (
        float(row.high) > float(row.prior_high_3) and float(row.low) > float(row.prior_low_3)
        if direction > 0 else
        float(row.low) < float(row.prior_low_3) and float(row.high) < float(row.prior_high_3)
    )
    micro_opposed = (
        float(row.low) < float(row.prior_low_3) and float(row.high) < float(row.prior_high_3)
        if direction > 0 else
        float(row.high) > float(row.prior_high_3) and float(row.low) > float(row.prior_low_3)
    )
    btc_ret1 = float(btc.raw_return_1m) if btc is not None else 0.0
    btc_directional = direction * btc_ret1
    return {
        "offset_minutes": float(decision_index - anchor_index),
        "impulse_size_atr": safe_ratio(impulse, atr),
        "impulse_volume_ratio": float(anchor.volume_ratio_20),
        "impulse_volume_zscore": float(anchor.volume_zscore_20),
        "impulse_body_ratio": float(anchor.body_ratio),
        "impulse_directional_taker_imbalance": direction * float(anchor.taker_imbalance),
        "directional_return_1m": direction * float(row.raw_return_1m),
        "range_atr_1m": safe_ratio(float(row.range), atr),
        "body_atr_1m": direction * safe_ratio(float(row.body), atr),
        "body_ratio_1m": float(row.body_ratio_1m),
        "clv_directional_1m": directional_clv(row.open, row.high, row.low, row.close, direction),
        "volume_ratio_1m": float(row.volume_ratio_1m),
        "volume_zscore_1m": float(row.volume_zscore_1m),
        "directional_taker_imbalance": direction * float(row.taker_imbalance),
        "trade_count_ratio_1m": float(row.trade_count_ratio_1m),
        "pullback_size_atr": safe_ratio(pullback * float(anchor.close), atr),
        "pullback_fraction": safe_ratio(pullback * float(anchor.close), impulse),
        "pullback_duration_minutes": float(len(path) - 1 - peak_index),
        "pullback_volume_vs_impulse": safe_ratio(
            float(path.quote_volume.mean()), float(anchor.quote_volume) / 5.0
        ),
        "velocity_1": direction * float(row.velocity),
        "velocity_mean_3": direction * float(row.velocity_mean_3),
        "velocity_slope": direction * float(row.velocity_slope),
        "velocity_acceleration": direction * float(row.velocity_acceleration),
        "taker_slope": direction * float(row.taker_slope),
        "taker_acceleration": direction * float(row.taker_acceleration),
        "taker_recovery": float(directional_taker[-1] - np.min(directional_taker)),
        "micro_structure_aligned": float(micro_aligned),
        "micro_structure_opposed": float(micro_opposed),
        "break_of_impulse_extreme": float(
            float(row.close) > float(anchor.high) if direction > 0 else float(row.close) < float(anchor.low)
        ),
        "distance_recent_favorable_atr": direction * safe_ratio(favorable_level - float(row.close), atr),
        "distance_recent_adverse_atr": direction * safe_ratio(float(row.close) - adverse_level, atr),
        "trend_5m": direction * context_5m["trend"],
        "velocity_5m": direction * context_5m["velocity"],
        "rsi_5m_directional": context_5m["rsi"] if direction > 0 else 100.0 - context_5m["rsi"],
        "trend_15m": direction * context_15m["trend"],
        "velocity_15m": direction * context_15m["velocity"],
        "rsi_15m_directional": context_15m["rsi"] if direction > 0 else 100.0 - context_15m["rsi"],
        "btc_directional_return_1m": btc_directional,
        "btc_directional_return_5m": direction * float(btc.raw_return_5m) if btc is not None else 0.0,
        "btc_directional_return_15m": direction * float(btc.raw_return_15m) if btc is not None else 0.0,
        "btc_directional_alignment": float(btc_directional >= 0.0),
    }


def _build_symbol(symbol: str, source_root: Path, output_root: Path, config: dict[str, Any]) -> dict[str, Any]:
    train_start = pd.Timestamp(config["partitions"]["train"][0])
    validation_end = pd.Timestamp(config["partitions"]["validation"][1])
    load_start = train_start - pd.Timedelta(days=7)
    load_end = validation_end + pd.Timedelta(minutes=45)
    to_ms = lambda value: int(value.timestamp() * 1000)
    minutes = _read_minutes(source_root / f"{symbol}.parquet", to_ms(load_start), to_ms(load_end))
    btc = _read_minutes(source_root / "BTCUSDT.parquet", to_ms(load_start), to_ms(load_end))
    five = _aggregate(minutes, 5)
    fifteen = _aggregate(minutes, 15)
    btc_five = _aggregate(btc, 5)
    minutes = _prepare_minutes(minutes, five)
    btc = _prepare_minutes(btc, btc_five).set_index("open_time_ms", drop=False)
    minute_index = {int(value): index for index, value in enumerate(minutes.open_time_ms)}
    five_context = _context_by_close(five)
    fifteen_context = _context_by_close(fifteen)
    five_context_keys = sorted(five_context)
    fifteen_context_keys = sorted(fifteen_context)
    anchor_cfg = config["anchor"]
    anchors = five.loc[
        five["volume_ratio_20"].ge(float(anchor_cfg["minimum_volume_ratio_20"]))
        & five["body_atr"].ge(float(anchor_cfg["minimum_absolute_body_atr"]))
        & five["close"].ne(five["open"])
    ].copy()
    entry_rows: list[dict[str, Any]] = []
    exit_rows: list[dict[str, Any]] = []
    episode_rows: list[dict[str, Any]] = []
    last_side_time = {"LONG": -10**18, "SHORT": -10**18}
    contracts = [config["w3a_entry"]["primary_contract"], *config["w3a_entry"]["secondary_contracts"]]
    offsets = tuple(int(value) for value in anchor_cfg["observation_offsets_minutes"])
    for anchor in anchors.itertuples():
        timestamp = pd.Timestamp(int(anchor.close_time_ms), unit="ms", tz="UTC")
        partition = _partition(timestamp, config)
        if partition is None:
            continue
        side = "LONG" if anchor.close > anchor.open else "SHORT"
        direction = 1 if side == "LONG" else -1
        if int(anchor.close_time_ms) - last_side_time[side] < int(anchor_cfg["cooldown_minutes_symbol_side"]) * 60_000:
            continue
        anchor_index = minute_index.get(int(anchor.close_time_ms))
        if anchor_index is None or anchor_index + 31 >= len(minutes):
            continue
        last_side_time[side] = int(anchor.close_time_ms)
        wave_id = stable_wave_episode_id(symbol, side, int(anchor.close_time_ms))
        c5 = _latest_context(five_context, five_context_keys, int(anchor.close_time_ms))
        c15 = _latest_context(fifteen_context, fifteen_context_keys, int(anchor.close_time_ms))
        common = {
            "wave_episode_id": wave_id, "symbol": symbol, "side": side,
            "direction": direction, "partition": partition,
            "impulse_close_time_ms": int(anchor.close_time_ms),
            "impulse_close": float(anchor.close), "impulse_high": float(anchor.high),
            "impulse_low": float(anchor.low), "atr": float(anchor.atr),
        }
        for offset in offsets:
            decision_index = anchor_index + offset
            entry_index = decision_index + 1
            if entry_index + 15 > len(minutes):
                continue
            features = _state_features(minutes, btc, decision_index, anchor_index, anchor, direction, c5, c15)
            if not all(np.isfinite(features[name]) for name in W3_FEATURE_COLUMNS):
                continue
            decision = minutes.iloc[decision_index]
            entry = minutes.iloc[entry_index]
            row = {
                **common, **features,
                "decision_time_ms": int(decision.open_time_ms),
                "entry_time_ms": int(entry.open_time_ms),
                "entry_price": float(entry.open),
            }
            for contract_index, contract in enumerate(contracts):
                horizon = int(contract["horizon_minutes"])
                path = minutes.iloc[entry_index:entry_index + horizon]
                gross, outcome, mfe, mae = barrier_return(
                    float(entry.open), float(anchor.atr), path.high, path.low, path.close,
                    direction, float(contract["favorable_atr"]), float(contract["adverse_atr"]),
                )
                prefix = "primary" if contract_index == 0 else f"secondary_{contract_index}"
                row[f"{prefix}_gross_return"] = gross
                row[f"{prefix}_outcome"] = outcome
                row[f"{prefix}_mfe_atr"] = safe_ratio(mfe * float(entry.open), float(anchor.atr))
                row[f"{prefix}_mae_atr"] = safe_ratio(mae * float(entry.open), float(anchor.atr))
            entry_rows.append(row)
        entry = minutes.iloc[anchor_index + 1]
        path = minutes.iloc[anchor_index + 1:anchor_index + 31]
        if len(path) != 30:
            continue
        entry_price = float(entry.open)
        if direction > 0:
            favorable_high = path.high.to_numpy(dtype=float) / entry_price - 1.0
            favorable_low = path.low.to_numpy(dtype=float) / entry_price - 1.0
            close_favorable = path.close.to_numpy(dtype=float) / entry_price - 1.0
        else:
            favorable_high = 1.0 - path.low.to_numpy(dtype=float) / entry_price
            favorable_low = 1.0 - path.high.to_numpy(dtype=float) / entry_price
            close_favorable = 1.0 - path.close.to_numpy(dtype=float) / entry_price
        adverse = -favorable_low
        peak = np.maximum.accumulate(favorable_high)
        gate_fraction = float(config["w3b_exit"]["activation_peak_mfe_atr"]) * float(anchor.atr) / entry_price
        gate_indices = np.flatnonzero(peak >= gate_fraction)
        episode = {
            **common, "entry_time_ms": int(entry.open_time_ms), "entry_price": entry_price,
            "gate_reached": bool(len(gate_indices)),
            "gate_minute": int(gate_indices[0] + 1) if len(gate_indices) else -1,
            "peak_mfe": float(np.max(favorable_high)),
            "mae": float(np.max(adverse)),
        }
        for index in range(30):
            suffix = index + 1
            episode[f"high_{suffix}"] = float(path.iloc[index].high)
            episode[f"low_{suffix}"] = float(path.iloc[index].low)
            episode[f"close_{suffix}"] = float(path.iloc[index].close)
        episode_rows.append(episode)
        if not len(gate_indices):
            continue
        for path_index in range(int(gate_indices[0]), 29):
            decision_index = anchor_index + 1 + path_index
            features = _state_features(minutes, btc, decision_index, anchor_index, anchor, direction, c5, c15)
            if not all(np.isfinite(features[name]) for name in W3_FEATURE_COLUMNS):
                continue
            peak_now = float(peak[path_index])
            current = float(close_favorable[path_index])
            peak_index = int(np.argmax(favorable_high[:path_index + 1]))
            future_end = min(30, path_index + 4)
            target = future_giveback_before_new_extreme(
                peak_favorable=peak_now,
                current_favorable=current,
                future_favorable_highs=favorable_high[path_index + 1:future_end],
                future_favorable_lows=favorable_low[path_index + 1:future_end],
                atr_fraction=float(anchor.atr) / entry_price,
            )
            exit_rows.append({
                **common, **features,
                "decision_time_ms": int(minutes.iloc[decision_index].open_time_ms),
                "episode_minute": path_index + 1,
                "entry_price": entry_price,
                "current_favorable_return": current,
                "peak_mfe": peak_now,
                "giveback": max(0.0, peak_now - current),
                "giveback_ratio": safe_ratio(max(0.0, peak_now - current), peak_now),
                "minutes_since_peak": path_index - peak_index,
                "target_giveback_before_new_extreme": target,
            })
    output_root.mkdir(parents=True, exist_ok=True)
    entry_path = output_root / f"{symbol}_entry.parquet"
    exit_path = output_root / f"{symbol}_exit.parquet"
    episode_path = output_root / f"{symbol}_episodes.parquet"
    pd.DataFrame(entry_rows).to_parquet(entry_path, index=False, compression="zstd")
    pd.DataFrame(exit_rows).to_parquet(exit_path, index=False, compression="zstd")
    pd.DataFrame(episode_rows).to_parquet(episode_path, index=False, compression="zstd")
    return {
        "symbol": symbol,
        "entry_rows": len(entry_rows), "exit_rows": len(exit_rows), "episodes": len(episode_rows),
        "entry_sha256": sha256_file(entry_path), "exit_sha256": sha256_file(exit_path),
        "episodes_sha256": sha256_file(episode_path),
        "source_sha256": sha256_file(source_root / f"{symbol}.parquet"),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("config/experiments/aegis_intrabar_wave_w3.yaml"))
    parser.add_argument("--source-root", type=Path, default=Path("data/market_event_fast_track_m1a/full_run_01/cache"))
    parser.add_argument("--output-root", type=Path, default=Path("data/intrabar_wave_w3/dataset_train_validation_01"))
    parser.add_argument("--symbols", nargs="*", default=list(CANONICAL_SYMBOLS))
    parser.add_argument("--workers", type=int, default=3)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    resolve = lambda value: value if value.is_absolute() else root / value
    config_path, source_root, output_root = map(resolve, (args.config, args.source_root, args.output_root))
    config = yaml.safe_load(config_path.read_text())
    if config["partitions"]["final_holdout_state"] != "SEALED" or not set(args.symbols).issubset(CANONICAL_SYMBOLS):
        raise RuntimeError("AEGIS_W3_BUILD_CONTRACT_INVALID")
    output_root.mkdir(parents=True, exist_ok=True)
    os.chmod(output_root, 0o700)
    if args.workers < 1:
        raise RuntimeError("AEGIS_W3_BUILD_WORKERS_INVALID")
    results = []
    context = multiprocessing.get_context("spawn")
    with concurrent.futures.ProcessPoolExecutor(
        max_workers=args.workers, mp_context=context
    ) as executor:
        futures = {
            executor.submit(_build_symbol, symbol, source_root, output_root, config): symbol
            for symbol in args.symbols
        }
        for future in concurrent.futures.as_completed(futures):
            result = future.result()
            results.append(result)
            print(json.dumps({
                "symbol": result["symbol"], "state": "BUILT",
                **{key: result[key] for key in ("entry_rows", "exit_rows", "episodes")},
            }), flush=True)
    results.sort(key=lambda item: list(CANONICAL_SYMBOLS).index(item["symbol"]))
    manifest = {
        "schema_version": "aegis-intrabar-wave-w3-dataset-manifest-v1",
        "config_sha256": sha256_file(config_path),
        "feature_schema": config["features"]["feature_schema"],
        "feature_columns": list(W3_FEATURE_COLUMNS),
        "results": results,
        "total_entry_rows": sum(item["entry_rows"] for item in results),
        "total_exit_rows": sum(item["exit_rows"] for item in results),
        "total_episodes": sum(item["episodes"] for item in results),
        "final_holdout_state": "SEALED",
        "final_holdout_outcomes_read": False,
        "aggregate_trade_ticks_used": False,
        "order_book_reconstructed": False,
        "authenticated_requests": 0,
        "exchange_mutations": 0,
    }
    path = output_root / "manifest.json"
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    os.chmod(path, 0o600)
    print(json.dumps({"manifest": str(path), "holdout": "SEALED"}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
