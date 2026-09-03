"""Causal early-path reconstruction for Aegis W12."""

from __future__ import annotations

import math
from typing import Any, Mapping

import numpy as np
import pandas as pd

from aegis.research.entry_safety_gate_w11 import reconstruct_path
from aegis.research.live_entry_multitimeframe import add_directional_context, attach_features


def w12_split(timestamp: pd.Timestamp, config: Mapping[str, Any]) -> str:
    value = pd.Timestamp(timestamp)
    if value.tzinfo is None:
        value = value.tz_localize("UTC")
    train_end = pd.Timestamp(config["splits"]["train_end_exclusive"])
    validation_end = pd.Timestamp(config["splits"]["validation_end_exclusive"])
    holdout_end = pd.Timestamp(config["splits"]["final_holdout_end_exclusive"])
    if value < train_end:
        return "W12_TRAIN"
    if value < validation_end:
        return "W12_VALIDATION"
    if value < holdout_end:
        return "W12_FINAL_HOLDOUT"
    return "EXCLUDED_W11_HOLDOUT"


def complete_bar_decision_time(signal_time: pd.Timestamp, state_index: int) -> pd.Timestamp:
    timestamp = pd.Timestamp(signal_time)
    first_complete_bar_open = timestamp.ceil("min")
    if timestamp == timestamp.floor("min"):
        first_complete_bar_open = timestamp
    return first_complete_bar_open + pd.Timedelta(minutes=state_index)


def early_path_features(
    candles: pd.DataFrame,
    *,
    signal_time: pd.Timestamp,
    decision_time: pd.Timestamp,
    signal_price: float,
    side: str,
    move_threshold_bps: float,
) -> dict[str, float]:
    frame = candles.copy()
    if "open_time" not in frame:
        frame["open_time"] = pd.to_datetime(frame["open_time_ms"], unit="ms", utc=True)
    timestamp = pd.Timestamp(signal_time)
    start = timestamp.ceil("min")
    if timestamp == timestamp.floor("min"):
        start = timestamp
    observed = frame.loc[
        frame["open_time"].ge(start) & frame["open_time"].lt(pd.Timestamp(decision_time))
    ].sort_values("open_time")
    if observed.empty:
        raise ValueError("early path requires at least one complete post-signal bar")
    direction = 1.0 if side == "LONG" else -1.0
    closes = observed["close"].to_numpy(float)
    highs = observed["high"].to_numpy(float)
    lows = observed["low"].to_numpy(float)
    favorable = np.where(direction > 0, highs / signal_price - 1.0, 1.0 - lows / signal_price) * 10_000.0
    adverse = np.where(direction > 0, 1.0 - lows / signal_price, highs / signal_price - 1.0) * 10_000.0
    favorable = np.maximum(favorable, 0.0)
    adverse = np.maximum(adverse, 0.0)
    directional_closes = direction * (closes / signal_price - 1.0) * 10_000.0
    path_points = np.concatenate([[0.0], directional_closes])
    path_length = float(np.abs(np.diff(path_points)).sum())
    net = float(directional_closes[-1])
    volume = observed["volume"].to_numpy(float)
    taker = observed["taker_buy_volume"].to_numpy(float)
    raw_imbalance = np.divide(2.0 * taker - volume, volume, out=np.zeros_like(volume), where=volume > 0)
    directional_imbalance = direction * raw_imbalance
    weighted_imbalance = float(np.average(directional_imbalance, weights=np.maximum(volume, 1e-12)))
    favorable_indices = np.flatnonzero(favorable >= move_threshold_bps)
    adverse_indices = np.flatnonzero(adverse >= move_threshold_bps)
    first_favorable = int(favorable_indices[0]) if len(favorable_indices) else -1
    first_adverse = int(adverse_indices[0]) if len(adverse_indices) else -1
    favorable_first = first_favorable >= 0 and (first_adverse < 0 or first_favorable < first_adverse)
    adverse_first = first_adverse >= 0 and (first_favorable < 0 or first_adverse <= first_favorable)
    elapsed = (pd.Timestamp(decision_time) - timestamp).total_seconds()
    acceleration = float(directional_closes[-1] - directional_closes[-2]) if len(directional_closes) > 1 else 0.0
    taker_velocity = float(directional_imbalance[-1] - directional_imbalance[0]) if len(directional_imbalance) > 1 else 0.0
    return {
        "early_elapsed_seconds": float(elapsed),
        "early_directional_return_bps": net,
        "early_favorable_excursion_bps": float(favorable.max()),
        "early_adverse_excursion_bps": float(adverse.max()),
        "early_path_efficiency": abs(net) / max(path_length, 1e-9),
        "early_velocity_bps_per_minute": net / max(elapsed / 60.0, 1e-9),
        "early_acceleration_bps": acceleration,
        "early_new_favorable_extreme": float(int(np.argmax(favorable)) == len(favorable) - 1),
        "early_new_adverse_extreme": float(int(np.argmax(adverse)) == len(adverse) - 1),
        "early_favorable_move_first": float(favorable_first),
        "early_adverse_move_first": float(adverse_first),
        "early_first_favorable_bar": float(first_favorable + 1) if first_favorable >= 0 else math.nan,
        "early_first_adverse_bar": float(first_adverse + 1) if first_adverse >= 0 else math.nan,
        "early_directional_taker_imbalance": weighted_imbalance,
        "early_taker_velocity": taker_velocity,
        "early_taker_acceleration": taker_velocity,
        "early_aligned_flow_persistence": float((directional_imbalance > 0.0).mean()),
        "early_volume_sum": float(volume.sum()),
        "early_price_response_per_flow": net / max(abs(weighted_imbalance), 0.05),
        "early_favorable_flow_effective": float(max(net, 0.0) * max(weighted_imbalance, 0.0)),
        "early_adverse_flow_effective": float(max(-net, 0.0) * max(-weighted_imbalance, 0.0)),
    }


def build_early_path_dataset(
    entries: pd.DataFrame,
    candles_by_symbol: dict[str, pd.DataFrame],
    entry_prices: Mapping[str, float],
    config: Mapping[str, Any],
    *,
    include_holdout: bool = False,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    split_counts: dict[str, int] = {}
    excluded_w11 = 0
    excluded_w12_holdout = 0
    for entry in entries.to_dict(orient="records"):
        signal_time = pd.Timestamp(entry["opened_at"])
        split = w12_split(signal_time, config)
        if split == "EXCLUDED_W11_HOLDOUT":
            excluded_w11 += 1
            continue
        if split == "W12_FINAL_HOLDOUT" and not include_holdout:
            excluded_w12_holdout += 1
            continue
        symbol = str(entry["symbol"])
        signal_price = float(entry_prices.get(str(entry["trade_id"]), math.nan))
        if not math.isfinite(signal_price) or signal_price <= 0:
            continue
        source = candles_by_symbol[symbol]
        for state_index in (1, 2):
            decision_time = complete_bar_decision_time(signal_time, state_index)
            source_with_time = source
            if "open_time" not in source_with_time:
                source_with_time = source.copy()
                source_with_time["open_time"] = pd.to_datetime(source_with_time["open_time_ms"], unit="ms", utc=True)
            entry_bar = source_with_time.loc[source_with_time["open_time"].eq(decision_time)].head(1)
            if entry_bar.empty:
                continue
            candidate_price = float(entry_bar.iloc[0]["open"])
            outcome = reconstruct_path(
                source_with_time, decision_time=decision_time, entry_price=candidate_price,
                side=str(entry["side"]), horizon_minutes=int(config["outcome"]["horizon_minutes_from_candidate_entry"]),
                favorable_barrier_bps=float(config["outcome"]["favorable_barrier_bps"]),
                adverse_barrier_bps=float(config["outcome"]["adverse_barrier_bps"]),
                cost_bps=float(config["outcome"]["baseline_cost_bps"]),
            )
            if outcome is None:
                continue
            early = early_path_features(
                source_with_time, signal_time=signal_time, decision_time=decision_time,
                signal_price=signal_price, side=str(entry["side"]),
                move_threshold_bps=float(config["outcome"]["early_move_threshold_bps"]),
            )
            direction = 1.0 if entry["side"] == "LONG" else -1.0
            moved = direction * (candidate_price / signal_price - 1.0) * 10_000.0
            rows.append({
                **entry,
                "original_live_signal_id": str(entry["trade_id_hash"]),
                "w12_split": split,
                "state_index": state_index,
                "state": f"FULL_POST_SIGNAL_BAR_{state_index}",
                "signal_timestamp": signal_time.isoformat(),
                "decision_timestamp": decision_time.isoformat(),
                "signal_entry_price": signal_price,
                "candidate_entry_price": candidate_price,
                "bps_moved_before_entry": moved,
                "entry_price_improvement_bps": -moved,
                "missed_mfe_before_entry_bps": early["early_favorable_excursion_bps"],
                **early,
                **{f"remaining_{key}": value for key, value in outcome.__dict__.items() if key not in {"entry_price"}},
            })
        split_counts[split] = split_counts.get(split, 0) + 1
    frame = pd.DataFrame(rows)
    complete = frame.groupby("original_live_signal_id")["state_index"].nunique()
    complete_ids = complete.loc[complete.eq(2)].index
    frame = frame.loc[frame["original_live_signal_id"].isin(complete_ids)].copy()

    feature_rows = frame.copy()
    feature_rows["opened_at"] = feature_rows["decision_timestamp"]
    feature_rows = add_directional_context(
        attach_features(feature_rows, candles_by_symbol, config["sources"]["timeframes_minutes"]),
        config["sources"]["timeframes_minutes"],
    )
    btc_rows = frame.copy()
    btc_rows["opened_at"] = btc_rows["decision_timestamp"]
    btc_rows["symbol"] = "BTCUSDT"
    btc_rows = add_directional_context(
        attach_features(btc_rows, candles_by_symbol, config["sources"]["timeframes_minutes"]),
        config["sources"]["timeframes_minutes"],
    )
    btc_columns = [column for column in btc_rows if column.startswith(("tf", "dir"))]
    btc_context = btc_rows[["original_live_signal_id", "state_index", *btc_columns]].rename(
        columns={column: "btc_" + column for column in btc_columns}
    )
    feature_rows = feature_rows.merge(
        btc_context, on=["original_live_signal_id", "state_index"], how="left", validate="one_to_one"
    ).copy()
    feature_rows["remaining_positive_label"] = (
        feature_rows["remaining_net_return_bps"].gt(0.0)
        & feature_rows["remaining_first_barrier_hit"].ne("ADVERSE_FIRST")
    ).astype(int)
    feature_rows["confirmation_too_late"] = (
        feature_rows["missed_mfe_before_entry_bps"].ge(float(config["outcome"]["favorable_barrier_bps"]))
        & feature_rows["remaining_net_return_bps"].le(0.0)
    ).astype(int)
    audit = {
        "rows": int(len(feature_rows)),
        "complete_episodes": int(feature_rows["original_live_signal_id"].nunique()),
        "split_episodes": feature_rows.groupby("w12_split")["original_live_signal_id"].nunique().astype(int).to_dict(),
        "excluded_w12_holdout": int(excluded_w12_holdout),
        "excluded_w11_august_holdout": int(excluded_w11),
        "subminute_features_available": False,
        "optional_l2_overlap": "NONE",
    }
    return feature_rows.sort_values(["decision_timestamp", "state_index"]).reset_index(drop=True), audit


def w12_feature_groups(columns: list[str]) -> dict[str, list[str]]:
    early_price = sorted(column for column in columns if column.startswith("early_") and not any(
        token in column for token in ("taker", "flow", "volume")
    )) + ["bps_moved_before_entry", "entry_price_improvement_bps", "missed_mfe_before_entry_bps"]
    flow = sorted(column for column in columns if column.startswith("early_") and any(
        token in column for token in ("taker", "flow", "volume", "response")
    ))
    context_suffixes = ("return_1_bps", "return_3_bps", "ema7_slope_atr", "atr_pct_bps", "volume_ratio20", "taker_imbalance")
    context = sorted(column for column in columns if column.startswith(("tf", "dir", "btc_tf", "btc_dir")) and column.endswith(context_suffixes))
    space_suffixes = ("favorable_space_atr", "adverse_space_atr", "ema25_extension_atr", "ema99_extension_atr", "prior_move_6_atr", "rsi12_extension")
    space = sorted(column for column in columns if column.startswith(("tf", "dir", "btc_tf", "btc_dir")) and column.endswith(space_suffixes))
    full_extra_suffixes = (
        "rsi6", "rsi12", "rsi24", "body_ratio", "path_efficiency_6",
        "aligned_breakout", "opposed_breakout", "trend_age", "volume_z50",
    )
    full_extra = sorted(column for column in columns if column.startswith(("tf", "dir", "btc_tf", "btc_dir")) and column.endswith(full_extra_suffixes))
    return {
        "PRICE_PATH_ONLY": sorted(set(early_price)),
        "PRICE_PATH_PLUS_FLOW": sorted(set(early_price + flow)),
        "PRICE_PATH_FLOW_CONTEXT": sorted(set(early_price + flow + context)),
        "PRICE_PATH_FLOW_CONTEXT_SPACE": sorted(set(early_price + flow + context + space)),
        "FULL_W12": sorted(set(early_price + flow + context + space + full_extra)),
    }
