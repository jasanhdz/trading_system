"""Causal path reconstruction and policy utilities for Aegis W11."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd

from aegis.research.live_entry_multitimeframe import add_directional_context, attach_features


ACTION_NAMES = {0: "ENTER_NOW", 1: "WAIT_1M", 2: "WAIT_2M", 3: "WAIT_3M"}


@dataclass(frozen=True)
class PathOutcome:
    entry_price: float
    terminal_price: float
    gross_return_bps: float
    net_return_bps: float
    mfe_bps: float
    mae_bps: float
    first_barrier_hit: str
    time_to_first_favorable_move_minutes: float
    time_to_first_adverse_move_minutes: float
    time_to_mfe_minutes: float
    time_to_mae_minutes: float


def w11_split(timestamp: pd.Timestamp, config: Mapping[str, Any]) -> str:
    value = pd.Timestamp(timestamp)
    if value.tzinfo is None:
        value = value.tz_localize("UTC")
    train_end = pd.Timestamp(config["splits"]["train_end_exclusive"])
    validation_end = pd.Timestamp(config["splits"]["validation_end_exclusive"])
    if value < train_end:
        return "W11_TRAIN"
    if value < validation_end:
        return "W11_VALIDATION"
    return "W11_FINAL_HOLDOUT"


def reconstruct_path(
    candles: pd.DataFrame,
    *,
    decision_time: pd.Timestamp,
    entry_price: float,
    side: str,
    horizon_minutes: int,
    favorable_barrier_bps: float,
    adverse_barrier_bps: float,
    cost_bps: float,
) -> PathOutcome | None:
    direction = 1.0 if side == "LONG" else -1.0
    timestamp = pd.Timestamp(decision_time)
    start = timestamp.ceil("min")
    if timestamp == timestamp.floor("min"):
        start = timestamp
    end = timestamp + pd.Timedelta(minutes=horizon_minutes)
    frame = candles.copy()
    if "open_time" not in frame:
        frame["open_time"] = pd.to_datetime(frame["open_time_ms"], unit="ms", utc=True)
    path = frame.loc[frame["open_time"].between(start, end, inclusive="left")].sort_values("open_time")
    if len(path) < max(3, int(horizon_minutes * 0.8)) or not math.isfinite(entry_price) or entry_price <= 0:
        return None

    favorable = np.where(
        direction > 0,
        (path["high"].to_numpy(float) / entry_price - 1.0) * 10_000.0,
        (1.0 - path["low"].to_numpy(float) / entry_price) * 10_000.0,
    )
    adverse = np.where(
        direction > 0,
        (1.0 - path["low"].to_numpy(float) / entry_price) * 10_000.0,
        (path["high"].to_numpy(float) / entry_price - 1.0) * 10_000.0,
    )
    favorable = np.maximum(favorable, 0.0)
    adverse = np.maximum(adverse, 0.0)

    first_barrier = "NEITHER"
    for favorable_value, adverse_value in zip(favorable, adverse):
        if adverse_value >= adverse_barrier_bps:
            first_barrier = "ADVERSE_FIRST"
            break
        if favorable_value >= favorable_barrier_bps:
            first_barrier = "FAVORABLE_FIRST"
            break

    times = (path["open_time"] - timestamp).dt.total_seconds().to_numpy(float) / 60.0
    terminal = float(path.iloc[-1]["close"])
    gross = direction * (terminal / entry_price - 1.0) * 10_000.0
    first_favorable = np.flatnonzero(favorable > 0.0)
    first_adverse = np.flatnonzero(adverse > 0.0)
    return PathOutcome(
        entry_price=float(entry_price),
        terminal_price=terminal,
        gross_return_bps=float(gross),
        net_return_bps=float(gross - cost_bps),
        mfe_bps=float(favorable.max()),
        mae_bps=float(adverse.max()),
        first_barrier_hit=first_barrier,
        time_to_first_favorable_move_minutes=float(times[first_favorable[0]]) if len(first_favorable) else math.nan,
        time_to_first_adverse_move_minutes=float(times[first_adverse[0]]) if len(first_adverse) else math.nan,
        time_to_mfe_minutes=float(times[int(np.argmax(favorable))]),
        time_to_mae_minutes=float(times[int(np.argmax(adverse))]),
    )


def delayed_entry_price(candles: pd.DataFrame, decision_time: pd.Timestamp) -> float:
    timestamp = pd.Timestamp(decision_time)
    frame = candles
    if "open_time" not in frame:
        frame = frame.copy()
        frame["open_time"] = pd.to_datetime(frame["open_time_ms"], unit="ms", utc=True)
    match = frame.loc[frame["open_time"].ge(timestamp.ceil("min"))].head(1)
    return float(match.iloc[0]["open"]) if not match.empty else math.nan


def build_candidate_dataset(
    entries: pd.DataFrame,
    candles_by_symbol: dict[str, pd.DataFrame],
    entry_prices: Mapping[str, float],
    config: Mapping[str, Any],
    *,
    include_holdout: bool = False,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    excluded_holdout = 0
    counterfactual = config["counterfactual"]
    for entry in entries.to_dict(orient="records"):
        original_time = pd.Timestamp(entry["opened_at"])
        split = w11_split(original_time, config)
        if split == "W11_FINAL_HOLDOUT" and not include_holdout:
            excluded_holdout += 1
            continue
        symbol = str(entry["symbol"])
        source = candles_by_symbol[symbol]
        for delay in counterfactual["delays_minutes"]:
            decision_time = original_time + pd.Timedelta(minutes=int(delay))
            price = (
                float(entry_prices.get(str(entry["trade_id"]), math.nan))
                if int(delay) == 0
                else delayed_entry_price(source, decision_time)
            )
            outcome = reconstruct_path(
                source,
                decision_time=decision_time,
                entry_price=price,
                side=str(entry["side"]),
                horizon_minutes=int(counterfactual["primary_horizon_minutes"]),
                favorable_barrier_bps=float(counterfactual["favorable_barrier_bps"]),
                adverse_barrier_bps=float(counterfactual["adverse_barrier_bps"]),
                cost_bps=float(counterfactual["baseline_cost_bps"]),
            )
            if outcome is None:
                continue
            row = dict(entry)
            row.update({
                "live_signal_episode_id": str(entry["trade_id_hash"]),
                "w11_split": split,
                "delay_minutes": int(delay),
                "candidate_action": ACTION_NAMES[int(delay)],
                "decision_timestamp": decision_time.isoformat(),
                **outcome.__dict__,
            })
            rows.append(row)
    candidates = pd.DataFrame(rows)
    if candidates.empty:
        raise RuntimeError("W11 produced no reconstructable candidate paths")
    complete = candidates.groupby("live_signal_episode_id")["delay_minutes"].nunique()
    complete_ids = complete.loc[complete.eq(len(counterfactual["delays_minutes"]))].index
    candidates = candidates.loc[candidates["live_signal_episode_id"].isin(complete_ids)].copy()

    feature_entries = candidates.copy()
    feature_entries["opened_at"] = feature_entries["decision_timestamp"]
    feature_entries = add_directional_context(
        attach_features(feature_entries, candles_by_symbol, config["sources"]["timeframes_minutes"]),
        config["sources"]["timeframes_minutes"],
    )
    btc_entries = candidates.copy()
    btc_entries["opened_at"] = btc_entries["decision_timestamp"]
    btc_entries["symbol"] = "BTCUSDT"
    btc_entries = add_directional_context(
        attach_features(btc_entries, candles_by_symbol, config["sources"]["timeframes_minutes"]),
        config["sources"]["timeframes_minutes"],
    )
    btc_columns = [column for column in btc_entries if column.startswith(("tf", "dir"))]
    btc_context = btc_entries[["live_signal_episode_id", "delay_minutes", *btc_columns]].rename(
        columns={column: "btc_" + column for column in btc_columns}
    )
    feature_entries = feature_entries.merge(
        btc_context, on=["live_signal_episode_id", "delay_minutes"], how="left", validate="one_to_one"
    ).copy()
    baseline = feature_entries.loc[feature_entries["delay_minutes"].eq(0), [
        "live_signal_episode_id", "entry_price", "decision_timestamp",
        "dir1m__taker_imbalance", "tf1m__atr_pct_bps", "tf1m__volume_ratio20",
    ]].rename(columns={
        "entry_price": "baseline_entry_price",
        "decision_timestamp": "baseline_decision_timestamp",
        "dir1m__taker_imbalance": "baseline_taker_imbalance",
        "tf1m__atr_pct_bps": "baseline_atr_bps",
        "tf1m__volume_ratio20": "baseline_volume_ratio",
    })
    feature_entries = feature_entries.merge(baseline, on="live_signal_episode_id", how="left", validate="many_to_one")
    direction = feature_entries["side"].map({"LONG": 1.0, "SHORT": -1.0}).astype(float)
    feature_entries["confirmation_directional_move_bps"] = direction * (
        feature_entries["entry_price"] / feature_entries["baseline_entry_price"] - 1.0
    ) * 10_000.0
    feature_entries["confirmation_taker_change"] = (
        feature_entries["dir1m__taker_imbalance"] - feature_entries["baseline_taker_imbalance"]
    )
    feature_entries["confirmation_atr_change"] = feature_entries["tf1m__atr_pct_bps"] - feature_entries["baseline_atr_bps"]
    feature_entries["confirmation_volume_change"] = feature_entries["tf1m__volume_ratio20"] - feature_entries["baseline_volume_ratio"]
    feature_entries["unsafe_now_label"] = (
        feature_entries["net_return_bps"].lt(0.0)
        | feature_entries["first_barrier_hit"].eq("ADVERSE_FIRST")
    ).astype(int)
    feature_entries["mfe_before_mae"] = feature_entries["time_to_mfe_minutes"].lt(feature_entries["time_to_mae_minutes"])
    feature_entries["mae_before_mfe"] = feature_entries["time_to_mae_minutes"].lt(feature_entries["time_to_mfe_minutes"])
    audit = {
        "candidate_rows": int(len(feature_entries)),
        "complete_episodes": int(feature_entries["live_signal_episode_id"].nunique()),
        "excluded_holdout_episodes": int(excluded_holdout),
        "split_episodes": feature_entries.groupby("w11_split")["live_signal_episode_id"].nunique().astype(int).to_dict(),
    }
    return feature_entries.sort_values(["decision_timestamp", "delay_minutes"]).reset_index(drop=True), audit


def feature_families(columns: Iterable[str], config: Mapping[str, Any]) -> dict[str, list[str]]:
    available = list(columns)
    families: dict[str, list[str]] = {}
    for name in ("exhaustion", "opposition", "space", "volatility"):
        suffixes = config["features"][name]["suffixes"]
        families[name] = sorted({
            column for column in available
            if column.startswith(("tf", "dir", "btc_tf", "btc_dir"))
            and any(column.endswith(suffix) for suffix in suffixes)
        })
    families["confirmation"] = sorted(set().union(*families.values()) | set(config["features"]["confirmation_extra"]))
    return families


def path_order_summary(frame: pd.DataFrame) -> list[dict[str, Any]]:
    baseline = frame.loc[frame["delay_minutes"].eq(0)]
    rows = []
    for label, group in baseline.groupby("entry_class", sort=True):
        rows.append({
            "entry_class": str(label),
            "episodes": int(len(group)),
            "favorable_first_rate": float(group["first_barrier_hit"].eq("FAVORABLE_FIRST").mean()),
            "adverse_first_rate": float(group["first_barrier_hit"].eq("ADVERSE_FIRST").mean()),
            "neither_rate": float(group["first_barrier_hit"].eq("NEITHER").mean()),
            "mfe_before_mae_rate": float(group["mfe_before_mae"].mean()),
            "mae_before_mfe_rate": float(group["mae_before_mfe"].mean()),
            "median_mfe_bps": float(group["mfe_bps"].median()),
            "median_mae_bps": float(group["mae_bps"].median()),
            "median_net_60m_bps": float(group["net_return_bps"].median()),
        })
    return rows
