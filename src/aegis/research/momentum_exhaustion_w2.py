"""Causal episode construction and policy evaluation for W2 research."""

from __future__ import annotations

import hashlib
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd


SCHEMA_VERSION = "aegis-momentum-exhaustion-w2-dataset-v1"
SIDES = ("LONG", "SHORT")
FEATURE_COLUMNS = (
    "peak_mfe_atr",
    "current_return_atr",
    "giveback_atr",
    "giveback_ratio",
    "bars_since_peak",
    "new_extreme_last_1",
    "new_extreme_last_2",
    "new_extreme_last_3",
    "distance_from_peak_atr",
    "directional_velocity_1",
    "directional_velocity_2",
    "directional_velocity_3",
    "velocity_decay",
    "mean_velocity_last_2",
    "mean_velocity_last_3",
    "velocity_slope",
    "directional_acceleration",
    "acceleration_slope",
    "volume_ratio_20",
    "volume_z_20",
    "volume_slope",
    "volume_decay",
    "directional_taker_imbalance",
    "taker_imbalance_slope",
    "taker_imbalance_decay",
    "directional_delta_velocity",
    "directional_delta_acceleration",
    "opposite_body_ratio",
    "opposite_volume_ratio",
    "opposite_clv",
    "retracement_fraction",
    "favorable_structure",
    "structure_deterioration",
    "directional_price_vs_ma7_atr",
    "directional_price_vs_ma25_atr",
    "directional_ma7_slope_atr",
    "directional_ma25_slope_atr",
    "ma7_slope_decay",
    "directional_rsi_extension",
    "rsi_slope",
    "atr_percentile",
    "atr_slope",
    "realized_volatility_12",
    "directional_15m_return",
    "directional_15m_ma25_slope",
    "directional_btc_5m_return",
    "directional_btc_15m_return",
    "directional_btc_taker_imbalance",
    "btc_opposes_position",
)


class MomentumExhaustionContractError(ValueError):
    """Raised when W2 causal or episode invariants are violated."""


def stable_episode_id(symbol: str, side: str, entry_timestamp_ms: int) -> str:
    identity = f"{symbol}:{side}:{entry_timestamp_ms}".encode()
    return f"W2-{hashlib.sha256(identity).hexdigest()[:24]}"


def partition_for_timestamp(timestamp_ms: int, config: Mapping[str, object]) -> str:
    for name, bounds in config["partitions"].items():
        if name in {"purge_minutes", "final_holdout_state", "repeated_holdout_selection"}:
            continue
        start = int(pd.Timestamp(bounds[0]).timestamp() * 1_000)
        end = int(pd.Timestamp(bounds[1]).timestamp() * 1_000)
        if start <= timestamp_ms < end:
            return name.upper()
    return "OUT_OF_SCOPE"


def select_nonoverlapping_candidates(
    candidates: pd.DataFrame,
    config: Mapping[str, object],
    *,
    include_partitions: Sequence[str] = ("TRAIN", "VALIDATION"),
) -> pd.DataFrame:
    required = {"symbol", "side", "timestamp_ms"}
    if not required.issubset(candidates.columns):
        raise MomentumExhaustionContractError("AEGIS_W2_CANDIDATE_SCHEMA_INVALID")
    cooldown_ms = int(
        config["populations"]["simulated"][
            "non_overlap_cooldown_minutes_by_symbol_side"
        ]
    ) * 60_000
    frame = candidates.copy()
    frame["partition"] = [
        partition_for_timestamp(int(value), config) for value in frame["timestamp_ms"]
    ]
    frame = frame.loc[frame["partition"].isin(include_partitions)].sort_values(
        ["symbol", "side", "timestamp_ms"]
    )
    purge_ms = int(config["partitions"]["purge_minutes"]) * 60_000
    partition_ends = {
        name.upper(): int(pd.Timestamp(bounds[1]).timestamp() * 1_000)
        for name, bounds in config["partitions"].items()
        if isinstance(bounds, list)
    }
    frame = frame.loc[
        frame.apply(
            lambda row: int(row.timestamp_ms)
            < partition_ends[str(row.partition)] - purge_ms,
            axis=1,
        )
    ]
    selected: list[int] = []
    for _, rows in frame.groupby(["symbol", "side"], sort=True):
        previous = -10**30
        for index, timestamp in zip(rows.index, rows["timestamp_ms"], strict=True):
            value = int(timestamp)
            if value >= previous + cooldown_ms:
                selected.append(index)
                previous = value
    return frame.loc[selected].sort_values("timestamp_ms").reset_index(drop=True)


def next_complete_minute_open(timestamp_ms: int) -> int:
    return ((int(timestamp_ms) // 60_000) + 1) * 60_000


def next_complete_five_minute_open(timestamp_ms: int) -> int:
    minute = next_complete_minute_open(timestamp_ms)
    return ((minute + 299_999) // 300_000) * 300_000


def _rolling_percentile(values: pd.Series, window: int) -> pd.Series:
    def percentile(sample: np.ndarray) -> float:
        if len(sample) == 0 or not np.isfinite(sample[-1]):
            return math.nan
        finite = sample[np.isfinite(sample)]
        if not len(finite):
            return math.nan
        return float((finite <= sample[-1]).mean())

    return values.rolling(window, min_periods=max(20, window // 4)).apply(
        percentile, raw=True
    )


def enrich_w2_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Add causal feature derivatives not already supplied by W1."""

    required = {
        "open_time_ms", "close_time_ms", "open", "high", "low", "close",
        "atr", "body_ratio", "clv", "quote_volume", "volume_ratio_20",
        "volume_z_20", "taker_imbalance", "delta_velocity",
        "delta_acceleration", "velocity_atr_1", "acceleration_atr", "rsi_6",
        "price_vs_ma_7_atr", "price_vs_ma_25_atr", "ma_7_slope_atr",
        "ma_25_slope_atr", "higher_high", "higher_low", "lower_high",
        "lower_low", "context_15m_return_1", "context_15m_ma_25_slope_atr",
        "btc_5m_return_1", "btc_15m_return_1", "btc_5m_taker_imbalance",
    }
    if not required.issubset(frame.columns):
        missing = sorted(required.difference(frame.columns))
        raise MomentumExhaustionContractError(
            f"AEGIS_W2_FEATURE_SOURCE_MISSING:{','.join(missing)}"
        )
    result = frame.copy().sort_values("open_time_ms").reset_index(drop=True)
    result["atr_percentile"] = _rolling_percentile(result["atr"], 288)
    result["atr_slope"] = result["atr"].pct_change(3, fill_method=None)
    result["realized_volatility_12"] = result["return_1"].rolling(
        12, min_periods=12
    ).std(ddof=0)
    result["volume_slope"] = result["volume_ratio_20"].diff(2) / 2.0
    result["volume_decay"] = result["volume_ratio_20"].diff()
    result["taker_imbalance_slope"] = result["taker_imbalance"].diff(2) / 2.0
    result["taker_imbalance_decay"] = result["taker_imbalance"].diff()
    result["ma7_slope_decay"] = result["ma_7_slope_atr"].diff()
    result["rsi_slope"] = result["rsi_6"].diff(2) / 2.0
    return result


def _first_hit(values: np.ndarray, threshold: float, *, greater: bool) -> int | None:
    hits = values >= threshold if greater else values <= threshold
    return int(np.flatnonzero(hits)[0]) if hits.any() else None


def _future_targets(
    favorable_high: np.ndarray,
    favorable_low: np.ndarray,
    closes: np.ndarray,
    *,
    index: int,
    peak_at_index: float,
    atr: float,
    maximum_horizon: int = 3,
) -> dict[str, float | bool]:
    result: dict[str, float | bool] = {}
    current_favorable = closes[index]
    for horizon in (1, 2, 3):
        stop = min(len(closes), index + horizon + 1)
        future_low = favorable_low[index + 1:stop]
        future_high = favorable_high[index + 1:stop]
        if not len(future_low):
            result[f"target_giveback_025_atr_next_{horizon}"] = False
            continue
        result[f"target_giveback_025_atr_next_{horizon}"] = bool(
            (current_favorable - future_low).max(initial=0.0) >= 0.25 * atr
        )
    stop = min(len(closes), index + maximum_horizon + 1)
    future_low = favorable_low[index + 1:stop]
    future_high = favorable_high[index + 1:stop]
    giveback_distance = current_favorable - future_low if len(future_low) else np.array([])
    new_extreme_distance = future_high - peak_at_index if len(future_high) else np.array([])
    giveback_hit = _first_hit(giveback_distance, 0.25 * atr, greater=True)
    extreme_hit = _first_hit(new_extreme_distance, 0.25 * atr, greater=True)
    result["target_giveback_before_new_extreme"] = bool(
        giveback_hit is not None and (extreme_hit is None or giveback_hit <= extreme_hit)
    )
    result["target_new_extreme_before_giveback"] = bool(
        extreme_hit is not None and (giveback_hit is None or extreme_hit < giveback_hit)
    )
    current_peak = max(peak_at_index, 1e-12)
    for fraction in (0.25, 0.40, 0.60):
        threshold = fraction * current_peak
        result[f"target_peak_giveback_{int(fraction * 100)}pct"] = bool(
            len(future_low) and (peak_at_index - future_low).max() >= threshold
        )
    result["target_additional_mfe_atr"] = float(
        max(0.0, future_high.max(initial=peak_at_index) - peak_at_index) / atr
    )
    result["target_future_giveback_atr"] = float(
        max(0.0, current_favorable - future_low.min(initial=current_favorable)) / atr
    )
    return result


def build_episode_tables(
    symbol: str,
    features: pd.DataFrame,
    candidates: pd.DataFrame,
    config: Mapping[str, object],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build one episode record and nested causal decisions for each anchor."""

    frame = enrich_w2_features(features)
    maximum_bars = int(
        config["populations"]["simulated"]["maximum_episode_minutes"]
    ) // int(config["universe"]["decision_interval_minutes"])
    minimum_gate = min(float(x) for x in config["profit_activation_gates_atr"])
    open_times = frame["open_time_ms"].to_numpy(dtype=np.int64)
    episodes: list[dict[str, object]] = []
    decisions: list[dict[str, object]] = []

    for candidate in candidates.loc[candidates["symbol"].eq(symbol)].itertuples():
        side = str(candidate.side)
        sign = 1.0 if side == "LONG" else -1.0
        entry_timestamp = next_complete_five_minute_open(int(candidate.timestamp_ms))
        start = int(np.searchsorted(open_times, entry_timestamp))
        if start >= len(frame) or open_times[start] != entry_timestamp:
            continue
        stop = start + maximum_bars
        if stop > len(frame):
            continue
        path = frame.iloc[start:stop].reset_index(drop=True)
        expected_last = entry_timestamp + (maximum_bars - 1) * 300_000
        if int(path.iloc[-1].open_time_ms) != expected_last:
            continue
        entry_price = float(path.iloc[0].open)
        entry_atr = float(path.iloc[0].atr)
        if not math.isfinite(entry_price) or not math.isfinite(entry_atr) or entry_atr <= 0:
            continue
        favorable_high = sign * (
            path["high"].to_numpy(dtype=np.float64) - entry_price
        )
        favorable_low = sign * (
            path["low"].to_numpy(dtype=np.float64) - entry_price
        )
        favorable_close = sign * (
            path["close"].to_numpy(dtype=np.float64) - entry_price
        )
        if side == "SHORT":
            favorable_high, favorable_low = favorable_low, favorable_high
        peak = np.maximum.accumulate(favorable_high)
        mae = np.maximum.accumulate(-favorable_low)
        peak_indices = np.maximum.accumulate(
            np.where(favorable_high >= peak - 1e-15, np.arange(maximum_bars), 0)
        )
        episode_id = stable_episode_id(symbol, side, entry_timestamp)
        episodes.append({
            "schema_version": SCHEMA_VERSION,
            "position_episode_id": episode_id,
            "outcome_source": "SIMULATED",
            "partition": str(candidate.partition),
            "symbol": symbol,
            "side": side,
            "candidate_timestamp_ms": int(candidate.timestamp_ms),
            "entry_timestamp_ms": entry_timestamp,
            "actual_entry": math.nan,
            "actual_exit": math.nan,
            "actual_gross_pnl": math.nan,
            "actual_net_pnl": math.nan,
            "simulated_entry": entry_price,
            "simulated_exit": float(path.iloc[-1].close),
            "simulated_gross_pnl": float(favorable_close[-1] / entry_price),
            "simulated_net_pnl": float(
                favorable_close[-1] / entry_price
                - float(config["economics"]["base_round_trip_cost_bps"]) / 10_000.0
            ),
            "entry_atr": entry_atr,
            "peak_mfe_atr": float(peak.max() / entry_atr),
            "mae_atr": float(mae.max() / entry_atr),
            "path_open": path["open"].astype(float).tolist(),
            "path_high": path["high"].astype(float).tolist(),
            "path_low": path["low"].astype(float).tolist(),
            "path_close": path["close"].astype(float).tolist(),
            "path_atr": path["atr"].astype(float).tolist(),
        })
        gate_reached = peak / entry_atr >= minimum_gate
        for index in np.flatnonzero(gate_reached):
            row = path.iloc[index]
            current_peak = float(peak[index])
            current = float(favorable_close[index])
            giveback = max(0.0, current_peak - current)
            velocity = sign * path["velocity_atr_1"].iloc[max(0, index - 2):index + 1].to_numpy(dtype=float)
            acceleration = sign * path["acceleration_atr"].iloc[max(0, index - 1):index + 1].to_numpy(dtype=float)
            directional_flow = sign * path["taker_imbalance"].iloc[max(0, index - 2):index + 1].to_numpy(dtype=float)
            is_opposite = sign * (float(row.close) - float(row.open)) < 0.0
            favorable_structure = (
                bool(row.higher_high and row.higher_low)
                if side == "LONG" else bool(row.lower_low and row.lower_high)
            )
            deterioration = (
                bool(row.lower_high or row.lower_low)
                if side == "LONG" else bool(row.higher_low or row.higher_high)
            )
            decision: dict[str, object] = {
                "schema_version": SCHEMA_VERSION,
                "position_episode_id": episode_id,
                "outcome_source": "SIMULATED",
                "partition": str(candidate.partition),
                "symbol": symbol,
                "side": side,
                "entry_timestamp_ms": entry_timestamp,
                "evaluation_timestamp_ms": int(row.close_time_ms),
                "bar_index": int(index + 1),
                "entry_price": entry_price,
                "current_price": float(row.close),
                "entry_atr": entry_atr,
                "peak_mfe_atr": current_peak / entry_atr,
                "current_return_atr": current / entry_atr,
                "giveback_atr": giveback / entry_atr,
                "giveback_ratio": giveback / current_peak if current_peak > 0 else 0.0,
                "bars_since_peak": int(index - peak_indices[index]),
                "new_extreme_last_1": bool(peak_indices[index] == index),
                "new_extreme_last_2": bool(peak_indices[index] >= index - 1),
                "new_extreme_last_3": bool(peak_indices[index] >= index - 2),
                "distance_from_peak_atr": giveback / entry_atr,
                "directional_velocity_1": float(velocity[-1]),
                "directional_velocity_2": float(velocity[-2:].mean()),
                "directional_velocity_3": float(velocity.mean()),
                "velocity_decay": float(velocity[-1] - velocity[-2]) if len(velocity) >= 2 else 0.0,
                "mean_velocity_last_2": float(velocity[-2:].mean()),
                "mean_velocity_last_3": float(velocity.mean()),
                "velocity_slope": float(velocity[-1] - velocity[0]) / max(1, len(velocity) - 1),
                "directional_acceleration": float(acceleration[-1]),
                "acceleration_slope": float(acceleration[-1] - acceleration[0]) if len(acceleration) >= 2 else 0.0,
                "volume_ratio_20": float(row.volume_ratio_20),
                "volume_z_20": float(row.volume_z_20),
                "volume_slope": float(row.volume_slope),
                "volume_decay": float(row.volume_decay),
                "directional_taker_imbalance": float(directional_flow[-1]),
                "taker_imbalance_slope": float(directional_flow[-1] - directional_flow[0]) / max(1, len(directional_flow) - 1),
                "taker_imbalance_decay": float(directional_flow[-1] - directional_flow[-2]) if len(directional_flow) >= 2 else 0.0,
                "directional_delta_velocity": sign * float(row.delta_velocity),
                "directional_delta_acceleration": sign * float(row.delta_acceleration),
                "opposite_body_ratio": float(row.body_ratio) if is_opposite else 0.0,
                "opposite_volume_ratio": float(row.volume_ratio_20) if is_opposite else 0.0,
                "opposite_clv": float(1.0 - row.clv if side == "LONG" else row.clv) if is_opposite else 0.0,
                "retracement_fraction": giveback / current_peak if current_peak > 0 else 0.0,
                "favorable_structure": float(favorable_structure),
                "structure_deterioration": float(deterioration),
                "directional_price_vs_ma7_atr": sign * float(row.price_vs_ma_7_atr),
                "directional_price_vs_ma25_atr": sign * float(row.price_vs_ma_25_atr),
                "directional_ma7_slope_atr": sign * float(row.ma_7_slope_atr),
                "directional_ma25_slope_atr": sign * float(row.ma_25_slope_atr),
                "ma7_slope_decay": sign * float(row.ma7_slope_decay),
                "directional_rsi_extension": float(row.rsi_6 if side == "LONG" else 100.0 - row.rsi_6),
                "rsi_slope": sign * float(row.rsi_slope),
                "atr_percentile": float(row.atr_percentile),
                "atr_slope": float(row.atr_slope),
                "realized_volatility_12": float(row.realized_volatility_12),
                "directional_15m_return": sign * float(row.context_15m_return_1),
                "directional_15m_ma25_slope": sign * float(row.context_15m_ma_25_slope_atr),
                "directional_btc_5m_return": sign * float(row.btc_5m_return_1),
                "directional_btc_15m_return": sign * float(row.btc_15m_return_1),
                "directional_btc_taker_imbalance": sign * float(row.btc_5m_taker_imbalance),
                "btc_opposes_position": float(sign * float(row.btc_5m_return_1) < 0.0),
                "gate_025": bool(current_peak / entry_atr >= 0.25),
                "gate_050": bool(current_peak / entry_atr >= 0.50),
                "gate_075": bool(current_peak / entry_atr >= 0.75),
                "gate_100": bool(current_peak / entry_atr >= 1.00),
                "volume_over_4": bool(float(row.volume_ratio_20) > 4.0),
            }
            decision.update(_future_targets(
                favorable_high, favorable_low, favorable_close,
                index=index, peak_at_index=current_peak, atr=entry_atr,
            ))
            decisions.append(decision)
    return pd.DataFrame(episodes), pd.DataFrame(decisions)


@dataclass(frozen=True)
class PolicyResult:
    exit_bar: int
    exit_price: float
    gross_return: float
    net_return: float
    peak_mfe: float
    mae: float
    profit_capture_ratio: float
    final_giveback: float
    exit_reason: str


def simulate_exit_at_bar(
    episode: Mapping[str, object],
    *,
    requested_exit_bar: int | None,
    cost_bps: float,
    reason: str,
) -> PolicyResult:
    """Exit at a requested close while preserving the common hard stop."""

    side = str(episode["side"])
    sign = 1.0 if side == "LONG" else -1.0
    entry = float(episode["simulated_entry"])
    highs = np.asarray(episode["path_high"], dtype=float)
    lows = np.asarray(episode["path_low"], dtype=float)
    closes = np.asarray(episode["path_close"], dtype=float)
    favorable = highs - entry if side == "LONG" else entry - lows
    adverse = entry - lows if side == "LONG" else highs - entry
    maximum_index = len(closes) - 1 if requested_exit_bar is None else min(
        len(closes) - 1, max(0, int(requested_exit_bar) - 1)
    )
    stop_hits = np.flatnonzero(adverse[:maximum_index + 1] >= 0.02 * entry)
    if len(stop_hits):
        exit_index = int(stop_hits[0])
        exit_price = entry - sign * 0.02 * entry
        exit_reason = "COMMON_HARD_STOP"
    else:
        exit_index = maximum_index
        exit_price = float(closes[exit_index])
        exit_reason = reason if requested_exit_bar is not None else "BOUNDED_HOLD"
    gross = sign * (exit_price - entry) / entry
    available = max(0.0, float(favorable.max(initial=0.0)) / entry)
    return PolicyResult(
        exit_bar=exit_index + 1,
        exit_price=exit_price,
        gross_return=gross,
        net_return=gross - cost_bps / 10_000.0,
        peak_mfe=available,
        mae=float(adverse[:exit_index + 1].max(initial=0.0) / entry),
        profit_capture_ratio=max(0.0, gross) / available if available > 0 else 0.0,
        final_giveback=max(0.0, available - gross),
        exit_reason=exit_reason,
    )


def simulate_policy(
    episode: Mapping[str, object],
    *,
    policy: str,
    parameter: float,
    gate_atr: float,
    cost_bps: float,
) -> PolicyResult:
    """Simulate one deterministic exit policy with adverse-first bar ordering."""

    side = str(episode["side"])
    sign = 1.0 if side == "LONG" else -1.0
    entry = float(episode["simulated_entry"])
    atr0 = float(episode["entry_atr"])
    highs = np.asarray(episode["path_high"], dtype=float)
    lows = np.asarray(episode["path_low"], dtype=float)
    closes = np.asarray(episode["path_close"], dtype=float)
    atrs = np.asarray(episode["path_atr"], dtype=float)
    favorable = highs - entry if side == "LONG" else entry - lows
    adverse = entry - lows if side == "LONG" else highs - entry
    peak = np.maximum.accumulate(favorable)
    gate_hits = np.flatnonzero(peak >= gate_atr * atr0)
    gate_index = int(gate_hits[0]) if len(gate_hits) else len(closes) - 1
    exit_bar = len(closes) - 1
    exit_price = float(closes[-1])
    reason = "BOUNDED_HOLD"
    hard_stop_distance = 0.02 * entry
    trail_peak = 0.0
    be_active = False
    for index in range(len(closes)):
        if adverse[index] >= hard_stop_distance:
            exit_bar = index
            exit_price = entry - sign * hard_stop_distance
            reason = "COMMON_HARD_STOP"
            break
        trail_peak = max(trail_peak, float(favorable[index]))
        if policy == "FIXED_TP" and favorable[index] >= parameter * atr0:
            exit_bar = index
            exit_price = entry + sign * parameter * atr0
            reason = "FIXED_TP"
            break
        if index < gate_index:
            continue
        if policy == "FIXED_TRAILING" and trail_peak - (-sign * (entry - closes[index])) >= parameter * atr0:
            exit_bar = index
            exit_price = float(closes[index])
            reason = "FIXED_TRAILING"
            break
        if policy == "PERCENT_GIVEBACK" and trail_peak > 0:
            current = sign * (closes[index] - entry)
            if (trail_peak - current) / trail_peak >= parameter:
                exit_bar = index
                exit_price = float(closes[index])
                reason = "PERCENT_GIVEBACK"
                break
        if policy == "TIME_EXIT" and index - gate_index >= int(parameter):
            exit_bar = index
            exit_price = float(closes[index])
            reason = "TIME_EXIT"
            break
        if policy == "CURRENT_ATR_TRAILING":
            activation = 0.15 / 20.0 * entry
            if trail_peak >= activation:
                trigger = trail_peak - 1.5 * float(atrs[index])
                current = sign * (closes[index] - entry)
                if current <= trigger:
                    exit_bar = index
                    exit_price = float(closes[index])
                    reason = "CURRENT_ATR_TRAILING"
                    break
        if policy == "CURRENT_BE_PROTECTION":
            be_active = be_active or trail_peak >= 0.08 / 20.0 * entry
            if be_active:
                locked = max(0.01 / 20.0, trail_peak / entry - 0.05 / 20.0)
                current = sign * (closes[index] - entry) / entry
                if current <= locked:
                    exit_bar = index
                    exit_price = float(closes[index])
                    reason = "CURRENT_BE_PROTECTION_UPPER_BOUND"
                    break
    gross = sign * (exit_price - entry) / entry
    net = gross - cost_bps / 10_000.0
    available = max(0.0, float(peak.max(initial=0.0)) / entry)
    capture = max(0.0, gross) / available if available > 0 else 0.0
    return PolicyResult(
        exit_bar=exit_bar + 1,
        exit_price=exit_price,
        gross_return=gross,
        net_return=net,
        peak_mfe=available,
        mae=float(adverse[:exit_bar + 1].max(initial=0.0) / entry),
        profit_capture_ratio=capture,
        final_giveback=max(0.0, available - gross),
        exit_reason=reason,
    )
