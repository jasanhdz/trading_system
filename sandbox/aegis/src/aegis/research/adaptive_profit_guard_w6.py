"""Causal guard replay and statistics for adaptive profit guard W6."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


FORBIDDEN_FEATURE_PREFIXES = ("target_", "actual_future_", "future_")


@dataclass(frozen=True)
class GuardResult:
    exit_bar: int
    exit_price: float
    gross_return: float
    net_return: float
    peak_mfe: float
    mae: float
    profit_capture_ratio: float
    final_giveback: float
    exit_reason: str


def validate_feature_contract(columns: Sequence[str]) -> None:
    if not columns or len(columns) != len(set(columns)):
        raise ValueError("AEGIS_W6_FEATURE_CONTRACT_INVALID")
    forbidden = [name for name in columns if name.startswith(FORBIDDEN_FEATURE_PREFIXES)]
    if forbidden:
        raise ValueError(f"AEGIS_W6_FUTURE_FEATURE_PROHIBITED:{','.join(forbidden)}")


def trailing_activation_atr(entry: float, entry_atr: float, activation_roe: float, leverage: float) -> float:
    if entry <= 0 or entry_atr <= 0 or leverage <= 0:
        raise ValueError("AEGIS_W6_ACTIVATION_INPUT_INVALID")
    return activation_roe / leverage * entry / entry_atr


def _result(
    *, side: str, entry: float, exit_price: float, exit_bar: int,
    favorable: np.ndarray, adverse: np.ndarray, cost_bps: float, reason: str,
) -> GuardResult:
    sign = 1.0 if side == "LONG" else -1.0
    gross = sign * (exit_price - entry) / entry
    available = max(0.0, float(np.max(favorable[: exit_bar + 1], initial=0.0)) / entry)
    return GuardResult(
        exit_bar=exit_bar + 1,
        exit_price=float(exit_price),
        gross_return=float(gross),
        net_return=float(gross - cost_bps / 10_000.0),
        peak_mfe=available,
        mae=float(np.max(adverse[: exit_bar + 1], initial=0.0) / entry),
        profit_capture_ratio=max(0.0, float(gross)) / available if available > 0 else 0.0,
        final_giveback=max(0.0, available - float(gross)),
        exit_reason=reason,
    )


def simulate_guard(
    episode: Mapping[str, Any], *, atr_multiplier: float, cost_bps: float,
    leverage: float = 20.0, be_trigger_roe: float = 0.08,
    be_offset_fraction: float = 0.003, activation_roe: float = 0.15,
    hard_stop_roe: float = -0.40,
) -> GuardResult:
    """Replay the current guardian with a frozen ATR multiplier.

    Stops present at the start of a bar are checked adverse-first. A newly
    calculated stop becomes effective on the next bar, preventing intrabar
    lookahead when only closed 5m OHLC is available.
    """

    side = str(episode["side"])
    if side not in {"LONG", "SHORT"} or atr_multiplier <= 0:
        raise ValueError("AEGIS_W6_GUARD_INPUT_INVALID")
    sign = 1.0 if side == "LONG" else -1.0
    entry = float(episode["simulated_entry"])
    highs = np.asarray(episode["path_high"], dtype=float)
    lows = np.asarray(episode["path_low"], dtype=float)
    closes = np.asarray(episode["path_close"], dtype=float)
    atrs = np.asarray(episode["path_atr"], dtype=float)
    if entry <= 0 or not (len(highs) == len(lows) == len(closes) == len(atrs)) or not len(closes):
        raise ValueError("AEGIS_W6_EPISODE_PATH_INVALID")
    favorable = highs - entry if side == "LONG" else entry - lows
    adverse = entry - lows if side == "LONG" else highs - entry
    hard_stop_distance = abs(hard_stop_roe) / leverage * entry
    active_stop = entry - sign * hard_stop_distance
    best_price = entry
    peak_roe = 0.0

    for index in range(len(closes)):
        stop_hit = lows[index] <= active_stop if side == "LONG" else highs[index] >= active_stop
        if stop_hit:
            reason = "COMMON_HARD_STOP" if abs(active_stop - entry) >= hard_stop_distance - 1e-12 else "PROFIT_GUARD"
            return _result(
                side=side, entry=entry, exit_price=active_stop, exit_bar=index,
                favorable=favorable, adverse=adverse, cost_bps=cost_bps, reason=reason,
            )

        best_price = max(best_price, float(highs[index])) if side == "LONG" else min(best_price, float(lows[index]))
        peak_roe = max(peak_roe, sign * (best_price - entry) / entry * leverage)
        candidate = active_stop
        if peak_roe >= activation_roe and np.isfinite(atrs[index]) and atrs[index] > 0:
            trailing = best_price - sign * atr_multiplier * float(atrs[index])
            candidate = max(candidate, trailing) if side == "LONG" else min(candidate, trailing)
        elif peak_roe >= be_trigger_roe:
            break_even = entry * (1 + sign * be_offset_fraction)
            candidate = max(candidate, break_even) if side == "LONG" else min(candidate, break_even)
        active_stop = candidate

    return _result(
        side=side, entry=entry, exit_price=float(closes[-1]), exit_bar=len(closes) - 1,
        favorable=favorable, adverse=adverse, cost_bps=cost_bps, reason="BOUNDED_HOLD",
    )


def simulate_simple_baseline(
    episode: Mapping[str, Any], *, policy: str, parameter: float,
    cost_bps: float, gate_atr: float = 0.25, leverage: float = 20.0,
    hard_stop_roe: float = -0.40,
) -> GuardResult:
    """Replay frozen simple baselines using adverse-first closed-bar ordering."""

    side = str(episode["side"])
    sign = 1.0 if side == "LONG" else -1.0
    entry = float(episode["simulated_entry"])
    atr0 = float(episode["entry_atr"])
    highs = np.asarray(episode["path_high"], dtype=float)
    lows = np.asarray(episode["path_low"], dtype=float)
    closes = np.asarray(episode["path_close"], dtype=float)
    favorable = highs - entry if side == "LONG" else entry - lows
    adverse = entry - lows if side == "LONG" else highs - entry
    peak = np.maximum.accumulate(favorable)
    gate_hits = np.flatnonzero(peak >= gate_atr * atr0)
    gate_index = int(gate_hits[0]) if len(gate_hits) else len(closes)
    hard_stop_distance = abs(hard_stop_roe) / leverage * entry

    for index in range(len(closes)):
        if adverse[index] >= hard_stop_distance:
            return _result(
                side=side, entry=entry, exit_price=entry - sign * hard_stop_distance,
                exit_bar=index, favorable=favorable, adverse=adverse,
                cost_bps=cost_bps, reason="COMMON_HARD_STOP",
            )
        if policy == "FIXED_TP" and favorable[index] >= parameter * atr0:
            return _result(
                side=side, entry=entry, exit_price=entry + sign * parameter * atr0,
                exit_bar=index, favorable=favorable, adverse=adverse,
                cost_bps=cost_bps, reason="FIXED_TP",
            )
        if index < gate_index:
            continue
        current = sign * (float(closes[index]) - entry)
        trail_peak = float(peak[index])
        if policy == "FIXED_TRAILING" and trail_peak - current >= parameter * atr0:
            return _result(
                side=side, entry=entry, exit_price=float(closes[index]), exit_bar=index,
                favorable=favorable, adverse=adverse, cost_bps=cost_bps,
                reason="FIXED_TRAILING",
            )
        if policy == "PERCENT_GIVEBACK" and trail_peak > 0 and (trail_peak - current) / trail_peak >= parameter:
            return _result(
                side=side, entry=entry, exit_price=float(closes[index]), exit_bar=index,
                favorable=favorable, adverse=adverse, cost_bps=cost_bps,
                reason="PERCENT_GIVEBACK",
            )
        if policy == "TIME_EXIT" and index - gate_index >= int(parameter):
            return _result(
                side=side, entry=entry, exit_price=float(closes[index]), exit_bar=index,
                favorable=favorable, adverse=adverse, cost_bps=cost_bps,
                reason="TIME_EXIT",
            )
    return _result(
        side=side, entry=entry, exit_price=float(closes[-1]), exit_bar=len(closes) - 1,
        favorable=favorable, adverse=adverse, cost_bps=cost_bps, reason="BOUNDED_HOLD",
    )


def choose_best_action(results: Mapping[str, GuardResult]) -> str:
    preference = {"NORMAL": 2, "DEFENSIVE": 1, "EXPANSION": 0}
    return max(results, key=lambda name: (results[name].net_return, preference.get(name, -1)))


def paired_day_bootstrap(
    frame: pd.DataFrame, *, repetitions: int, seed: int,
    value_column: str = "improvement_bps",
) -> tuple[list[float], float]:
    daily = frame.groupby("utc_day", observed=True)[value_column].mean().to_numpy(float)
    if not len(daily):
        return [float("nan"), float("nan")], float("nan")
    rng = np.random.default_rng(seed)
    samples = rng.choice(daily, size=(repetitions, len(daily)), replace=True).mean(axis=1)
    return [float(x) for x in np.quantile(samples, [0.025, 0.975])], float((samples <= 0).mean())


def policy_summary(frame: pd.DataFrame, return_column: str = "net_return_bps") -> dict[str, float | int]:
    values = frame[return_column].to_numpy(float)
    positive = values[values > 0].sum()
    negative = -values[values < 0].sum()
    giveback = frame["final_giveback_bps"].to_numpy(float)
    return {
        "episodes": int(len(frame)),
        "net_expectancy_bps": float(values.mean()) if len(values) else 0.0,
        "profit_factor": float(positive / negative) if negative else 1_000_000_000.0,
        "median_profit_capture_ratio": float(frame["profit_capture_ratio"].median()),
        "median_peak_mfe_bps": float(frame["peak_mfe_bps"].median()),
        "median_mae_bps": float(frame["mae_bps"].median()),
        "median_giveback_bps": float(np.median(giveback)),
        "median_early_exit_regret_bps": float(frame["early_exit_regret_bps"].median()),
        "median_hold_too_long_regret_bps": float(frame["hold_too_long_regret_bps"].median()),
        "p95_giveback_bps": float(np.quantile(giveback, 0.95)),
        "p99_giveback_bps": float(np.quantile(giveback, 0.99)),
    }


def benjamini_hochberg(pvalues: Mapping[str, float], alpha: float = 0.05) -> dict[str, bool]:
    ordered = sorted(pvalues.items(), key=lambda item: item[1])
    accepted = 0
    for rank, (_, value) in enumerate(ordered, start=1):
        if value <= alpha * rank / max(len(ordered), 1):
            accepted = rank
    accepted_names = {name for name, _ in ordered[:accepted]}
    return {name: name in accepted_names for name in pvalues}
