"""Deterministic mechanical exit replay over the frozen E3 SHORT entries."""

from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import subprocess
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import yaml

from aegis.data import CanonicalBar, CanonicalSeriesSource, DataPurpose
from aegis.training.econ import CostScenario
from aegis.utils import Sha256HashProvider, canonical_json, sha256_file
from scripts.diagnostics.exit_excursion_d1a.experiment import (
    FrozenEntry,
    atomic_json,
    atomic_parquet,
    load_frozen_entries,
)


class E4AError(RuntimeError):
    """Base fail-closed E4A error."""


class PreregistrationError(E4AError):
    pass


class EntryDriftError(E4AError):
    pass


class DataCoverageError(E4AError):
    pass


class PolicySemanticsError(E4AError):
    pass


class NondeterministicError(E4AError):
    pass


@dataclass(frozen=True)
class PolicyParameters:
    leverage: float
    stop_roe: float
    take_profit_roe: float
    break_even_activation_roe: float
    break_even_offset_fraction: float
    trailing_activation_roe: float
    atr_multiplier: float
    callback_fraction: float
    protection_min_peak_roe: float
    protection_giveback_roe: float
    protection_min_roe: float
    immediate_buffer_fraction: float
    timeout_bars: int


@dataclass(frozen=True)
class TradeTrajectory:
    entry: FrozenEntry
    pre_entry: tuple[CanonicalBar, ...]
    path: tuple[CanonicalBar, ...]
    initial_atr: float


POLICY_FEATURES: Mapping[str, frozenset[str]] = {
    "P0": frozenset({"h12"}),
    "P1": frozenset({"timeout"}),
    "P2": frozenset({"bracket", "timeout"}),
    "P3": frozenset({"bracket", "break_even", "timeout"}),
    "P4": frozenset({"bracket", "break_even", "trailing", "callback", "protection", "timeout"}),
}
INTRABAR_MODES = ("CONSERVATIVE", "OPTIMISTIC")
SCIENTIFIC_FILES = (
    "entry_source_manifest.json",
    "trajectory_manifest.json",
    "atr_manifest.json",
    "trajectories.parquet",
    "atr.parquet",
    "policy_definition.json",
    "policy_events.parquet",
    "trades.parquet",
    "policy_results.json",
    "fold_results.json",
    "econ_results.json",
    "intrabar_sensitivity.json",
    "outlier_sensitivity.json",
    "symbol_concentration.json",
    "diagnostic_summary.json",
    "diagnostic_summary.md",
    "scientific_aggregate.json",
)


def _utc(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise ValueError("timezone required")
    return parsed.astimezone(timezone.utc)


def _iso(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _atomic_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def short_roe(entry_price: float, current_price: float, leverage: float) -> float:
    if entry_price <= 0 or leverage <= 0:
        raise ValueError("entry price and leverage must be positive")
    return (entry_price - current_price) / entry_price * leverage


def initial_stop_price(entry_price: float, stop_roe: float, leverage: float) -> float:
    return entry_price * (1.0 + abs(stop_roe) / leverage)


def take_profit_price(entry_price: float, take_profit_roe: float, leverage: float) -> float:
    return entry_price * (1.0 - take_profit_roe / leverage)


def break_even_price(entry_price: float, offset_fraction: float) -> float:
    return entry_price * (1.0 - offset_fraction)


def true_range(bar: CanonicalBar, previous_close: float) -> float:
    return max(bar.high - bar.low, abs(bar.high - previous_close), abs(bar.low - previous_close))


def initial_atr(history: Sequence[CanonicalBar], period: int = 14) -> float:
    if len(history) != period + 1:
        raise ValueError("initial ATR requires period + 1 completed bars")
    values = [true_range(history[index], history[index - 1].close) for index in range(1, len(history))]
    return float(np.mean(np.asarray(values, dtype=np.float64)))


def update_atr(previous_atr: float, completed_true_range: float, period: int = 14) -> float:
    if previous_atr <= 0 or completed_true_range < 0:
        raise ValueError("ATR values must be valid")
    return ((previous_atr * (period - 1)) + completed_true_range) / period


def callback_trigger_roe(observed_peak_roe: float, callback_fraction: float) -> float:
    return observed_peak_roe * (1.0 - callback_fraction)


def protection_target_roe(peak_roe: float, giveback_roe: float, minimum_roe: float) -> float:
    return max(minimum_roe, peak_roe - giveback_roe)


def _parameters(config: Mapping[str, Any]) -> PolicyParameters:
    values = config["policy_parameters"]
    return PolicyParameters(
        leverage=float(config["fixed_leverage"]["value"]),
        stop_roe=float(values["stop_roe"]),
        take_profit_roe=float(values["take_profit_roe"]),
        break_even_activation_roe=float(values["break_even_activation_roe"]),
        break_even_offset_fraction=float(values["break_even_price_offset_fraction"]),
        trailing_activation_roe=float(values["trailing_activation_peak_roe"]),
        atr_multiplier=float(config["atr"]["multiplier"]),
        callback_fraction=float(values["callback_fraction_of_peak_roe"]),
        protection_min_peak_roe=float(values["profit_protection_min_peak_roe"]),
        protection_giveback_roe=float(values["profit_protection_giveback_roe"]),
        protection_min_roe=float(values["profit_protection_min_protected_roe"]),
        immediate_buffer_fraction=float(values["profit_protection_immediate_trigger_buffer_fraction"]),
        timeout_bars=int(values["timeout_bars"]),
    )


def load_preregistration(path: Path, repository: Path) -> Mapping[str, Any]:
    binding_path = repository / "reports/governance/e4a_mechanical_exit/preregistration_binding.json"
    binding = json.loads(binding_path.read_text(encoding="utf-8"))
    if sha256_file(path) != binding["preregistration"]["physical_sha256"]:
        raise PreregistrationError("E4A preregistration physical hash mismatch")
    owner_path = repository / binding["owner_decision"]["path"]
    if sha256_file(owner_path) != binding["owner_decision"]["physical_sha256"]:
        raise PreregistrationError("E4A owner decision hash mismatch")
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(config, Mapping):
        raise PreregistrationError("E4A preregistration must be a mapping")
    required = {
        "schema_version", "experiment_id", "classification", "entry_source", "canonical_source",
        "trajectory", "fixed_leverage", "atr", "policy_parameters", "resolved_historical_formulas",
        "state_machine", "intrabar", "policies", "cost_scenarios", "metrics",
        "e4b_justification_criteria", "determinism", "failure_states", "forbidden",
    }
    if not required <= set(config):
        raise PreregistrationError(f"missing preregistration fields: {sorted(required - set(config))}")
    if (
        config["experiment_id"] != "E4A_MECHANICAL_EXIT"
        or config["classification"] != "PREREGISTERED_DEV_EXPERIMENT"
        or float(config["fixed_leverage"]["value"]) != 20.0
        or int(config["trajectory"]["maximum_horizon_bars"]) != 96
        or tuple(config["policies"]) != tuple(POLICY_FEATURES)
        or config["intrabar"]["primary"] != "CONSERVATIVE"
    ):
        raise PreregistrationError("E4A frozen protocol mismatch")
    return config


def build_trajectories(
    entries: Sequence[FrozenEntry],
    prices: Mapping[str, Sequence[CanonicalBar]],
    *,
    horizon_bars: int,
    pre_entry_bars: int,
    dev_boundary: datetime,
) -> tuple[tuple[TradeTrajectory, ...], list[dict[str, Any]], list[dict[str, Any]], Mapping[str, Any]]:
    indices = {symbol: {bar.timestamp: index for index, bar in enumerate(rows)} for symbol, rows in prices.items()}
    complete: list[TradeTrajectory] = []
    trajectory_rows: list[dict[str, Any]] = []
    atr_rows: list[dict[str, Any]] = []
    invalid: list[dict[str, str]] = []
    for entry in entries:
        rows = prices.get(entry.symbol)
        index = indices.get(entry.symbol, {}).get(entry.signal_timestamp)
        reason = None
        if rows is None or index is None:
            reason = "signal_bar_missing"
        elif index - pre_entry_bars + 1 < 0:
            reason = "pre_entry_history_missing"
        elif index + horizon_bars >= len(rows):
            reason = "post_entry_history_missing"
        if reason:
            invalid.append({"trade_id": entry.trade_id, "reason": reason})
            continue
        history = tuple(rows[index - pre_entry_bars + 1:index + 1])
        path = tuple(rows[index + 1:index + horizon_bars + 1])
        combined = history + path
        if len(history) != pre_entry_bars or len(path) != horizon_bars:
            invalid.append({"trade_id": entry.trade_id, "reason": "trajectory_length_mismatch"})
            continue
        if path[0].timestamp != entry.entry_timestamp or abs(path[0].open - entry.entry_price) > 1e-12:
            raise EntryDriftError(f"entry alignment drift for {entry.trade_id}")
        if path[-1].timestamp > dev_boundary:
            raise DataCoverageError("E4A path crosses dev boundary")
        if any(right.timestamp - left.timestamp != timedelta(minutes=5) for left, right in zip(combined, combined[1:])):
            invalid.append({"trade_id": entry.trade_id, "reason": "five_minute_gap"})
            continue
        atr = initial_atr(history)
        if not math.isfinite(atr) or atr <= 0:
            invalid.append({"trade_id": entry.trade_id, "reason": "invalid_initial_atr"})
            continue
        complete.append(TradeTrajectory(entry, history, path, atr))
        previous_close = history[-1].close
        current_atr = atr
        for bar_index, bar in enumerate(path, 1):
            tr = true_range(bar, previous_close)
            updated = update_atr(current_atr, tr)
            base = {
                "trade_id": entry.trade_id, "symbol": entry.symbol, "fold": entry.fold,
                "signal_timestamp": _iso(entry.signal_timestamp), "entry_timestamp": _iso(entry.entry_timestamp),
                "entry_price": entry.entry_price, "bar_index": bar_index, "bar_timestamp": _iso(bar.timestamp),
            }
            trajectory_rows.append({
                **base, "open": bar.open, "high": bar.high, "low": bar.low, "close": bar.close,
                "volume": bar.volume, "is_final": True, "gap_from_previous": False,
            })
            atr_rows.append({
                **base, "previous_close": previous_close, "true_range": tr,
                "atr_available_at_bar_open": current_atr, "atr_after_completed_bar": updated,
            })
            previous_close = bar.close
            current_atr = updated
    coverage = len(complete) / max(1, len(entries))
    quality = {
        "entry_count": len(entries), "complete_count": len(complete), "invalid_count": len(invalid),
        "complete_fraction": coverage, "minimum_fraction": 0.90, "invalid_entries": invalid,
        "trajectory_rows": len(trajectory_rows), "atr_rows": len(atr_rows),
        "horizon_bars": horizon_bars, "pre_entry_bars": pre_entry_bars,
    }
    if coverage < 0.90:
        raise DataCoverageError(canonical_json(quality))
    return tuple(complete), trajectory_rows, atr_rows, quality


def _event(
    trajectory: TradeTrajectory,
    policy: str,
    mode: str,
    bar_index: int,
    event_type: str,
    **details: Any,
) -> dict[str, Any]:
    return {
        "trade_id": trajectory.entry.trade_id,
        "symbol": trajectory.entry.symbol,
        "fold": trajectory.entry.fold,
        "policy": policy,
        "intrabar_mode": mode,
        "bar_index": bar_index,
        "event_type": event_type,
        "details_json": canonical_json(details),
    }


def simulate_policy(
    trajectory: TradeTrajectory,
    policy: str,
    intrabar_mode: str,
    parameters: PolicyParameters,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if policy not in POLICY_FEATURES or intrabar_mode not in INTRABAR_MODES:
        raise ValueError("unsupported policy or intrabar mode")
    features = POLICY_FEATURES[policy]
    entry = trajectory.entry
    path = trajectory.path
    horizon = 12 if "h12" in features else parameters.timeout_bars
    if len(path) < horizon:
        raise DataCoverageError(f"insufficient path for {entry.trade_id}")
    stop = initial_stop_price(entry.entry_price, parameters.stop_roe, parameters.leverage) if "bracket" in features else None
    stop_reason = "CLOSED_STOP"
    take_profit = take_profit_price(entry.entry_price, parameters.take_profit_roe, parameters.leverage) if "bracket" in features else None
    peak_roe = 0.0
    lowest_price = entry.entry_price
    current_atr = trajectory.initial_atr
    previous_close = trajectory.pre_entry[-1].close
    trailing_armed = False
    break_even_armed = False
    protection_armed = False
    ambiguous = False
    events: list[dict[str, Any]] = []
    exit_price = path[horizon - 1].close
    exit_bar = horizon
    exit_reason = "CLOSED_TIMEOUT"

    for bar_index, bar in enumerate(path[:horizon], 1):
        stop_hit = stop is not None and bar.high >= stop
        take_profit_hit = take_profit is not None and bar.low <= take_profit
        if stop_hit and take_profit_hit:
            ambiguous = True
            events.append(_event(trajectory, policy, intrabar_mode, bar_index, "INTRABAR_AMBIGUITY", stop=stop, take_profit=take_profit))
            if intrabar_mode == "CONSERVATIVE":
                exit_price, exit_reason = float(stop), stop_reason
            else:
                exit_price, exit_reason = float(take_profit), "CLOSED_TAKE_PROFIT"
            exit_bar = bar_index
            break
        if stop_hit:
            exit_price, exit_reason, exit_bar = float(stop), stop_reason, bar_index
            break
        if take_profit_hit:
            exit_price, exit_reason, exit_bar = float(take_profit), "CLOSED_TAKE_PROFIT", bar_index
            break

        old_peak = peak_roe
        lowest_price = min(lowest_price, bar.low)
        peak_roe = max(peak_roe, short_roe(entry.entry_price, lowest_price, parameters.leverage))
        current_roe = short_roe(entry.entry_price, bar.close, parameters.leverage)
        tr = true_range(bar, previous_close)
        updated_atr = update_atr(current_atr, tr)
        candidates: list[tuple[float, str, str]] = []

        if "protection" in features and peak_roe >= parameters.protection_min_peak_roe:
            if not protection_armed:
                events.append(_event(trajectory, policy, intrabar_mode, bar_index, "PROFIT_PROTECTION_ARMED", peak_roe=peak_roe))
            protection_armed = True
            protected_roe = protection_target_roe(
                peak_roe, parameters.protection_giveback_roe, parameters.protection_min_roe,
            )
            raw_price = entry.entry_price * (1.0 - protected_roe / parameters.leverage)
            safe_price = max(raw_price, bar.close * (1.0 + parameters.immediate_buffer_fraction))
            candidates.append((safe_price, "CLOSED_PROFIT_PROTECTION", "PROFIT_PROTECTION_STOP_UPDATED"))

        newly_trailing = False
        if "trailing" in features and peak_roe >= parameters.trailing_activation_roe:
            newly_trailing = not trailing_armed
            if newly_trailing:
                events.append(_event(trajectory, policy, intrabar_mode, bar_index, "TRAILING_ARMED", peak_roe=peak_roe))
            trailing_armed = True
            if updated_atr > 0:
                trailing_price = lowest_price + updated_atr * parameters.atr_multiplier
                candidates.append((trailing_price, "CLOSED_TRAILING", "TRAILING_STOP_UPDATED"))
            else:
                trigger = callback_trigger_roe(peak_roe, parameters.callback_fraction)
                callback_price = entry.entry_price * (1.0 - trigger / parameters.leverage)
                candidates.append((callback_price, "CLOSED_CALLBACK", "CALLBACK_STOP_UPDATED"))

        if "break_even" in features and peak_roe >= parameters.break_even_activation_roe:
            if not break_even_armed:
                events.append(_event(trajectory, policy, intrabar_mode, bar_index, "BREAK_EVEN_ARMED", peak_roe=peak_roe))
            break_even_armed = True
            candidate = break_even_price(entry.entry_price, parameters.break_even_offset_fraction)
            if candidate > bar.close * (1.0 + parameters.immediate_buffer_fraction):
                candidates.append((candidate, "CLOSED_BREAK_EVEN", "BREAK_EVEN_STOP_UPDATED"))
            else:
                events.append(_event(trajectory, policy, intrabar_mode, bar_index, "BREAK_EVEN_UPDATE_SKIPPED", reason="immediate_trigger_risk"))

        precedence = {"CLOSED_PROFIT_PROTECTION": 0, "CLOSED_TRAILING": 1, "CLOSED_CALLBACK": 1, "CLOSED_BREAK_EVEN": 2}
        candidates.sort(key=lambda item: (item[0], precedence[item[1]]))
        improved = next((item for item in candidates if stop is None or item[0] < stop), None)
        if improved is not None:
            candidate_price, candidate_reason, event_type = improved
            same_bar_reachable = bar.high >= candidate_price and candidate_price > bar.low
            newly_armed = (candidate_reason == "CLOSED_TRAILING" and newly_trailing) or old_peak < parameters.break_even_activation_roe <= peak_roe
            if same_bar_reachable and newly_armed:
                ambiguous = True
                events.append(_event(
                    trajectory, policy, intrabar_mode, bar_index, "INTRABAR_AMBIGUITY",
                    candidate_stop=candidate_price, candidate_reason=candidate_reason,
                ))
                if intrabar_mode == "OPTIMISTIC":
                    exit_price, exit_reason, exit_bar = candidate_price, candidate_reason, bar_index
                    break
            stop, stop_reason = candidate_price, candidate_reason
            events.append(_event(trajectory, policy, intrabar_mode, bar_index, event_type, stop=stop, peak_roe=peak_roe, atr=updated_atr))

        previous_close = bar.close
        current_atr = updated_atr
        if bar_index == horizon:
            exit_price, exit_reason, exit_bar = bar.close, "CLOSED_TIMEOUT", bar_index

    held_path = path[:exit_bar]
    mfe = max(0.0, max((entry.entry_price - bar.low) / entry.entry_price for bar in held_path))
    mae = max(0.0, max((bar.high - entry.entry_price) / entry.entry_price for bar in held_path))
    gross = (entry.entry_price - exit_price) / entry.entry_price
    capture = None if mfe == 0.0 else gross / mfe
    trade = {
        "trade_id": entry.trade_id, "symbol": entry.symbol, "fold": entry.fold,
        "signal_timestamp": _iso(entry.signal_timestamp), "entry_timestamp": _iso(entry.entry_timestamp),
        "entry_price": entry.entry_price, "policy": policy, "intrabar_mode": intrabar_mode,
        "exit_bar": exit_bar, "exit_timestamp": _iso(path[exit_bar - 1].timestamp), "exit_price": exit_price,
        "exit_reason": exit_reason, "gross_return": gross, "mfe": mfe, "mae": mae,
        "giveback": mfe - gross, "capture_ratio": capture, "peak_roe": peak_roe,
        "initial_atr": trajectory.initial_atr, "final_atr": current_atr,
        "intrabar_ambiguous": ambiguous,
    }
    events.append(_event(trajectory, policy, intrabar_mode, exit_bar, "EXIT", reason=exit_reason, price=exit_price))
    return trade, events


def _scenario_cost(scenario: CostScenario, holding_bars: int) -> float:
    return scenario.cost_fraction(holding_bars)


def _metrics(trades: Sequence[Mapping[str, Any]], scenario: CostScenario) -> Mapping[str, Any]:
    if not trades:
        return {
            "trades": 0, "gross_expectancy": 0.0, "net_expectancy": 0.0, "profit_factor": None,
            "win_rate": 0.0, "average_win": None, "average_loss": None, "payoff_ratio": None,
            "median_return": None, "p01_return": None, "p05_return": None, "cvar_05": None,
            "maximum_drawdown_non_compounded": 0.0, "average_duration_bars": None,
            "median_duration_bars": None, "cost_per_trade": 0.0, "capture_ratio_mean": None,
            "capture_ratio_median": None, "giveback_mean": None, "giveback_median": None,
            "exit_reason_counts": {}, "intrabar_ambiguity_count": 0,
        }
    gross = np.asarray([float(row["gross_return"]) for row in trades], dtype=np.float64)
    costs = np.asarray([_scenario_cost(scenario, int(row["exit_bar"])) for row in trades], dtype=np.float64)
    net = gross - costs
    gains = net[net > 0]; losses = net[net < 0]
    equity = np.cumsum(net); peaks = np.maximum.accumulate(np.concatenate(([0.0], equity)))
    cutoff = max(1, math.ceil(len(net) * 0.05))
    capture = np.asarray([float(row["capture_ratio"]) for row in trades if row["capture_ratio"] is not None], dtype=np.float64)
    giveback = np.asarray([float(row["giveback"]) for row in trades], dtype=np.float64)
    durations = np.asarray([int(row["exit_bar"]) for row in trades], dtype=np.float64)
    return {
        "trades": len(trades), "gross_expectancy": float(gross.mean()), "net_expectancy": float(net.mean()),
        "profit_factor": float(gains.sum() / -losses.sum()) if len(gains) and len(losses) else (0.0 if not len(gains) else None),
        "win_rate": float(np.mean(net > 0)), "average_win": float(gains.mean()) if len(gains) else None,
        "average_loss": float(losses.mean()) if len(losses) else None,
        "payoff_ratio": float(gains.mean() / -losses.mean()) if len(gains) and len(losses) else None,
        "median_return": float(np.median(net)), "p01_return": float(np.quantile(net, 0.01)),
        "p05_return": float(np.quantile(net, 0.05)), "cvar_05": float(np.mean(np.sort(net)[:cutoff])),
        "maximum_drawdown_non_compounded": float(np.max(peaks[1:] - equity)),
        "average_duration_bars": float(durations.mean()), "median_duration_bars": float(np.median(durations)),
        "cost_per_trade": float(costs.mean()),
        "capture_ratio_mean": float(capture.mean()) if len(capture) else None,
        "capture_ratio_median": float(np.median(capture)) if len(capture) else None,
        "giveback_mean": float(giveback.mean()), "giveback_median": float(np.median(giveback)),
        "exit_reason_counts": dict(sorted(Counter(str(row["exit_reason"]) for row in trades).items())),
        "intrabar_ambiguity_count": sum(bool(row["intrabar_ambiguous"]) for row in trades),
    }


def _paired_bootstrap_delta(
    candidate: Sequence[Mapping[str, Any]],
    baseline: Sequence[Mapping[str, Any]],
    scenario: CostScenario,
    *,
    repetitions: int,
    seed: int,
) -> tuple[float, float]:
    baseline_by_id = {str(row["trade_id"]): row for row in baseline}
    weeks: dict[tuple[int, int], list[float]] = {}
    for row in candidate:
        base = baseline_by_id[str(row["trade_id"])]
        delta = (
            float(row["gross_return"]) - _scenario_cost(scenario, int(row["exit_bar"]))
            - float(base["gross_return"]) + _scenario_cost(scenario, int(base["exit_bar"]))
        )
        iso = _utc(str(row["signal_timestamp"])).isocalendar()
        weeks.setdefault((iso.year, iso.week), []).append(delta)
    keys = sorted(weeks)
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(repetitions):
        sampled = rng.choice(len(keys), size=len(keys), replace=True)
        values.append(float(np.mean([value for index in sampled for value in weeks[keys[int(index)]]])))
    quantiles = np.quantile(np.asarray(values, dtype=np.float64), (0.05, 0.95))
    return float(quantiles[0]), float(quantiles[1])


def economic_results(
    trades: Sequence[Mapping[str, Any]],
    scenarios: Sequence[CostScenario],
    *,
    repetitions: int,
    seed: int,
) -> Mapping[str, Any]:
    output: dict[str, Any] = {"schema_version": "aegis-e4a-econ-results-v1", "intrabar_modes": {}}
    for mode in INTRABAR_MODES:
        mode_trades = [row for row in trades if row["intrabar_mode"] == mode]
        policies = {policy: [row for row in mode_trades if row["policy"] == policy] for policy in POLICY_FEATURES}
        mode_output: dict[str, Any] = {}
        for scenario in scenarios:
            scenario_output: dict[str, Any] = {}
            baseline = policies["P0"]
            baseline_metrics = _metrics(baseline, scenario)
            for policy, rows in policies.items():
                pooled = _metrics(rows, scenario)
                folds = {str(fold): _metrics([row for row in rows if int(row["fold"]) == fold], scenario) for fold in range(1, 5)}
                pooled["positive_folds"] = sum(item["net_expectancy"] > 0.0 for item in folds.values())
                pooled["folds_improved_vs_p0"] = sum(
                    folds[str(fold)]["net_expectancy"]
                    > _metrics([row for row in baseline if int(row["fold"]) == fold], scenario)["net_expectancy"]
                    for fold in range(1, 5)
                )
                comparison = {
                    "delta_gross_expectancy": pooled["gross_expectancy"] - baseline_metrics["gross_expectancy"],
                    "delta_net_expectancy": pooled["net_expectancy"] - baseline_metrics["net_expectancy"],
                    "delta_profit_factor": None if pooled["profit_factor"] is None or baseline_metrics["profit_factor"] is None else pooled["profit_factor"] - baseline_metrics["profit_factor"],
                    "delta_drawdown": pooled["maximum_drawdown_non_compounded"] - baseline_metrics["maximum_drawdown_non_compounded"],
                    "delta_capture_ratio": (
                        None if pooled["capture_ratio_mean"] is None or baseline_metrics["capture_ratio_mean"] is None
                        else pooled["capture_ratio_mean"] - baseline_metrics["capture_ratio_mean"]
                    ),
                    "bootstrap_ci90_delta_expectancy": list(_paired_bootstrap_delta(
                        rows, baseline, scenario, repetitions=repetitions, seed=seed,
                    )),
                }
                scenario_output[policy] = {"pooled": pooled, "by_fold": folds, "comparison_to_p0": comparison}
            mode_output[scenario.scenario_id] = scenario_output
        output["intrabar_modes"][mode] = mode_output
    return output


def _net_value(row: Mapping[str, Any], scenario: CostScenario) -> float:
    return float(row["gross_return"]) - _scenario_cost(scenario, int(row["exit_bar"]))


def outlier_and_concentration(
    trades: Sequence[Mapping[str, Any]], scenario: CostScenario,
) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
    primary = [row for row in trades if row["intrabar_mode"] == "CONSERVATIVE"]
    p0 = {str(row["trade_id"]): row for row in primary if row["policy"] == "P0"}
    p4 = [row for row in primary if row["policy"] == "P4"]
    remove = math.ceil(len(p4) * 0.01)
    ordered = sorted(p4, key=lambda row: (_net_value(row, scenario), str(row["trade_id"])), reverse=True)
    without_best = ordered[remove:]
    without_worst = ordered[:-remove]

    def comparison(rows: Sequence[Mapping[str, Any]]) -> Mapping[str, Any]:
        candidate = np.asarray([_net_value(row, scenario) for row in rows], dtype=np.float64)
        baseline = np.asarray([_net_value(p0[str(row["trade_id"])], scenario) for row in rows], dtype=np.float64)
        return {
            "trades": len(rows), "p4_net_expectancy": float(candidate.mean()),
            "p0_net_expectancy_same_entries": float(baseline.mean()),
            "delta": float((candidate - baseline).mean()),
        }

    p4_values = np.asarray([_net_value(row, scenario) for row in p4], dtype=np.float64)
    p0_values = np.asarray([_net_value(p0[str(row["trade_id"])], scenario) for row in p4], dtype=np.float64)
    p4_bounds = np.quantile(p4_values, (0.01, 0.99)); p0_bounds = np.quantile(p0_values, (0.01, 0.99))
    outliers = {
        "schema_version": "aegis-e4a-outlier-sensitivity-v1",
        "all_trades": comparison(p4), "excluding_best_one_percent": comparison(without_best),
        "excluding_worst_one_percent": comparison(without_worst), "removed_count": remove,
        "winsorized_p01_p99": {
            "trades": len(p4), "p4_net_expectancy": float(np.clip(p4_values, *p4_bounds).mean()),
            "p0_net_expectancy": float(np.clip(p0_values, *p0_bounds).mean()),
            "delta": float((np.clip(p4_values, *p4_bounds) - np.clip(p0_values, *p0_bounds)).mean()),
        },
        "by_fold": {
            str(fold): comparison([row for row in p4 if int(row["fold"]) == fold]) for fold in range(1, 5)
        },
        "by_symbol": {
            symbol: comparison([row for row in p4 if row["symbol"] == symbol])
            for symbol in sorted({str(row["symbol"]) for row in p4})
        },
    }
    positive = {symbol: sum(max(0.0, _net_value(row, scenario)) for row in p4 if row["symbol"] == symbol) for symbol in sorted({str(row["symbol"]) for row in p4})}
    total_positive = sum(positive.values())
    sorted_positive_trades = sorted((max(0.0, _net_value(row, scenario)) for row in p4), reverse=True)
    total = sum(sorted_positive_trades)
    concentration = {
        "schema_version": "aegis-e4a-symbol-concentration-v1",
        "positive_pnl_by_symbol": positive,
        "maximum_positive_pnl_symbol_share": max(positive.values()) / total_positive if total_positive else 0.0,
        "best_trade_positive_pnl_share": sorted_positive_trades[0] / total if total else 0.0,
        "best_one_percent_positive_pnl_share": sum(sorted_positive_trades[:remove]) / total if total else 0.0,
        "best_five_percent_positive_pnl_share": sum(sorted_positive_trades[:math.ceil(len(p4) * 0.05)]) / total if total else 0.0,
    }
    return outliers, concentration


def _classification_and_decision(
    econ: Mapping[str, Any], outliers: Mapping[str, Any], concentration: Mapping[str, Any], quality: Mapping[str, Any],
) -> tuple[str, str, list[Mapping[str, Any]]]:
    base = econ["intrabar_modes"]["CONSERVATIVE"]["B_BASE"]
    p0 = base["P0"]["pooled"]; p1 = base["P1"]["pooled"]; p4 = base["P4"]["pooled"]
    comparison = base["P4"]["comparison_to_p0"]
    finite = all(math.isfinite(float(value)) for value in (p4["gross_expectancy"], p4["net_expectancy"], comparison["delta_net_expectancy"]))
    if quality["complete_fraction"] < 1.0 or not finite:
        classification = "MECHANICAL_EXIT_DIAGNOSTIC_INCONCLUSIVE"
    elif p4["net_expectancy"] > 0.0 and p4["net_expectancy"] > p0["net_expectancy"]:
        classification = "MECHANICAL_EXIT_ECONOMICALLY_PROMISING"
    elif p4["gross_expectancy"] > p0["gross_expectancy"]:
        classification = "MECHANICAL_EXIT_GROSS_ONLY"
    else:
        classification = "MECHANICAL_EXIT_NO_VALUE"
    checks = [
        {"name": "net_expectancy_above_zero", "passed": p4["net_expectancy"] > 0.0, "actual": p4["net_expectancy"]},
        {"name": "profit_factor_at_least_1_05", "passed": p4["profit_factor"] is not None and p4["profit_factor"] >= 1.05, "actual": p4["profit_factor"]},
        {"name": "net_expectancy_above_p0", "passed": comparison["delta_net_expectancy"] > 0.0, "actual": comparison["delta_net_expectancy"]},
        {"name": "ci90_delta_positive", "passed": comparison["bootstrap_ci90_delta_expectancy"][0] > 0.0, "actual": comparison["bootstrap_ci90_delta_expectancy"]},
        {"name": "three_positive_folds", "passed": p4["positive_folds"] >= 3, "actual": p4["positive_folds"]},
        {"name": "three_folds_improve_p0", "passed": p4["folds_improved_vs_p0"] >= 3, "actual": p4["folds_improved_vs_p0"]},
        {"name": "superior_to_p1", "passed": p4["net_expectancy"] > p1["net_expectancy"], "actual": p4["net_expectancy"] - p1["net_expectancy"]},
        {"name": "superior_without_best_one_percent", "passed": outliers["excluding_best_one_percent"]["delta"] > 0.0, "actual": outliers["excluding_best_one_percent"]["delta"]},
        {"name": "tail_risk_not_worse", "passed": p4["cvar_05"] >= p0["cvar_05"] and p4["maximum_drawdown_non_compounded"] <= p0["maximum_drawdown_non_compounded"] + 1e-12, "actual": {"p4_cvar": p4["cvar_05"], "p0_cvar": p0["cvar_05"], "p4_drawdown": p4["maximum_drawdown_non_compounded"], "p0_drawdown": p0["maximum_drawdown_non_compounded"]}},
        {"name": "conservative_intrabar_preserves_improvement", "passed": comparison["delta_net_expectancy"] > 0.0, "actual": comparison["delta_net_expectancy"]},
        {"name": "symbol_concentration_at_most_30pct", "passed": concentration["maximum_positive_pnl_symbol_share"] <= 0.30, "actual": concentration["maximum_positive_pnl_symbol_share"]},
        {"name": "minimum_100_trades", "passed": p4["trades"] >= 100, "actual": p4["trades"]},
    ]
    decision = "E4B_CONFIRMATORY_EXIT_HYPOTHESIS_JUSTIFIED" if all(item["passed"] for item in checks) else "E4B_CONFIRMATORY_EXIT_HYPOTHESIS_NOT_JUSTIFIED"
    return classification, decision, checks


def _answers(econ: Mapping[str, Any], outliers: Mapping[str, Any], concentration: Mapping[str, Any], events: Sequence[Mapping[str, Any]]) -> Mapping[str, Any]:
    base = econ["intrabar_modes"]["CONSERVATIVE"]["B_BASE"]
    pessimistic = econ["intrabar_modes"]["CONSERVATIVE"]["C_PESSIMISTIC"]
    p0 = base["P0"]; p1 = base["P1"]; p2 = base["P2"]; p3 = base["P3"]; p4 = base["P4"]
    p4_events = Counter(row["event_type"] for row in events if row["policy"] == "P4" and row["intrabar_mode"] == "CONSERVATIVE")
    by_symbol = outliers["by_symbol"]
    return {
        "1_horizon_extension": {"delta_net_p1_vs_p0": p1["pooled"]["net_expectancy"] - p0["pooled"]["net_expectancy"]},
        "2_fixed_bracket": {"delta_net_p2_vs_p0": p2["pooled"]["net_expectancy"] - p0["pooled"]["net_expectancy"]},
        "3_break_even": {"delta_net_p3_vs_p2": p3["pooled"]["net_expectancy"] - p2["pooled"]["net_expectancy"]},
        "4_full_core": {"delta_net_p4_vs_p3": p4["pooled"]["net_expectancy"] - p3["pooled"]["net_expectancy"]},
        "5_activation_counts": dict(sorted(p4_events.items())),
        "6_fixed_tp_exit_count": p4["pooled"]["exit_reason_counts"].get("CLOSED_TAKE_PROFIT", 0),
        "7_initial_stop_exit_count": p4["pooled"]["exit_reason_counts"].get("CLOSED_STOP", 0),
        "8_p4_positive_gross": p4["pooled"]["gross_expectancy"] > 0.0,
        "9_p4_positive_b_base_net": p4["pooled"]["net_expectancy"] > 0.0,
        "10_p4_positive_pessimistic_net": pessimistic["P4"]["pooled"]["net_expectancy"] > 0.0,
        "11_folds_improved_vs_p0": p4["pooled"]["folds_improved_vs_p0"],
        "12_p4_superior_to_p1": p4["pooled"]["net_expectancy"] > p1["pooled"]["net_expectancy"],
        "13_capture_ratio_delta": p4["comparison_to_p0"]["delta_capture_ratio"],
        "14_giveback_delta": p4["pooled"]["giveback_mean"] - p0["pooled"]["giveback_mean"],
        "15_tail_and_drawdown": {"p4_cvar": p4["pooled"]["cvar_05"], "p0_cvar": p0["pooled"]["cvar_05"], "p4_drawdown": p4["pooled"]["maximum_drawdown_non_compounded"], "p0_drawdown": p0["pooled"]["maximum_drawdown_non_compounded"]},
        "16_without_best_one_percent_delta": outliers["excluding_best_one_percent"]["delta"],
        "17_symbols_with_positive_delta": sum(item["delta"] > 0.0 for item in by_symbol.values()),
        "17_symbol_count": len(by_symbol),
        "18_intrabar_ambiguous_trades": p4["pooled"]["intrabar_ambiguity_count"],
        "19_conservative_delta_positive": p4["comparison_to_p0"]["delta_net_expectancy"] > 0.0,
        "20_maximum_symbol_positive_pnl_share": concentration["maximum_positive_pnl_symbol_share"],
    }


def _markdown(summary: Mapping[str, Any], econ: Mapping[str, Any]) -> str:
    base = econ["intrabar_modes"]["CONSERVATIVE"]["B_BASE"]
    lines = [
        "# E4A_MECHANICAL_EXIT",
        "",
        "Preregistered dev-only experiment over the 1,292 frozen E3 SHORT entries. No lockbox or operational path was used.",
        "",
        "## Primary B_BASE results",
        "",
        "| Policy | Trades | Gross expectancy | Net expectancy | PF | Positive folds | Ambiguous |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for policy in POLICY_FEATURES:
        item = base[policy]["pooled"]
        lines.append(
            f"| {policy} | {item['trades']} | {item['gross_expectancy']:.12f} | "
            f"{item['net_expectancy']:.12f} | {item['profit_factor']} | "
            f"{item['positive_folds']} | {item['intrabar_ambiguity_count']} |"
        )
    lines.extend((
        "", "## Classification", "", f"`{summary['scientific_classification']}`",
        "", "## Next decision", "", f"`{summary['next_decision']}`", "",
        "## Mandatory questions", "",
    ))
    for key, value in summary["mandatory_answers"].items():
        lines.append(f"- **{key}**: `{canonical_json(value)}`")
    lines.extend(("", "Optimistic intrabar sensitivity is diagnostic only. The conservative result is the sole promotion input.", ""))
    return "\n".join(lines)


def execute_attempt(repository: Path, preregistration_path: Path, output: Path) -> Mapping[str, Any]:
    config = load_preregistration(preregistration_path, repository)
    try:
        entries, entry_manifest = load_frozen_entries(config, repository)
    except Exception as exc:
        raise EntryDriftError(str(exc)) from exc
    dev_boundary = _utc(config["canonical_source"]["dev_boundary_inclusive"])
    pre_bars = int(config["trajectory"]["required_pre_entry_bars"])
    horizon = int(config["trajectory"]["required_post_entry_bars"])
    start = min(item.signal_timestamp for item in entries) - timedelta(minutes=(pre_bars - 1) * 5)
    end = max(item.entry_timestamp for item in entries) + timedelta(minutes=horizon * 5)
    if end - timedelta(microseconds=1) > dev_boundary:
        raise DataCoverageError("requested E4A interval crosses dev boundary")
    source = CanonicalSeriesSource(
        Path(config["canonical_source"]["path"]), DataPurpose.REPLAY,
        expected_manifest_sha256=config["canonical_source"]["manifest_sha256"],
    )
    prices = source.load(start=start, end=end)
    audit = source.audit(verify_content=False)
    trajectories, trajectory_rows, atr_rows, quality = build_trajectories(
        entries, prices, horizon_bars=horizon, pre_entry_bars=pre_bars, dev_boundary=dev_boundary,
    )
    parameters = _parameters(config)
    trades: list[dict[str, Any]] = []
    events: list[dict[str, Any]] = []
    for trajectory in trajectories:
        for mode in INTRABAR_MODES:
            for policy in POLICY_FEATURES:
                trade, policy_events = simulate_policy(trajectory, policy, mode, parameters)
                trades.append(trade); events.extend(policy_events)
    p0 = [row for row in trades if row["policy"] == "P0" and row["intrabar_mode"] == "CONSERVATIVE"]
    p0_by_id = {str(row["trade_id"]): row for row in p0}
    maximum_p0_error = max(abs(float(p0_by_id[item.trade_id]["gross_return"]) - item.expected_gross) for item in entries)
    scenarios = tuple(CostScenario(**item) for item in config["cost_scenarios"])
    base_scenario = next(item for item in scenarios if item.scenario_id == "B_BASE")
    maximum_p0_net_error = max(
        abs(_net_value(p0_by_id[item.trade_id], base_scenario) - item.expected_net) for item in entries
    )
    if max(maximum_p0_error, maximum_p0_net_error) > 1e-12:
        raise PolicySemanticsError(f"P0 reproduction failed: gross={maximum_p0_error}, net={maximum_p0_net_error}")
    bootstrap = config["metrics"]["bootstrap"]
    econ = economic_results(
        trades, scenarios, repetitions=int(bootstrap["repetitions"]), seed=int(bootstrap["seed"]),
    )
    outliers, concentration = outlier_and_concentration(trades, base_scenario)
    classification, decision, checks = _classification_and_decision(econ, outliers, concentration, quality)
    answers = _answers(econ, outliers, concentration, events)
    policy_results = {
        "schema_version": "aegis-e4a-policy-results-v1",
        "trade_count_per_policy_and_mode": len(trajectories),
        "event_counts": {
            f"{mode}:{policy}": dict(sorted(Counter(
                row["event_type"] for row in events if row["intrabar_mode"] == mode and row["policy"] == policy
            ).items())) for mode in INTRABAR_MODES for policy in POLICY_FEATURES
        },
        "exit_reason_counts": {
            f"{mode}:{policy}": dict(sorted(Counter(
                row["exit_reason"] for row in trades if row["intrabar_mode"] == mode and row["policy"] == policy
            ).items())) for mode in INTRABAR_MODES for policy in POLICY_FEATURES
        },
    }
    fold_results = {
        "schema_version": "aegis-e4a-fold-results-v1",
        "primary": {
            policy: econ["intrabar_modes"]["CONSERVATIVE"]["B_BASE"][policy]["by_fold"]
            for policy in POLICY_FEATURES
        },
    }
    sensitivity = {
        "schema_version": "aegis-e4a-intrabar-sensitivity-v1",
        "primary": "CONSERVATIVE", "optimistic_promotion_use": False,
        "by_policy": {
            policy: {
                "conservative": econ["intrabar_modes"]["CONSERVATIVE"]["B_BASE"][policy]["pooled"],
                "optimistic": econ["intrabar_modes"]["OPTIMISTIC"]["B_BASE"][policy]["pooled"],
            } for policy in POLICY_FEATURES
        },
    }
    summary = {
        "schema_version": "aegis-e4a-diagnostic-summary-v1", "technical_status": "E4A_COMPLETE",
        "scientific_classification": classification, "next_decision": decision,
        "quality": quality, "p0_reproduction": {"maximum_gross_error": maximum_p0_error, "maximum_net_error": maximum_p0_net_error, "tolerance": 1e-12},
        "e4b_criteria": checks, "all_e4b_criteria_passed": all(item["passed"] for item in checks),
        "mandatory_answers": answers,
        "selection_warning": "E4A is exploratory. No policy is a Candidate or operationally approved.",
    }
    trajectory_manifest = {
        "schema_version": "aegis-e4a-trajectory-manifest-v1", "canonical_artifact_id": audit.artifact_id,
        "canonical_manifest_sha256": audit.manifest_sha256, "canonical_content_sha256": audit.content_sha256,
        "source_interval": [_iso(start), _iso(end)], "resolution": "5m", "timezone": "UTC",
        "read_only": audit.read_only, "finality_verified": audit.finality_verified,
        "dev_boundary_inclusive": _iso(dev_boundary), "quality": quality,
    }
    atr_manifest = {
        "schema_version": "aegis-e4a-atr-manifest-v1", "period": 14,
        "initialisation": "simple mean of 14 completed TR values using 15 pre-entry bars",
        "recursive_update": "Wilder", "multiplier": 1.5, "lookahead": False,
        "row_count": len(atr_rows), "initial_atr_hash": Sha256HashProvider().digest_value({item.entry.trade_id: item.initial_atr for item in trajectories}),
    }
    policy_definition = {
        "schema_version": "aegis-e4a-policy-definition-v1",
        "preregistration_sha256": sha256_file(preregistration_path),
        "parameters": parameters.__dict__, "policies": {key: sorted(value) for key, value in POLICY_FEATURES.items()},
        "intrabar_modes": INTRABAR_MODES, "primary_intrabar_mode": "CONSERVATIVE",
        "exit_eye": "EXCLUDED", "partial_exits": False,
    }
    output.mkdir(parents=True, exist_ok=True)
    atomic_json(output / "entry_source_manifest.json", entry_manifest)
    atomic_json(output / "trajectory_manifest.json", trajectory_manifest)
    atomic_json(output / "atr_manifest.json", atr_manifest)
    atomic_parquet(output / "trajectories.parquet", trajectory_rows, tuple(trajectory_rows[0]))
    atomic_parquet(output / "atr.parquet", atr_rows, tuple(atr_rows[0]))
    atomic_json(output / "policy_definition.json", policy_definition)
    atomic_parquet(output / "policy_events.parquet", events, tuple(events[0]))
    atomic_parquet(output / "trades.parquet", trades, tuple(trades[0]))
    atomic_json(output / "policy_results.json", policy_results)
    atomic_json(output / "fold_results.json", fold_results)
    atomic_json(output / "econ_results.json", econ)
    atomic_json(output / "intrabar_sensitivity.json", sensitivity)
    atomic_json(output / "outlier_sensitivity.json", outliers)
    atomic_json(output / "symbol_concentration.json", concentration)
    atomic_json(output / "diagnostic_summary.json", summary)
    _atomic_bytes(output / "diagnostic_summary.md", _markdown(summary, econ).encode("utf-8"))
    hashes = {name: sha256_file(output / name) for name in SCIENTIFIC_FILES if name != "scientific_aggregate.json"}
    aggregate = {
        "schema_version": "aegis-e4a-scientific-aggregate-v1",
        "preregistration_sha256": sha256_file(preregistration_path), "artifact_sha256": hashes,
        "aggregate_hash": Sha256HashProvider().digest_value(hashes),
    }
    atomic_json(output / "scientific_aggregate.json", aggregate)
    return {"summary": summary, "aggregate": aggregate, "quality": quality}


def write_run_manifest(
    path: Path, *, attempt: int, repository: Path, preregistration_path: Path, result: Mapping[str, Any],
) -> None:
    commit = subprocess.run(("git", "rev-parse", "HEAD"), cwd=repository, check=True, capture_output=True, text=True).stdout.strip()
    atomic_json(path, {
        "schema_version": "aegis-e4a-run-manifest-v1", "attempt": attempt,
        "completed_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "code_commit": commit, "preregistration_sha256": sha256_file(preregistration_path),
        "scientific_aggregate_hash": result["aggregate"]["aggregate_hash"],
        "technical_status": result["summary"]["technical_status"],
    })


def finalize_attempts(repository: Path, preregistration_path: Path, root: Path) -> Mapping[str, Any]:
    attempts = (root / "attempt_1", root / "attempt_2")
    comparisons = {}
    for name in SCIENTIFIC_FILES:
        hashes = [sha256_file(path / name) for path in attempts]
        comparisons[name] = {"attempt_1": hashes[0], "attempt_2": hashes[1], "byte_identical": hashes[0] == hashes[1]}
    if not all(item["byte_identical"] for item in comparisons.values()):
        raise NondeterministicError(canonical_json(comparisons))
    normalized_manifests = []
    for attempt in attempts:
        payload = json.loads((attempt / "run_manifest.json").read_text(encoding="utf-8"))
        payload.pop("completed_at_utc", None); payload.pop("attempt", None)
        normalized_manifests.append(Sha256HashProvider().digest_value(payload))
    if len(set(normalized_manifests)) != 1:
        raise NondeterministicError("normalized run manifests differ")
    for name in SCIENTIFIC_FILES:
        shutil.copyfile(attempts[0] / name, root / name)
    for index, attempt in enumerate(attempts, 1):
        shutil.copyfile(attempt / "run_manifest.json", root / f"run_manifest_attempt_{index}.json")
    report = {
        "schema_version": "aegis-e4a-determinism-v1", "tolerance": 1e-12,
        "scientific_artifacts_byte_identical": True, "comparisons": comparisons,
        "normalized_run_manifest_hash": normalized_manifests[0],
        "aggregate_hash": json.loads((root / "scientific_aggregate.json").read_text(encoding="utf-8"))["aggregate_hash"],
    }
    atomic_json(root / "determinism_report.json", report)
    return {
        "determinism": report,
        "summary": json.loads((root / "diagnostic_summary.json").read_text(encoding="utf-8")),
    }
