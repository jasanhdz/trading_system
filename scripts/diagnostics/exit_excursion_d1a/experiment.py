"""Deterministic descriptive excursion analysis over frozen E3 SHORT entries."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import shutil
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from aegis.data import CanonicalBar, CanonicalSeriesSource, DataPurpose
from aegis.training.econ import CostScenario
from aegis.utils import Sha256HashProvider, canonical_json, sha256_file


class ExitDiagnosticError(RuntimeError):
    """Base fail-closed diagnostic error."""


class EntryDriftError(ExitDiagnosticError):
    pass


class BaselineReproductionError(ExitDiagnosticError):
    pass


class DataLimitationError(ExitDiagnosticError):
    pass


class NondeterministicError(ExitDiagnosticError):
    pass


@dataclass(frozen=True)
class FrozenEntry:
    trade_id: str
    symbol: str
    fold: int
    signal_timestamp: datetime
    entry_timestamp: datetime
    exit_timestamp: datetime
    entry_price: float
    exit_price: float
    expected_gross: float
    expected_mfe: float
    expected_mae: float
    expected_cost: float
    expected_net: float


def _utc(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise ValueError("timestamps must include a timezone")
    return parsed.astimezone(timezone.utc)


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _atomic_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    descriptor = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def atomic_json(path: Path, value: Any) -> None:
    _atomic_bytes(path, (canonical_json(value) + "\n").encode("utf-8"))


def atomic_csv(path: Path, rows: Sequence[Mapping[str, Any]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n", extrasaction="raise")
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name) for name in fieldnames})
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def load_preregistration(path: Path) -> Mapping[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    required = {
        "schema_version", "experiment_id", "classification", "side", "timeframe_minutes",
        "holding_bars", "entry_rule", "exit_rule", "dev_boundary_inclusive", "entry_source",
        "canonical_source", "temporal_horizons_bars", "favorable_thresholds_bps",
        "adverse_thresholds_bps", "percentiles", "cost_scenarios", "bootstrap", "forbidden",
    }
    if not required <= set(payload):
        raise DataLimitationError(f"preregistration fields missing: {sorted(required - set(payload))}")
    if (
        payload["experiment_id"] != "EXIT_EXCURSION_D1A"
        or payload["side"] != "SHORT"
        or payload["holding_bars"] != 12
        or payload["entry_rule"] != "NEXT_BAR_OPEN"
        or payload["exit_rule"] != "H12_CLOSE"
        or payload["temporal_horizons_bars"] != [1, 2, 3, 6, 9, 12]
    ):
        raise DataLimitationError("D1A frozen temporal protocol mismatch")
    return payload


def load_frozen_entries(config: Mapping[str, Any], repository: Path) -> tuple[tuple[FrozenEntry, ...], Mapping[str, Any]]:
    source_config = config["entry_source"]
    source_path = repository / source_config["path"]
    actual_physical = sha256_file(source_path)
    if actual_physical != source_config["physical_sha256"]:
        raise EntryDriftError("frozen E3 econ_report physical hash mismatch")
    payload = json.loads(source_path.read_text(encoding="utf-8"))
    source_trades = tuple(
        trade for trade in payload["report"]["trades"]
        if trade["scenario_id"] == source_config["scenario_id"]
        and trade["signal"]["strategy_id"] == source_config["strategy_id"]
    )
    digest = Sha256HashProvider().digest_value(source_trades)
    if len(source_trades) != source_config["expected_trade_count"] or digest != source_config["expected_trade_set_hash"]:
        raise EntryDriftError("frozen E3 entry set count or canonical hash mismatch")
    entries = []
    for trade in source_trades:
        signal = trade["signal"]
        if signal["side"] != "SHORT":
            raise EntryDriftError("entry set contains a non-SHORT trade")
        identity = {
            "symbol": signal["symbol"], "fold": signal["fold"],
            "signal_timestamp": signal["timestamp"], "entry_timestamp": trade["entry_timestamp"],
        }
        entries.append(FrozenEntry(
            Sha256HashProvider().digest_value(identity)[:24], signal["symbol"], int(signal["fold"]),
            _utc(signal["timestamp"]), _utc(trade["entry_timestamp"]), _utc(trade["exit_timestamp"]),
            float(trade["entry_price"]), float(trade["exit_price"]), float(trade["gross_return_fraction"]),
            float(trade["mfe_fraction"]), float(trade["mae_fraction"]), float(trade["cost_fraction"]),
            float(trade["net_return_fraction"]),
        ))
    entries.sort(key=lambda item: (item.signal_timestamp, item.symbol, item.fold))
    if len({item.trade_id for item in entries}) != len(entries):
        raise EntryDriftError("frozen E3 entry identities are not unique")
    return tuple(entries), {
        "source_path": str(source_config["path"]), "physical_sha256": actual_physical,
        "trade_set_hash": digest, "trade_count": len(entries), "side": "SHORT",
        "minimum_signal_timestamp": entries[0].signal_timestamp,
        "maximum_signal_timestamp": entries[-1].signal_timestamp,
        "e3_b_base_metrics": payload["report"]["metrics"]["full_stack"]["B_BASE"],
    }


def short_mfe_mae(entry_price: float, path: Sequence[CanonicalBar]) -> tuple[float, float, int, int]:
    if entry_price <= 0 or not path:
        raise ValueError("positive entry and non-empty path required")
    lows = np.asarray([bar.low for bar in path], dtype=np.float64)
    highs = np.asarray([bar.high for bar in path], dtype=np.float64)
    mfe_values = np.maximum(0.0, (entry_price - lows) / entry_price)
    mae_values = np.maximum(0.0, (highs - entry_price) / entry_price)
    return float(mfe_values.max()), float(mae_values.max()), int(mfe_values.argmax()) + 1, int(mae_values.argmax()) + 1


def giveback_and_capture(mfe: float, gross_return: float) -> tuple[float, float | None]:
    giveback = mfe - gross_return
    return giveback, None if mfe == 0.0 else gross_return / mfe


def cumulative_short_mfe(entry_price: float, path: Sequence[CanonicalBar]) -> tuple[float, ...]:
    best = entry_price
    result = []
    for bar in path:
        best = min(best, bar.low)
        result.append(max(0.0, (entry_price - best) / entry_price))
    return tuple(result)


def temporal_short_returns(entry_price: float, path: Sequence[CanonicalBar]) -> tuple[float, ...]:
    return tuple((entry_price - bar.close) / entry_price for bar in path)


def threshold_before_adverse(
    entry_price: float, path: Sequence[CanonicalBar], favorable_bps: int, adverse_bps: int,
) -> tuple[bool, bool]:
    favorable_level = favorable_bps / 10_000.0
    adverse_level = adverse_bps / 10_000.0
    favorable_bar = next((index for index, bar in enumerate(path, 1) if (entry_price - bar.low) / entry_price >= favorable_level), None)
    adverse_bar = next((index for index, bar in enumerate(path, 1) if (bar.high - entry_price) / entry_price >= adverse_level), None)
    if favorable_bar is None:
        return False, False
    if adverse_bar is None or favorable_bar < adverse_bar:
        return True, False
    return False, favorable_bar == adverse_bar


def _percentiles(values: Sequence[float], percentiles: Sequence[int]) -> Mapping[str, float | None]:
    if not values:
        return {f"p{value}": None for value in percentiles}
    array = np.asarray(values, dtype=np.float64)
    return {f"p{value}": float(np.percentile(array, value)) for value in percentiles}


def _distribution(values: Sequence[float], percentiles: Sequence[int]) -> Mapping[str, Any]:
    if not values:
        return {"count": 0, "mean": None, "median": None, "percentiles": _percentiles((), percentiles)}
    array = np.asarray(values, dtype=np.float64)
    return {
        "count": len(values), "mean": float(array.mean()), "median": float(np.median(array)),
        "percentiles": _percentiles(values, percentiles),
    }


def _correlation(left: Sequence[float], right: Sequence[float]) -> float | None:
    if len(left) < 2 or np.std(left) == 0.0 or np.std(right) == 0.0:
        return None
    return float(np.corrcoef(np.asarray(left), np.asarray(right))[0, 1])


def build_trajectories(
    entries: Sequence[FrozenEntry], prices: Mapping[str, Sequence[CanonicalBar]], *,
    holding_bars: int, dev_boundary: datetime, tolerance: float = 1e-12,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], Mapping[str, Any]]:
    indices = {symbol: {bar.timestamp: index for index, bar in enumerate(rows)} for symbol, rows in prices.items()}
    trajectory_rows: list[dict[str, Any]] = []
    excursion_rows: list[dict[str, Any]] = []
    baseline_errors = []
    maximum_error = 0.0
    tied_extrema = 0
    for entry in entries:
        rows = prices.get(entry.symbol)
        index = indices.get(entry.symbol, {}).get(entry.signal_timestamp)
        if rows is None or index is None or index + holding_bars >= len(rows):
            raise DataLimitationError(f"missing H12 trajectory for {entry.trade_id}")
        path = tuple(rows[index + 1:index + holding_bars + 1])
        if len(path) != holding_bars or path[0].timestamp != entry.entry_timestamp:
            raise EntryDriftError(f"entry alignment drift for {entry.trade_id}")
        for offset in range(1, len(path)):
            if path[offset].timestamp - path[offset - 1].timestamp != timedelta(minutes=5):
                raise DataLimitationError(f"trajectory gap for {entry.trade_id}")
        if path[-1].timestamp > dev_boundary:
            raise DataLimitationError("trajectory crosses the frozen dev boundary")
        mfe, mae, mfe_bar, mae_bar = short_mfe_mae(entry.entry_price, path)
        gross = (entry.entry_price - path[-1].close) / entry.entry_price
        giveback, capture = giveback_and_capture(mfe, gross)
        cumulative_mfe = cumulative_short_mfe(entry.entry_price, path)
        mtm = temporal_short_returns(entry.entry_price, path)
        errors = {
            "entry": abs(entry.entry_price - path[0].open), "exit": abs(entry.exit_price - path[-1].close),
            "gross": abs(entry.expected_gross - gross), "mfe": abs(entry.expected_mfe - mfe),
            "mae": abs(entry.expected_mae - mae),
            "net": abs(entry.expected_net - (gross - entry.expected_cost)),
        }
        maximum_error = max(maximum_error, *errors.values())
        if max(errors.values()) > tolerance or entry.exit_timestamp != path[-1].timestamp:
            baseline_errors.append({"trade_id": entry.trade_id, "errors": errors})
        same_extreme_bar = mfe_bar == mae_bar
        tied_extrema += int(same_extreme_bar)
        row: dict[str, Any] = {
            "trade_id": entry.trade_id, "symbol": entry.symbol, "fold": entry.fold,
            "signal_timestamp": entry.signal_timestamp.isoformat().replace("+00:00", "Z"),
            "entry_timestamp": entry.entry_timestamp.isoformat().replace("+00:00", "Z"),
            "entry_price": entry.entry_price, "h12_close": path[-1].close, "gross_h12_return": gross,
            "mfe_h12": mfe, "mae_h12": mae, "giveback": giveback, "capture_ratio": capture,
            "mfe_bar": mfe_bar, "mfe_minutes": mfe_bar * 5, "mae_bar": mae_bar,
            "mae_minutes": mae_bar * 5, "ever_positive": mfe > 0.0,
            "favorable_then_negative": mfe > 0.0 and gross < 0.0, "ended_positive": gross > 0.0,
            "ended_positive_giveback_25": gross > 0.0 and mfe > 0.0 and giveback / mfe >= 0.25,
            "giveback_50": mfe > 0.0 and giveback / mfe >= 0.50,
            "giveback_75": mfe > 0.0 and giveback / mfe >= 0.75,
            "giveback_100": mfe > 0.0 and giveback / mfe >= 1.00,
            "mae_before_mfe": mae_bar < mfe_bar, "mfe_before_mae": mfe_bar < mae_bar,
            "intrabar_extrema_tied": same_extreme_bar,
        }
        for horizon in (1, 2, 3, 6, 9, 12):
            row[f"mfe_t{horizon * 5}"] = cumulative_mfe[horizon - 1]
            row[f"mtm_t{horizon * 5}"] = mtm[horizon - 1]
        excursion_rows.append(row)
        for bar_index, bar in enumerate(path, 1):
            trajectory_rows.append({
                "trade_id": entry.trade_id, "symbol": entry.symbol, "fold": entry.fold,
                "signal_timestamp": entry.signal_timestamp.isoformat().replace("+00:00", "Z"),
                "entry_timestamp": entry.entry_timestamp.isoformat().replace("+00:00", "Z"),
                "entry_price": entry.entry_price, "bar_index": bar_index,
                "bar_timestamp": bar.timestamp.isoformat().replace("+00:00", "Z"),
                "open": bar.open, "high": bar.high, "low": bar.low, "close": bar.close,
                "volume": bar.volume, "is_final": True, "gap_from_previous": False,
            })
    if baseline_errors:
        raise BaselineReproductionError(canonical_json({"first_errors": baseline_errors[:5], "count": len(baseline_errors)}))
    return trajectory_rows, excursion_rows, {
        "trade_count": len(entries), "trajectory_rows": len(trajectory_rows), "holding_bars": holding_bars,
        "gap_count": 0, "missing_path_count": 0, "intrabar_extrema_tied": tied_extrema,
        "baseline_maximum_absolute_error": maximum_error, "baseline_tolerance": tolerance,
    }


def summarize_excursions(
    rows: Sequence[Mapping[str, Any]], *, percentiles: Sequence[int], favorable_bps: Sequence[int],
    adverse_bps: Sequence[int], paths: Mapping[str, Sequence[CanonicalBar]], entries: Mapping[str, FrozenEntry],
) -> Mapping[str, Any]:
    def group(group_rows: Sequence[Mapping[str, Any]]) -> Mapping[str, Any]:
        mfe = [float(row["mfe_h12"]) for row in group_rows]
        mae = [float(row["mae_h12"]) for row in group_rows]
        gross = [float(row["gross_h12_return"]) for row in group_rows]
        giveback = [float(row["giveback"]) for row in group_rows]
        capture = [float(row["capture_ratio"]) for row in group_rows if row["capture_ratio"] is not None]
        result: dict[str, Any] = {
            "trades": len(group_rows), "mfe": _distribution(mfe, percentiles),
            "mae": _distribution(mae, percentiles), "gross_h12_return": _distribution(gross, percentiles),
            "giveback": _distribution(giveback, percentiles), "capture_ratio": _distribution(capture, percentiles),
            "time_to_mfe_minutes": _distribution([float(row["mfe_minutes"]) for row in group_rows], percentiles),
            "time_to_mae_minutes": _distribution([float(row["mae_minutes"]) for row in group_rows], percentiles),
            "ever_positive_fraction": sum(bool(row["ever_positive"]) for row in group_rows) / max(1, len(group_rows)),
            "favorable_then_negative_fraction": sum(bool(row["favorable_then_negative"]) for row in group_rows) / max(1, len(group_rows)),
            "mfe_before_mae_fraction": sum(bool(row["mfe_before_mae"]) for row in group_rows) / max(1, len(group_rows)),
            "mae_before_mfe_fraction": sum(bool(row["mae_before_mfe"]) for row in group_rows) / max(1, len(group_rows)),
            "intrabar_extrema_tied_fraction": sum(bool(row["intrabar_extrema_tied"]) for row in group_rows) / max(1, len(group_rows)),
            "mfe_final_return_correlation": _correlation(mfe, gross),
            "mae_final_return_correlation": _correlation(mae, gross),
            "zero_mfe_count": sum(value == 0.0 for value in mfe),
        }
        result["giveback_fractions"] = {
            key: sum(bool(row[key]) for row in group_rows) / max(1, len(group_rows))
            for key in ("ended_positive_giveback_25", "giveback_50", "giveback_75", "giveback_100")
        }
        return result

    pooled = group(rows)
    threshold_table: dict[str, Any] = {}
    row_by_id = {str(row["trade_id"]): row for row in rows}
    for favorable in favorable_bps:
        independent = sum(float(row["mfe_h12"]) >= favorable / 10_000.0 for row in rows)
        threshold = {"independent": independent, "fraction": independent / len(rows), "before_adverse": {}, "ambiguous_same_bar": {}}
        for adverse in adverse_bps:
            before = ambiguous = 0
            for trade_id in row_by_id:
                reached, tied = threshold_before_adverse(entries[trade_id].entry_price, paths[trade_id], favorable, adverse)
                before += int(reached); ambiguous += int(tied)
            threshold["before_adverse"][str(adverse)] = before
            threshold["ambiguous_same_bar"][str(adverse)] = ambiguous
        threshold_table[str(favorable)] = threshold
    return {
        "schema_version": "aegis-exit-excursion-summary-v1", "pooled": pooled,
        "by_fold": {str(fold): group([row for row in rows if int(row["fold"]) == fold]) for fold in range(1, 5)},
        "favorable_thresholds_bps": threshold_table,
    }


def _economic_metrics(
    records: Sequence[Mapping[str, Any]], *, scenario: CostScenario, holding_bars: int,
    bootstrap_repetitions: int, seed: int,
) -> Mapping[str, Any]:
    if not records:
        return {
            "trades": 0, "gross_expectancy": 0.0, "net_expectancy": 0.0,
            "profit_factor": None, "win_rate": 0.0, "payoff_ratio": None,
            "maximum_drawdown_non_compounded": 0.0,
            "cost_per_trade": scenario.cost_fraction(holding_bars), "total_cost": 0.0,
            "bootstrap_ci_90": [0.0, 0.0],
        }
    gross = np.asarray([float(row["gross"]) for row in records], dtype=np.float64)
    cost = scenario.cost_fraction(holding_bars)
    net = gross - cost
    gains = float(net[net > 0].sum()); losses = float(-net[net < 0].sum())
    equity = np.cumsum(net); peaks = np.maximum.accumulate(np.concatenate(([0.0], equity)))
    wins = net[net > 0]; losing = net[net < 0]
    weeks: dict[tuple[int, int], list[float]] = {}
    for row, value in zip(records, net, strict=True):
        iso = _utc(str(row["signal_timestamp"])).isocalendar()
        weeks.setdefault((iso.year, iso.week), []).append(float(value))
    rng = np.random.default_rng(seed); keys = sorted(weeks); bootstrap = []
    for _ in range(bootstrap_repetitions):
        sampled = rng.choice(len(keys), size=len(keys), replace=True)
        bootstrap.append(float(np.mean([value for index in sampled for value in weeks[keys[int(index)]]])))
    return {
        "trades": len(records), "gross_expectancy": float(gross.mean()), "net_expectancy": float(net.mean()),
        "profit_factor": gains / losses if gains > 0 and losses > 0 else (0.0 if gains == 0 else None),
        "win_rate": float(np.mean(net > 0)), "payoff_ratio": (
            float(wins.mean() / -losing.mean()) if len(wins) and len(losing) else None
        ),
        "maximum_drawdown_non_compounded": float(np.max(peaks[1:] - equity)),
        "cost_per_trade": cost, "total_cost": cost * len(records),
        "bootstrap_ci_90": [float(value) for value in np.quantile(bootstrap, (0.05, 0.95))],
    }


def temporal_exit_analysis(
    excursions: Sequence[Mapping[str, Any]], *, horizons: Sequence[int], scenarios: Sequence[CostScenario],
    bootstrap_repetitions: int, seed: int,
) -> Mapping[str, Any]:
    output: dict[str, Any] = {"schema_version": "aegis-temporal-exit-results-v1", "horizons": {}}
    for horizon in horizons:
        records = [
            {"gross": row[f"mtm_t{horizon * 5}"], "signal_timestamp": row["signal_timestamp"], "fold": row["fold"]}
            for row in excursions
        ]
        scenario_results: dict[str, Any] = {}
        for scenario in scenarios:
            pooled = _economic_metrics(
                records, scenario=scenario, holding_bars=horizon,
                bootstrap_repetitions=bootstrap_repetitions, seed=seed,
            )
            folds = {
                str(fold): _economic_metrics(
                    [row for row in records if int(row["fold"]) == fold], scenario=scenario,
                    holding_bars=horizon, bootstrap_repetitions=bootstrap_repetitions, seed=seed,
                )
                for fold in range(1, 5)
            }
            pooled["positive_folds"] = sum(item["net_expectancy"] > 0.0 for item in folds.values())
            scenario_results[scenario.scenario_id] = {"pooled": pooled, "by_fold": folds}
        output["horizons"][f"T{horizon * 5}"] = scenario_results
    output["exploratory_selection_bias"] = {
        scenario.scenario_id: {
            "label": "EXPLORATORY_SELECTION_BIASED",
            "maximum_net_expectancy": max(
                (output["horizons"][name][scenario.scenario_id]["pooled"]["net_expectancy"], name)
                for name in output["horizons"]
            ),
        }
        for scenario in scenarios
    }
    return output


def outlier_sensitivity(
    rows: Sequence[Mapping[str, Any]], percentiles: Sequence[int],
) -> Mapping[str, Any]:
    def summarize(group: Sequence[Mapping[str, Any]]) -> Mapping[str, Any]:
        return {
            "trades": len(group),
            "mfe": _distribution([float(row["mfe_h12"]) for row in group], percentiles),
            "mae": _distribution([float(row["mae_h12"]) for row in group], percentiles),
            "gross_h12_return": _distribution([float(row["gross_h12_return"]) for row in group], percentiles),
        }
    ordered = sorted(rows, key=lambda row: (float(row["mfe_h12"]), str(row["trade_id"])), reverse=True)
    remove = math.ceil(len(rows) * 0.01)
    trimmed = ordered[remove:]
    winsorized = []
    keys = ("mfe_h12", "mae_h12", "gross_h12_return")
    bounds = {key: np.quantile([float(row[key]) for row in rows], (0.01, 0.99)) for key in keys}
    for row in rows:
        copy = dict(row)
        for key in keys:
            copy[key] = float(np.clip(float(row[key]), bounds[key][0], bounds[key][1]))
        winsorized.append(copy)
    return {
        "schema_version": "aegis-exit-outlier-sensitivity-v1", "all_trades": summarize(rows),
        "remove_best_one_percent_by_mfe": summarize(trimmed), "removed_count": remove,
        "winsorized_p1_p99": summarize(winsorized),
        "by_fold": {str(fold): summarize([row for row in rows if int(row["fold"]) == fold]) for fold in range(1, 5)},
        "by_symbol": {symbol: summarize([row for row in rows if row["symbol"] == symbol]) for symbol in sorted({str(row["symbol"]) for row in rows})},
    }


def _scientific_summary(
    excursion_summary: Mapping[str, Any], temporal: Mapping[str, Any], outliers: Mapping[str, Any],
) -> Mapping[str, Any]:
    pooled = excursion_summary["pooled"]
    median_mfe = float(pooled["mfe"]["median"])
    median_giveback = float(pooled["giveback"]["median"])
    material = median_mfe >= 0.001 or excursion_summary["favorable_thresholds_bps"]["10"]["fraction"] >= 0.5
    given_back = material and (
        pooled["giveback_fractions"]["giveback_50"] >= 0.5
        or pooled["favorable_then_negative_fraction"] >= 0.25
        or median_giveback >= median_mfe * 0.5
    )
    too_adverse = material and pooled["mae"]["median"] > median_mfe * 2.0 and pooled["mfe_before_mae_fraction"] < 0.4
    if too_adverse:
        classification = "EXCURSION_TOO_ADVERSE_TO_CAPTURE"
    elif given_back:
        classification = "MATERIAL_EXCURSION_GIVEN_BACK"
    elif material:
        classification = "MATERIAL_EXCURSION_WITH_LOW_GIVEBACK"
    else:
        classification = "NO_MATERIAL_FAVORABLE_EXCURSION"
    decision = "ADAPTIVE_EXIT_RESEARCH_JUSTIFIED" if classification == "MATERIAL_EXCURSION_GIVEN_BACK" else "ADAPTIVE_EXIT_RESEARCH_NOT_JUSTIFIED"
    shorter = {}
    base_h12 = temporal["horizons"]["T60"]["B_BASE"]
    for horizon, result in temporal["horizons"].items():
        if horizon == "T60":
            continue
        folds = sum(
            result["B_BASE"]["by_fold"][str(fold)]["net_expectancy"]
            > base_h12["by_fold"][str(fold)]["net_expectancy"]
            for fold in range(1, 5)
        )
        shorter[horizon] = {
            "pooled_delta_vs_t60": result["B_BASE"]["pooled"]["net_expectancy"] - base_h12["pooled"]["net_expectancy"],
            "folds_improved": folds,
        }
    return {
        "schema_version": "aegis-exit-diagnostic-summary-v1", "technical_status": "EXIT_D1A_COMPLETE",
        "classification": classification, "next_decision": decision,
        "selection_bias_warning": "All temporal horizons are exploratory; the observed maximum is not validated.",
        "materiality_rule": "median MFE >= 10 bps or at least 50% of entries reach 10 bps",
        "research_rule": "justified only for material excursion with frequent giveback; no exit policy is approved",
        "shorter_horizons_vs_t60": shorter,
        "key_metrics": {
            "trades": pooled["trades"], "median_mfe": median_mfe, "median_mae": pooled["mae"]["median"],
            "median_giveback": median_giveback, "median_capture_ratio": pooled["capture_ratio"]["median"],
            "ever_positive_fraction": pooled["ever_positive_fraction"],
            "favorable_then_negative_fraction": pooled["favorable_then_negative_fraction"],
            "mfe_before_mae_fraction": pooled["mfe_before_mae_fraction"],
        },
        "outlier_comparison": {
            "all_mfe_mean": outliers["all_trades"]["mfe"]["mean"],
            "without_best_1pct_mfe_mean": outliers["remove_best_one_percent_by_mfe"]["mfe"]["mean"],
            "winsorized_mfe_mean": outliers["winsorized_p1_p99"]["mfe"]["mean"],
        },
    }


def _format_bps(value: float | None) -> str:
    return "N/A" if value is None else f"{value * 10_000.0:.6f} bps"


def diagnostic_markdown(
    summary: Mapping[str, Any], excursions: Mapping[str, Any], temporal: Mapping[str, Any],
    outliers: Mapping[str, Any], quality: Mapping[str, Any],
) -> str:
    pooled = excursions["pooled"]
    thresholds = excursions["favorable_thresholds_bps"]
    base_t60 = temporal["horizons"]["T60"]["B_BASE"]["pooled"]
    shorter = summary["shorter_horizons_vs_t60"]
    consistent = [name for name, item in shorter.items() if item["folds_improved"] >= 3]
    all_scenarios = {
        horizon: {
            scenario: values["pooled"]["net_expectancy"]
            for scenario, values in result.items()
        }
        for horizon, result in temporal["horizons"].items()
    }
    return "\n".join((
        "# EXIT_EXCURSION_D1A",
        "",
        "Exploratory descriptive analysis of the 1,292 frozen E3 SHORT entries. No stop, target, trailing, callback or risk unit is evaluated.",
        "",
        "## Data quality",
        "",
        f"- P0 maximum absolute reproduction error: `{quality['baseline_maximum_absolute_error']}` (tolerance `1e-12`).",
        f"- Complete 5m trajectories: `{quality['trade_count']}`; gaps: `{quality['gap_count']}`; same-bar extrema order ambiguities: `{quality['intrabar_extrema_tied']}`.",
        "",
        "## Required questions",
        "",
        f"1. Typical MFE: median {_format_bps(pooled['mfe']['median'])}; mean {_format_bps(pooled['mfe']['mean'])}.",
        f"2. Typical MAE: median {_format_bps(pooled['mae']['median'])}; mean {_format_bps(pooled['mae']['mean'])}.",
        f"3. Ever favorable: {pooled['ever_positive_fraction']:.6%}. Reached 10 bps: {thresholds['10']['fraction']:.6%}; 25 bps: {thresholds['25']['fraction']:.6%}; 50 bps: {thresholds['50']['fraction']:.6%}.",
        f"4. Finished negative after being favorable: {pooled['favorable_then_negative_fraction']:.6%}.",
        f"5. Median H12 capture ratio among nonzero-MFE trades: {pooled['capture_ratio']['median']:.6f}.",
        f"6. Giveback of at least 50%: {pooled['giveback_fractions']['giveback_50']:.6%}; at least 100%: {pooled['giveback_fractions']['giveback_100']:.6%}.",
        f"7. Median time to MFE: {pooled['time_to_mfe_minutes']['median']:.1f} minutes within H12.",
        f"8. MFE preceded MAE in {pooled['mfe_before_mae_fraction']:.6%}; MAE preceded MFE in {pooled['mae_before_mfe_fraction']:.6%}; ties are reported separately.",
        f"9. Favorable movement before adverse movement is reported for every frozen bps pair in `excursion_summary.json`; same-bar events are not ordered.",
        f"10. Shorter deterministic exits are listed without selection in `temporal_exit_results.json`; deltas versus T60 are {canonical_json(shorter)}.",
        f"11. Horizons improving versus T60 in at least 3/4 folds: {consistent or 'none'}.",
        f"12. T60 B_BASE net expectancy is {base_t60['net_expectancy']:.12f}; all horizon/scenario net results are {canonical_json(all_scenarios)}.",
        f"13. Mean MFE all trades {_format_bps(outliers['all_trades']['mfe']['mean'])}; without best 1% {_format_bps(outliers['remove_best_one_percent_by_mfe']['mfe']['mean'])}; winsorized {_format_bps(outliers['winsorized_p1_p99']['mfe']['mean'])}.",
        f"14. Descriptive decision: `{summary['next_decision']}`. This does not approve an exit policy.",
        "15. Yes. A new unambiguous risk unit would be required before any R-based trailing or callback experiment.",
        "",
        "## Classification",
        "",
        f"`{summary['classification']}`",
        "",
        "## Next decision",
        "",
        f"`{summary['next_decision']}`",
        "",
    ))


TRAJECTORY_FIELDS = (
    "trade_id", "symbol", "fold", "signal_timestamp", "entry_timestamp", "entry_price", "bar_index",
    "bar_timestamp", "open", "high", "low", "close", "volume", "is_final", "gap_from_previous",
)
EXCURSION_FIELDS = (
    "trade_id", "symbol", "fold", "signal_timestamp", "entry_timestamp", "entry_price", "h12_close",
    "gross_h12_return", "mfe_h12", "mae_h12", "giveback", "capture_ratio", "mfe_bar", "mfe_minutes",
    "mae_bar", "mae_minutes", "ever_positive", "favorable_then_negative", "ended_positive",
    "ended_positive_giveback_25", "giveback_50", "giveback_75", "giveback_100", "mae_before_mfe",
    "mfe_before_mae", "intrabar_extrema_tied", "mfe_t5", "mtm_t5", "mfe_t10", "mtm_t10",
    "mfe_t15", "mtm_t15", "mfe_t30", "mtm_t30", "mfe_t45", "mtm_t45", "mfe_t60", "mtm_t60",
)
SCIENTIFIC_FILES = (
    "entry_source_manifest.json", "trajectory_manifest.json", "trajectories.csv",
    "excursions_per_trade.csv", "excursion_summary.json", "temporal_exit_results.json",
    "fold_results.json", "outlier_sensitivity.json", "diagnostic_summary.json", "diagnostic_summary.md",
    "scientific_aggregate.json",
)


def execute_attempt(repository: Path, preregistration_path: Path, output: Path) -> Mapping[str, Any]:
    config = load_preregistration(preregistration_path)
    entries, entry_manifest = load_frozen_entries(config, repository)
    dev_boundary = _utc(config["dev_boundary_inclusive"])
    start = min(entry.signal_timestamp for entry in entries)
    end = max(entry.exit_timestamp for entry in entries) + timedelta(minutes=5)
    if end - timedelta(microseconds=1) > dev_boundary:
        raise DataLimitationError("requested canonical interval crosses dev boundary")
    source = CanonicalSeriesSource(
        Path(config["canonical_source"]["path"]), DataPurpose.REPLAY,
        expected_manifest_sha256=config["canonical_source"]["manifest_sha256"],
    )
    prices = source.load(start=start, end=end)
    audit = source.audit(verify_content=False)
    trajectories, excursions, quality = build_trajectories(
        entries, prices, holding_bars=int(config["holding_bars"]), dev_boundary=dev_boundary,
    )
    path_by_id: dict[str, list[CanonicalBar]] = {entry.trade_id: [] for entry in entries}
    for row in trajectories:
        path_by_id[str(row["trade_id"])].append(CanonicalBar(
            _utc(str(row["bar_timestamp"])), float(row["open"]), float(row["high"]),
            float(row["low"]), float(row["close"]), float(row["volume"]),
        ))
    entry_by_id = {entry.trade_id: entry for entry in entries}
    excursion_summary = summarize_excursions(
        excursions, percentiles=config["percentiles"], favorable_bps=config["favorable_thresholds_bps"],
        adverse_bps=config["adverse_thresholds_bps"], paths=path_by_id, entries=entry_by_id,
    )
    scenarios = tuple(CostScenario(**item) for item in config["cost_scenarios"])
    base_scenario = next(item for item in scenarios if item.scenario_id == "B_BASE")
    expected_base_cost = base_scenario.cost_fraction(int(config["holding_bars"]))
    if max(abs(entry.expected_cost - expected_base_cost) for entry in entries) > 1e-12:
        raise BaselineReproductionError("E3 B_BASE cost contract drift")
    temporal = temporal_exit_analysis(
        excursions, horizons=config["temporal_horizons_bars"], scenarios=scenarios,
        bootstrap_repetitions=int(config["bootstrap"]["repetitions"]), seed=int(config["bootstrap"]["seed"]),
    )
    outliers = outlier_sensitivity(excursions, config["percentiles"])
    summary = _scientific_summary(excursion_summary, temporal, outliers)
    trajectory_manifest = {
        "schema_version": "aegis-exit-trajectory-manifest-v1", "canonical_artifact_id": audit.artifact_id,
        "canonical_manifest_sha256": audit.manifest_sha256, "canonical_content_sha256": audit.content_sha256,
        "resolution": "5m", "timezone": "UTC", "source_interval": [start, end],
        "dev_boundary_inclusive": dev_boundary, "read_only": audit.read_only,
        "finality_verified": audit.finality_verified, "gap_free": audit.gap_free, "quality": quality,
        "ohlc_semantics": "bar timestamp is open time; entry is next bar open; H12 exits at bar 12 close",
        "intrabar_order": "not inferred; same-bar MFE/MAE order is marked ambiguous",
    }
    original = entry_manifest["e3_b_base_metrics"]
    reproduced = temporal["horizons"]["T60"]["B_BASE"]["pooled"]
    aggregate_checks = {
        "expectancy": abs(float(original["expectancy"]) - float(reproduced["net_expectancy"])),
        "profit_factor": abs(float(original["profit_factor"]) - float(reproduced["profit_factor"])),
        "win_rate": abs(float(original["win_rate"]) - float(reproduced["win_rate"])),
        "maximum_drawdown": abs(float(original["maximum_drawdown"]) - float(reproduced["maximum_drawdown_non_compounded"])),
    }
    if max(aggregate_checks.values()) > 1e-12:
        raise BaselineReproductionError(canonical_json({"aggregate_errors": aggregate_checks}))
    quality = {**quality, "aggregate_reproduction_errors": aggregate_checks}
    trajectory_manifest = {**trajectory_manifest, "quality": quality}
    fold_results = {
        "schema_version": "aegis-exit-fold-results-v1", "excursions": excursion_summary["by_fold"],
        "temporal_exits": {
            horizon: {scenario: result[scenario]["by_fold"] for scenario in result}
            for horizon, result in temporal["horizons"].items()
        },
    }
    output.mkdir(parents=True, exist_ok=True)
    atomic_json(output / "entry_source_manifest.json", entry_manifest)
    atomic_json(output / "trajectory_manifest.json", trajectory_manifest)
    atomic_csv(output / "trajectories.csv", trajectories, TRAJECTORY_FIELDS)
    atomic_csv(output / "excursions_per_trade.csv", excursions, EXCURSION_FIELDS)
    atomic_json(output / "excursion_summary.json", excursion_summary)
    atomic_json(output / "temporal_exit_results.json", temporal)
    atomic_json(output / "fold_results.json", fold_results)
    atomic_json(output / "outlier_sensitivity.json", outliers)
    atomic_json(output / "diagnostic_summary.json", summary)
    _atomic_bytes(
        output / "diagnostic_summary.md",
        diagnostic_markdown(summary, excursion_summary, temporal, outliers, quality).encode("utf-8"),
    )
    hashes = {name: sha256_file(output / name) for name in SCIENTIFIC_FILES if name != "scientific_aggregate.json"}
    aggregate = {
        "schema_version": "aegis-exit-d1a-scientific-aggregate-v1",
        "preregistration_sha256": sha256_file(preregistration_path), "artifact_sha256": hashes,
        "aggregate_hash": Sha256HashProvider().digest_value(hashes),
    }
    atomic_json(output / "scientific_aggregate.json", aggregate)
    return {"summary": summary, "aggregate": aggregate, "quality": quality}


def finalize_attempts(repository: Path, preregistration_path: Path, root: Path) -> Mapping[str, Any]:
    attempts = (root / "attempt_1", root / "attempt_2")
    comparisons = {}
    for name in SCIENTIFIC_FILES:
        hashes = [sha256_file(path / name) for path in attempts]
        comparisons[name] = {"attempt_1": hashes[0], "attempt_2": hashes[1], "byte_identical": hashes[0] == hashes[1]}
    if not all(item["byte_identical"] for item in comparisons.values()):
        raise NondeterministicError(canonical_json(comparisons))
    root.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(preregistration_path, root / "preregistration.json")
    for name in SCIENTIFIC_FILES:
        shutil.copyfile(attempts[0] / name, root / name)
    report = {
        "schema_version": "aegis-exit-d1a-determinism-v1", "tolerance": 1e-12,
        "scientific_artifacts_byte_identical": True, "comparisons": comparisons,
        "aggregate_hash": json.loads((root / "scientific_aggregate.json").read_text())["aggregate_hash"],
    }
    atomic_json(root / "determinism_report.json", report)
    summary = json.loads((root / "diagnostic_summary.json").read_text())
    return {"determinism": report, "summary": summary}


def write_run_manifest(
    path: Path, *, attempt: int, repository: Path, preregistration_path: Path, result: Mapping[str, Any],
) -> None:
    import subprocess
    commit = subprocess.run(
        ("git", "rev-parse", "HEAD"), cwd=repository, check=True, capture_output=True, text=True,
    ).stdout.strip()
    atomic_json(path, {
        "schema_version": "aegis-exit-d1a-run-manifest-v1", "attempt": attempt,
        "completed_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "code_commit": commit, "preregistration_sha256": sha256_file(preregistration_path),
        "scientific_aggregate_hash": result["aggregate"]["aggregate_hash"],
        "technical_status": result["summary"]["technical_status"],
    })
