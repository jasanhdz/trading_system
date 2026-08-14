#!/usr/bin/env python3
"""Evaluate preregistered W3A entry and W3B exit timing on TRAIN/VALIDATION."""

from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, precision_recall_curve, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from aegis.research.intrabar_wave_w3 import (
    W3_FEATURE_COLUMNS,
    assert_probability_contract,
    episode_weights,
    profit_capture_ratio,
    safe_ratio,
    summarize_returns,
)
from aegis.utils import sha256_file


EXIT_EXTRA_FEATURES = (
    "current_favorable_return", "peak_mfe", "giveback", "giveback_ratio",
    "minutes_since_peak", "episode_minute",
)


@dataclass
class CalibratedModel:
    estimator: Any
    calibrator: LogisticRegression
    feature_columns: tuple[str, ...]

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        raw = self.estimator.predict_proba(frame[list(self.feature_columns)])[:, 1]
        values = self.calibrator.predict_proba(raw.reshape(-1, 1))[:, 1]
        assert_probability_contract(values)
        return values


def _fit_models(
    rows: pd.DataFrame, features: tuple[str, ...], target: str, config: dict[str, Any]
) -> tuple[dict[str, CalibratedModel], pd.DataFrame, dict[str, Any]]:
    ordered = rows.sort_values("decision_time_ms").copy()
    cutoff_index = max(1, min(len(ordered) - 1, int(len(ordered) * float(config["models"]["temporal_fit_fraction"]))))
    fit, calibration = ordered.iloc[:cutoff_index], ordered.iloc[cutoff_index:]
    weights = episode_weights(fit)
    logistic_cfg = config["models"]["logistic_l2"]
    hgb_cfg = config["models"]["hist_gradient_boosting"]
    estimators = {
        "LOGISTIC_L2": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(
                C=float(logistic_cfg["C"]), max_iter=int(logistic_cfg["max_iter"]),
                random_state=int(config["models"]["random_seed"]),
            )),
        ]),
        "HIST_GRADIENT_BOOSTING": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", HistGradientBoostingClassifier(
                learning_rate=float(hgb_cfg["learning_rate"]),
                max_iter=int(hgb_cfg["max_iter"]), max_depth=int(hgb_cfg["max_depth"]),
                l2_regularization=float(hgb_cfg["l2_regularization"]),
                random_state=int(config["models"]["random_seed"]),
            )),
        ]),
    }
    models: dict[str, CalibratedModel] = {}
    metadata: dict[str, Any] = {}
    for name, estimator in estimators.items():
        kwargs = {"model__sample_weight": weights}
        estimator.fit(fit[list(features)], fit[target].astype(int), **kwargs)
        raw = estimator.predict_proba(calibration[list(features)])[:, 1]
        calibrator = LogisticRegression(C=1_000_000.0, max_iter=300)
        calibrator.fit(raw.reshape(-1, 1), calibration[target].astype(int), sample_weight=episode_weights(calibration))
        model = CalibratedModel(estimator, calibrator, features)
        probabilities = model.predict(calibration)
        metadata[name] = _calibration_metrics(calibration[target].to_numpy(dtype=int), probabilities)
        metadata[name].update({"fit_rows": len(fit), "calibration_rows": len(calibration)})
        models[name] = model
    return models, calibration, metadata


def _calibration_metrics(target: np.ndarray, probability: np.ndarray) -> dict[str, Any]:
    assert_probability_contract(probability)
    bins = np.minimum((probability * 10).astype(int), 9)
    ece = 0.0
    for bucket in range(10):
        mask = bins == bucket
        if mask.any():
            ece += mask.mean() * abs(probability[mask].mean() - target[mask].mean())
    precision, recall, _ = precision_recall_curve(target, probability)
    pr_auc = float(abs(np.trapezoid(precision, recall)))
    return {
        "rows": int(len(target)), "positive_rate": float(target.mean()),
        "brier": float(brier_score_loss(target, probability)),
        "log_loss": float(log_loss(target, probability, labels=[0, 1])),
        "roc_auc": float(roc_auc_score(target, probability)),
        "pr_auc": pr_auc, "expected_calibration_error": float(ece),
    }


def _costed(frame: pd.DataFrame, cost_bps: float) -> pd.DataFrame:
    result = frame.copy()
    result["net_return"] = result["gross_return"] - result["traded"].astype(float) * cost_bps / 10_000.0
    return result


def _entry_universe(rows: pd.DataFrame) -> pd.DataFrame:
    return rows.sort_values("decision_time_ms").drop_duplicates("wave_episode_id")[[
        "wave_episode_id", "symbol", "side", "partition", "impulse_close_time_ms",
    ]]


def _entry_policy(rows: pd.DataFrame, selector: pd.Series, cost_bps: float) -> pd.DataFrame:
    universe = _entry_universe(rows)
    selected = rows.loc[selector].sort_values(["wave_episode_id", "offset_minutes"]).drop_duplicates("wave_episode_id")
    selected = selected[["wave_episode_id", "primary_gross_return", "primary_mae_atr"]]
    result = universe.merge(selected, on="wave_episode_id", how="left")
    result["traded"] = result["primary_gross_return"].notna()
    result["gross_return"] = result["primary_gross_return"].fillna(0.0)
    result["mae_atr"] = result["primary_mae_atr"].fillna(0.0)
    return _costed(result, cost_bps)


def _entry_baselines(rows: pd.DataFrame, cost_bps: float) -> dict[str, pd.DataFrame]:
    return {
        "IMPULSE_CLOSE": _entry_policy(rows, rows["offset_minutes"].eq(0), cost_bps),
        "WAIT_1_MINUTE": _entry_policy(rows, rows["offset_minutes"].eq(1), cost_bps),
        "IMPULSE_EXTREME_BREAK": _entry_policy(
            rows, rows["offset_minutes"].gt(0) & rows["break_of_impulse_extreme"].gt(0), cost_bps
        ),
        "PULLBACK_025_TO_050": _entry_policy(
            rows, rows["offset_minutes"].gt(0) & rows["pullback_fraction"].between(0.25, 0.50), cost_bps
        ),
        "NO_TRADE": _entry_policy(rows, pd.Series(False, index=rows.index), cost_bps),
    }


def _episode_directional_path(row: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    highs = np.array([getattr(row, f"high_{index}") for index in range(1, 31)], dtype=float)
    lows = np.array([getattr(row, f"low_{index}") for index in range(1, 31)], dtype=float)
    closes = np.array([getattr(row, f"close_{index}") for index in range(1, 31)], dtype=float)
    if row.direction > 0:
        favorable_high = highs / row.entry_price - 1.0
        favorable_low = lows / row.entry_price - 1.0
        close_return = closes / row.entry_price - 1.0
    else:
        favorable_high = 1.0 - lows / row.entry_price
        favorable_low = 1.0 - highs / row.entry_price
        close_return = 1.0 - closes / row.entry_price
    return favorable_high, favorable_low, close_return


def _exit_baseline_rows(episodes: pd.DataFrame, cost_bps: float) -> dict[str, pd.DataFrame]:
    policies = {name: [] for name in (
        "BOUNDED_HOLD", "FIXED_TRAILING_025_ATR", "FIXED_TRAILING_050_ATR",
        "GIVEBACK_030", "GIVEBACK_050", "TIME_EXIT_3", "TIME_EXIT_5", "TIME_EXIT_10",
    )}
    for row in episodes.itertuples():
        favorable_high, favorable_low, close_return = _episode_directional_path(row)
        peak = np.maximum.accumulate(favorable_high)
        gate = int(row.gate_minute) - 1
        atr_fraction = row.atr / row.entry_price
        base = {
            "wave_episode_id": row.wave_episode_id, "symbol": row.symbol,
            "side": row.side, "partition": row.partition,
            "impulse_close_time_ms": row.impulse_close_time_ms,
            "traded": True, "mae_atr": row.mae * row.entry_price / row.atr,
            "peak_mfe": row.peak_mfe,
        }
        policies["BOUNDED_HOLD"].append({**base, "gross_return": float(close_return[-1])})
        for trail_atr in (0.25, 0.50):
            gross = float(close_return[-1])
            for index in range(gate, 30):
                threshold = peak[index] - trail_atr * atr_fraction
                if favorable_low[index] <= threshold:
                    gross = float(threshold)
                    break
            policies[f"FIXED_TRAILING_{int(trail_atr * 100):03d}_ATR"].append({**base, "gross_return": gross})
        for fraction in (0.30, 0.50):
            gross = float(close_return[-1])
            for index in range(gate, 30):
                threshold = peak[index] * (1.0 - fraction)
                if favorable_low[index] <= threshold:
                    gross = float(threshold)
                    break
            policies[f"GIVEBACK_{int(fraction * 100):03d}"].append({**base, "gross_return": gross})
        for delay in (3, 5, 10):
            index = min(29, gate + delay)
            policies[f"TIME_EXIT_{delay}"].append({**base, "gross_return": float(close_return[index])})
    result = {}
    for name, records in policies.items():
        frame = pd.DataFrame(records)
        frame["profit_capture_ratio"] = [
            profit_capture_ratio(gross, peak)
            for gross, peak in zip(frame.gross_return, frame.peak_mfe, strict=True)
        ]
        result[name] = _costed(frame, cost_bps)
    return result


def _exit_model_policy(
    decisions: pd.DataFrame, episodes: pd.DataFrame, selector: pd.Series, cost_bps: float
) -> pd.DataFrame:
    selected = decisions.loc[selector].sort_values(["wave_episode_id", "episode_minute"]).drop_duplicates("wave_episode_id")
    chosen = selected.set_index("wave_episode_id")["exit_execution_return"].to_dict()
    records = []
    for row in episodes.itertuples():
        _, _, close_return = _episode_directional_path(row)
        gross = float(chosen.get(row.wave_episode_id, close_return[-1]))
        records.append({
            "wave_episode_id": row.wave_episode_id, "symbol": row.symbol,
            "side": row.side, "partition": row.partition,
            "impulse_close_time_ms": row.impulse_close_time_ms,
            "traded": True, "gross_return": gross,
            "mae_atr": row.mae * row.entry_price / row.atr,
            "peak_mfe": row.peak_mfe,
            "profit_capture_ratio": profit_capture_ratio(gross, row.peak_mfe),
        })
    return _costed(pd.DataFrame(records), cost_bps)


def _paired_bootstrap(
    candidate: pd.DataFrame, baseline: pd.DataFrame, repetitions: int, seed: int
) -> dict[str, Any]:
    joined = candidate[["wave_episode_id", "net_return"]].merge(
        baseline[["wave_episode_id", "net_return"]], on="wave_episode_id", suffixes=("_candidate", "_baseline")
    ).merge(candidate[["wave_episode_id", "impulse_close_time_ms"]], on="wave_episode_id")
    joined["day"] = pd.to_datetime(joined.impulse_close_time_ms, unit="ms", utc=True).dt.floor("D")
    daily = joined.groupby("day").agg(
        candidate=("net_return_candidate", "mean"), baseline=("net_return_baseline", "mean")
    )
    delta = (daily.candidate - daily.baseline).to_numpy(dtype=float)
    rng = np.random.default_rng(seed)
    draws = rng.choice(delta, size=(repetitions, len(delta)), replace=True).mean(axis=1)
    result = {
        "days": len(delta), "mean_net_delta": float((joined.net_return_candidate - joined.net_return_baseline).mean()),
        "net_delta_ci_95": [float(value) for value in np.quantile(draws, [0.025, 0.975])],
        "posterior_probability_net_positive": float((draws > 0).mean()),
        "p_net_delta_le_zero": float((draws <= 0).mean()),
    }
    if "profit_capture_ratio" in candidate and "profit_capture_ratio" in baseline:
        capture = candidate[["wave_episode_id", "profit_capture_ratio"]].merge(
            baseline[["wave_episode_id", "profit_capture_ratio"]], on="wave_episode_id", suffixes=("_candidate", "_baseline")
        ).merge(candidate[["wave_episode_id", "impulse_close_time_ms"]], on="wave_episode_id")
        capture["day"] = pd.to_datetime(capture.impulse_close_time_ms, unit="ms", utc=True).dt.floor("D")
        daily_capture = capture.groupby("day").apply(
            lambda frame: float((frame.profit_capture_ratio_candidate - frame.profit_capture_ratio_baseline).median()),
            include_groups=False,
        ).to_numpy(dtype=float)
        capture_draws = rng.choice(daily_capture, size=(repetitions, len(daily_capture)), replace=True).mean(axis=1)
        result.update({
            "median_capture_delta": float((capture.profit_capture_ratio_candidate - capture.profit_capture_ratio_baseline).median()),
            "capture_delta_ci_95": [float(value) for value in np.quantile(capture_draws, [0.025, 0.975])],
            "posterior_probability_capture_positive": float((capture_draws > 0).mean()),
        })
    return result


def _temporal_folds(candidate: pd.DataFrame, baseline: pd.DataFrame, count: int) -> dict[str, Any]:
    baseline_columns = ["wave_episode_id", "net_return"]
    if "profit_capture_ratio" in baseline:
        baseline_columns.append("profit_capture_ratio")
    joined = candidate.merge(
        baseline[baseline_columns], on="wave_episode_id", suffixes=("_candidate", "_baseline")
    )
    joined.sort_values("impulse_close_time_ms", inplace=True)
    folds = []
    for number, indexes in enumerate(np.array_split(np.arange(len(joined)), count), 1):
        fold = joined.iloc[indexes]
        delta = float((fold.net_return_candidate - fold.net_return_baseline).mean())
        capture_delta = None
        if "profit_capture_ratio_candidate" in fold and "profit_capture_ratio_baseline" in fold:
            capture_delta = float((fold.profit_capture_ratio_candidate - fold.profit_capture_ratio_baseline).median())
        folds.append({"fold": number, "episodes": len(fold), "net_delta": delta, "capture_delta": capture_delta, "positive": delta > 0})
    return {"classification": "FROZEN_POLICY_TEMPORAL_VALIDATION", "folds": folds, "positive_folds": sum(item["positive"] for item in folds)}


def _positive_symbols(candidate: pd.DataFrame, baseline: pd.DataFrame) -> tuple[int, dict[str, float]]:
    joined = candidate.merge(baseline[["wave_episode_id", "net_return"]], on="wave_episode_id", suffixes=("_candidate", "_baseline"))
    deltas = joined.groupby("symbol").apply(
        lambda frame: float((frame.net_return_candidate - frame.net_return_baseline).mean()), include_groups=False
    ).to_dict()
    return sum(value > 0 for value in deltas.values()), deltas


def _bh_significant(p_values: list[float], alpha: float) -> list[bool]:
    order = np.argsort(p_values)
    largest = -1
    for rank, index in enumerate(order, 1):
        if p_values[index] <= alpha * rank / len(p_values):
            largest = rank
    threshold = alpha * largest / len(p_values) if largest > 0 else -1.0
    return [value <= threshold for value in p_values]


def _evaluate_study_side(
    study: str,
    side: str,
    decisions: pd.DataFrame,
    episodes: pd.DataFrame,
    config: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    cost = float(config["economics"]["base_round_trip_cost_bps"])
    target = "primary_outcome" if study == "W3A" else "target_giveback_before_new_extreme"
    features = W3_FEATURE_COLUMNS if study == "W3A" else (*W3_FEATURE_COLUMNS, *EXIT_EXTRA_FEATURES)
    decisions = decisions.loc[decisions.side.eq(side)].copy()
    decisions["target_binary"] = decisions[target].eq(1).astype(int)
    train = decisions.loc[decisions.partition.eq("TRAIN")]
    validation = decisions.loc[decisions.partition.eq("VALIDATION")]
    models, selection_rows, calibration = _fit_models(train, features, "target_binary", config)
    if study == "W3A":
        selection_episodes = _entry_universe(selection_rows)
        validation_episodes = _entry_universe(validation)
        train_baselines = _entry_baselines(selection_rows, cost)
        validation_baselines = _entry_baselines(validation, cost)
        thresholds = config["w3a_entry"]["probability_thresholds"]
        policy = lambda rows, universe, selector, c: _entry_policy(rows, selector, c)
    else:
        selection_ids = set(selection_rows.wave_episode_id)
        validation_ids = set(validation.wave_episode_id)
        selection_episodes = episodes.loc[episodes.wave_episode_id.isin(selection_ids) & episodes.gate_reached]
        validation_episodes = episodes.loc[episodes.wave_episode_id.isin(validation_ids) & episodes.gate_reached]
        train_baselines = _exit_baseline_rows(selection_episodes, cost)
        validation_baselines = _exit_baseline_rows(validation_episodes, cost)
        thresholds = config["w3b_exit"]["probability_thresholds"]
        policy = _exit_model_policy
    baseline_name, baseline_train = max(
        train_baselines.items(), key=lambda item: (summarize_returns(item[1])["net_expectancy"], item[0])
    )
    candidates = []
    validation_probabilities = {}
    for model_name, model in models.items():
        select_probability = model.predict(selection_rows)
        validation_probability = model.predict(validation)
        validation_probabilities[model_name] = validation_probability
        for threshold in thresholds:
            train_policy = policy(selection_rows, selection_episodes, pd.Series(select_probability >= threshold, index=selection_rows.index), cost)
            metrics = summarize_returns(train_policy)
            candidates.append((metrics["net_expectancy"], -metrics["mean_mae_atr"], model_name, float(threshold), train_policy, metrics))
    _, _, selected_model_name, selected_threshold, selected_train, selected_train_metrics = max(candidates)
    selected_model = models[selected_model_name]
    selected_probability = validation_probabilities[selected_model_name]
    selected_validation = policy(
        validation, validation_episodes,
        pd.Series(selected_probability >= selected_threshold, index=validation.index), cost,
    )
    baseline_validation = validation_baselines[baseline_name]
    selected_metrics = summarize_returns(selected_validation)
    baseline_metrics = summarize_returns(baseline_validation)
    bootstrap = _paired_bootstrap(
        selected_validation, baseline_validation, int(config["statistics"]["bootstrap_repetitions"]),
        int(config["models"]["random_seed"]) + (0 if side == "LONG" else 1) + (0 if study == "W3A" else 10),
    )
    temporal = _temporal_folds(selected_validation, baseline_validation, int(config["statistics"]["temporal_validation_folds"]))
    positive_symbols, symbol_deltas = _positive_symbols(selected_validation, baseline_validation)
    stress = {}
    for stress_cost in config["economics"]["stress_round_trip_cost_bps"]:
        stressed = _costed(selected_validation, float(stress_cost))
        stress[str(stress_cost)] = summarize_returns(stressed)["net_expectancy"]
    gate = config["w3a_gate"] if study == "W3A" else config["w3b_gate"]
    blockers = []
    improvement_bps = (selected_metrics["net_expectancy"] - baseline_metrics["net_expectancy"]) * 10_000.0
    if study == "W3A":
        checks = [
            (selected_metrics["net_expectancy"] * 10_000.0 >= float(gate["minimum_net_expectancy_bps"]), "NET_EXPECTANCY_BELOW_MINIMUM"),
            (improvement_bps >= float(gate["minimum_improvement_over_best_baseline_bps"]), "BASELINE_IMPROVEMENT_BELOW_MINIMUM"),
            (bootstrap["net_delta_ci_95"][0] > 0.0, "BOOTSTRAP_IMPROVEMENT_CI_FAILED"),
            (selected_metrics["profit_factor"] >= float(gate["minimum_profit_factor"]), "PROFIT_FACTOR_FAILED"),
            (selected_metrics["mean_mae_atr"] <= float(gate["maximum_mean_mae_atr"]), "MAE_GATE_FAILED"),
        ]
    else:
        capture_delta = selected_metrics["median_profit_capture_ratio"] - baseline_metrics["median_profit_capture_ratio"]
        checks = [
            (improvement_bps >= float(gate["minimum_net_improvement_over_best_baseline_bps"]), "NET_IMPROVEMENT_BELOW_MINIMUM"),
            (capture_delta >= float(gate["minimum_median_capture_improvement"]), "CAPTURE_IMPROVEMENT_BELOW_MINIMUM"),
            (bootstrap["net_delta_ci_95"][0] > 0.0, "BOOTSTRAP_NET_CI_FAILED"),
            (bootstrap.get("capture_delta_ci_95", [0.0])[0] > 0.0, "BOOTSTRAP_CAPTURE_CI_FAILED"),
        ]
    checks.extend([
        (positive_symbols >= int(gate["minimum_positive_symbols"]), "POSITIVE_SYMBOLS_INSUFFICIENT"),
        (temporal["positive_folds"] >= int(gate["minimum_positive_temporal_folds"]), "TEMPORAL_FOLDS_INSUFFICIENT"),
        (stress["20"] > 0.0, "FAILS_20BPS_COST"),
        (selected_metrics["maximum_symbol_share"] <= float(gate["maximum_symbol_share"]), "SYMBOL_CONCENTRATION_FAILED"),
    ])
    blockers.extend(reason for passed, reason in checks if not passed)
    validation_calibration = _calibration_metrics(validation.target_binary.to_numpy(dtype=int), selected_probability)
    return ({
        "study": study, "side": side,
        "train_decision_rows": len(train), "selection_decision_rows": len(selection_rows),
        "validation_decision_rows": len(validation), "validation_episodes": len(selected_validation),
        "selected_model": selected_model_name, "selected_threshold": selected_threshold,
        "best_train_baseline": baseline_name,
        "selected_train_metrics": selected_train_metrics,
        "validation_metrics": selected_metrics, "validation_baseline_metrics": baseline_metrics,
        "net_improvement_bps": improvement_bps,
        "bootstrap": bootstrap, "temporal_validation": temporal,
        "positive_symbols": positive_symbols, "per_symbol_net_delta": symbol_deltas,
        "cost_stress_net_expectancy": stress,
        "validation_calibration": validation_calibration,
        "gate_pass_before_fdr": not blockers, "gate_blockers": blockers,
    }, calibration)


def _write_private(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n")
    os.chmod(path, 0o600)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("config/experiments/aegis_intrabar_wave_w3.yaml"))
    parser.add_argument("--dataset-root", type=Path, default=Path("data/intrabar_wave_w3/dataset_train_validation_01"))
    parser.add_argument("--output-root", type=Path, default=Path("data/intrabar_wave_w3/evaluation_01"))
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    resolve = lambda value: value if value.is_absolute() else root / value
    config_path, dataset_root, output_root = map(resolve, (args.config, args.dataset_root, args.output_root))
    config = yaml.safe_load(config_path.read_text())
    manifest_path = dataset_root / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if manifest["final_holdout_state"] != "SEALED" or manifest["final_holdout_outcomes_read"]:
        raise RuntimeError("AEGIS_W3_HOLDOUT_CONTRACT_VIOLATION")
    entry = pd.concat([pd.read_parquet(path) for path in sorted(dataset_root.glob("*_entry.parquet"))], ignore_index=True)
    exit_decisions = pd.concat([pd.read_parquet(path) for path in sorted(dataset_root.glob("*_exit.parquet"))], ignore_index=True)
    episodes = pd.concat([pd.read_parquet(path) for path in sorted(dataset_root.glob("*_episodes.parquet"))], ignore_index=True)
    results = {}
    calibration = {}
    for study, decisions in (("W3A", entry), ("W3B", exit_decisions)):
        for side in ("LONG", "SHORT"):
            key = f"{study}:{side}"
            result, model_meta = _evaluate_study_side(study, side, decisions, episodes, config)
            results[key] = result
            calibration[key] = model_meta
            print(json.dumps({"hypothesis": key, "blockers": result["gate_blockers"]}), flush=True)
    p_values = [results[key]["bootstrap"]["p_net_delta_le_zero"] for key in sorted(results)]
    significant = _bh_significant(p_values, float(config["statistics"]["fdr_alpha"]))
    for key, accepted in zip(sorted(results), significant, strict=True):
        results[key]["fdr_significant"] = accepted
        if not accepted:
            results[key]["gate_blockers"].append("FDR_NOT_SIGNIFICANT")
        results[key]["gate_pass"] = results[key]["gate_pass_before_fdr"] and accepted
    w3a_pass = any(results[f"W3A:{side}"]["gate_pass"] for side in ("LONG", "SHORT"))
    w3b_pass = any(results[f"W3B:{side}"]["gate_pass"] for side in ("LONG", "SHORT"))
    verdict = {
        "schema_version": "aegis-intrabar-wave-w3-evaluation-v1",
        "config_sha256": sha256_file(config_path), "dataset_manifest_sha256": sha256_file(manifest_path),
        "results": results, "calibration": calibration,
        "W3A_ENTRY_EDGE_FOUND": w3a_pass, "W3A_MODELING_JUSTIFIED": w3a_pass,
        "W3A_READY_FOR_SHADOW": False, "W3A_READY_FOR_LIVE": False,
        "W3B_EXIT_EDGE_FOUND": w3b_pass, "W3B_MODELING_JUSTIFIED": w3b_pass,
        "W3B_READY_FOR_SHADOW": False, "W3B_READY_FOR_LIVE": False,
        "W3_INTRABAR_WAVE_EDGE_FOUND": w3a_pass or w3b_pass,
        "W3_READY_FOR_SHADOW": False, "W3_READY_FOR_LIVE": False,
        "final_holdout_state": "SEALED", "final_holdout_outcomes_read": False,
        "authenticated_requests": 0, "exchange_mutations": 0,
    }
    output_root.mkdir(parents=True, exist_ok=True)
    os.chmod(output_root, 0o700)
    _write_private(output_root / "evaluation.json", verdict)
    _write_private(output_root / "feature_dictionary.json", {"schema": config["features"]["feature_schema"], "entry": list(W3_FEATURE_COLUMNS), "exit_extra": list(EXIT_EXTRA_FEATURES)})
    _write_private(output_root / "hypothesis_registry.json", {key: {"pass": value["gate_pass"], "blockers": value["gate_blockers"]} for key, value in results.items()})
    _write_private(output_root / "fold_results.json", {key: value["temporal_validation"] for key, value in results.items()})
    _write_private(output_root / "per_symbol_results.json", {key: value["per_symbol_net_delta"] for key, value in results.items()})
    _write_private(output_root / "cost_stress_results.json", {key: value["cost_stress_net_expectancy"] for key, value in results.items()})
    _write_private(output_root / "bootstrap_results.json", {key: value["bootstrap"] for key, value in results.items()})
    _write_private(output_root / "calibration_results.json", {"selection": calibration, "validation": {key: value["validation_calibration"] for key, value in results.items()}})
    print(json.dumps({"evaluation": str(output_root / "evaluation.json"), "holdout": "SEALED"}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
