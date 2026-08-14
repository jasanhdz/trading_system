#!/usr/bin/env python3
"""Evaluate W2 on TRAIN/VALIDATION while keeping the final holdout sealed."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.frozen import FrozenEstimator
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from aegis.research.momentum_exhaustion_w2 import (
    FEATURE_COLUMNS,
    PolicyResult,
    simulate_exit_at_bar,
    simulate_policy,
)
from aegis.research.volume_wave_w1 import benjamini_hochberg
from aegis.utils import sha256_file


def _safe(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        value = float(value)
    if isinstance(value, float) and not math.isfinite(value):
        return "INF" if value > 0 else "-INF" if value < 0 else "NAN"
    if isinstance(value, dict):
        return {str(key): _safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_safe(item) for item in value]
    return value


def _gate_key(gate: float) -> str:
    return f"gate_{int(round(gate * 100)):03d}"


def _eligible_before_hard_stop(episode: pd.Series, gate: float) -> bool:
    side = str(episode.side)
    entry = float(episode.simulated_entry)
    atr = float(episode.entry_atr)
    highs = np.asarray(episode.path_high, dtype=float)
    lows = np.asarray(episode.path_low, dtype=float)
    favorable = highs - entry if side == "LONG" else entry - lows
    adverse = entry - lows if side == "LONG" else highs - entry
    gate_hits = np.flatnonzero(np.maximum.accumulate(favorable) >= gate * atr)
    stop_hits = np.flatnonzero(adverse >= 0.02 * entry)
    if not len(gate_hits):
        return False
    return not len(stop_hits) or int(gate_hits[0]) < int(stop_hits[0])


def _result_record(
    episode: pd.Series, result: PolicyResult, *, policy: str, parameter: float
) -> dict[str, Any]:
    closes = np.asarray(episode.path_close, dtype=float)
    side = str(episode.side)
    sign = 1.0 if side == "LONG" else -1.0
    entry = float(episode.simulated_entry)
    later_best = sign * (closes[result.exit_bar:] - entry) / entry
    missed = max(0.0, float(later_best.max(initial=result.gross_return)) - result.gross_return)
    return {
        "position_episode_id": episode.position_episode_id,
        "symbol": episode.symbol,
        "side": side,
        "entry_timestamp_ms": int(episode.entry_timestamp_ms),
        "policy": policy,
        "parameter": float(parameter),
        "exit_bar": result.exit_bar,
        "gross_return": result.gross_return,
        "net_return": result.net_return,
        "peak_mfe": result.peak_mfe,
        "mae": result.mae,
        "profit_capture_ratio": result.profit_capture_ratio,
        "final_giveback": result.final_giveback,
        "exit_reason": result.exit_reason,
        "exit_too_early": missed >= 0.25 * float(episode.entry_atr) / entry,
        "exit_too_late": result.peak_mfe > 0
        and result.final_giveback / result.peak_mfe >= 0.40,
    }


def _simulate_baseline(
    episodes: pd.DataFrame,
    definition: tuple[str, float],
    *,
    gate: float,
    cost_bps: float,
) -> pd.DataFrame:
    policy, parameter = definition
    records = []
    for _, episode in episodes.iterrows():
        result = simulate_policy(
            episode,
            policy=policy,
            parameter=parameter,
            gate_atr=gate,
            cost_bps=cost_bps,
        )
        records.append(_result_record(
            episode, result, policy=policy, parameter=parameter
        ))
    return pd.DataFrame(records)


def _simulate_requested_exits(
    episodes: pd.DataFrame,
    requested: dict[str, int],
    *,
    policy: str,
    parameter: float,
    cost_bps: float,
) -> pd.DataFrame:
    records = []
    for _, episode in episodes.iterrows():
        result = simulate_exit_at_bar(
            episode,
            requested_exit_bar=requested.get(str(episode.position_episode_id)),
            cost_bps=cost_bps,
            reason=policy,
        )
        records.append(_result_record(
            episode, result, policy=policy, parameter=parameter
        ))
    return pd.DataFrame(records)


def _metrics(rows: pd.DataFrame) -> dict[str, Any]:
    returns = rows["net_return"].to_numpy(dtype=float)
    gains = np.clip(returns, 0.0, None).sum()
    losses = -np.clip(returns, None, 0.0).sum()
    ordered = rows.sort_values("entry_timestamp_ms")
    equity = ordered["net_return"].cumsum().to_numpy(dtype=float)
    drawdown = np.maximum.accumulate(np.r_[0.0, equity])[1:] - equity
    downside = returns[returns < 0.0]
    maximum_drawdown = float(drawdown.max(initial=0.0))
    symbol_metrics = rows.groupby("symbol", sort=True).agg(
        net_expectancy=("net_return", "mean"),
        median_capture=("profit_capture_ratio", "median"),
        episodes=("position_episode_id", "size"),
    )
    return {
        "episodes": int(len(rows)),
        "net_expectancy": float(returns.mean()),
        "gross_expectancy": float(rows["gross_return"].mean()),
        "profit_factor": float(gains / losses) if losses > 0 else math.inf,
        "median_profit_capture_ratio": float(rows["profit_capture_ratio"].median()),
        "mean_profit_capture_ratio": float(rows["profit_capture_ratio"].mean()),
        "median_final_giveback": float(rows["final_giveback"].median()),
        "mean_mae": float(rows["mae"].mean()),
        "maximum_additive_drawdown": maximum_drawdown,
        "sortino_ratio": float(returns.mean() / downside.std(ddof=0))
        if len(downside) and downside.std(ddof=0) > 0 else 0.0,
        "calmar_proxy": float(returns.sum() / maximum_drawdown)
        if maximum_drawdown > 0 else 0.0,
        "expected_shortfall_05": float(np.sort(returns)[:max(1, len(returns) // 20)].mean()),
        "tail_giveback_p95": float(rows["final_giveback"].quantile(0.95)),
        "exit_too_early_rate": float(rows["exit_too_early"].mean()),
        "exit_too_late_rate": float(rows["exit_too_late"].mean()),
        "maximum_symbol_share": float(symbol_metrics["episodes"].max() / len(rows)),
        "symbol_metrics": symbol_metrics.to_dict(orient="index"),
    }


def _baseline_definitions(config: dict[str, Any]) -> list[tuple[str, float]]:
    baselines = config["baselines"]
    result = [("BOUNDED_HOLD", 0.0)]
    result.extend(("FIXED_TP", float(x)) for x in baselines["B_FIXED_TP_ATR"])
    result.extend(("FIXED_TRAILING", float(x)) for x in baselines["C_FIXED_TRAILING_ATR"])
    result.extend(("PERCENT_GIVEBACK", float(x)) for x in baselines["D_PEAK_GIVEBACK_FRACTION"])
    result.extend(("TIME_EXIT", float(x)) for x in baselines["E_TIME_EXIT_BARS_AFTER_GATE"])
    result.extend((name, 0.0) for name in (
        "CURRENT_ATR_TRAILING", "CURRENT_BE_PROTECTION",
    ))
    return result


def _interpretable_score(rows: pd.DataFrame) -> np.ndarray:
    return (
        20 * rows["giveback_ratio"].ge(0.30).to_numpy(dtype=int)
        + 15 * rows["bars_since_peak"].ge(2).to_numpy(dtype=int)
        + 15 * rows["directional_velocity_1"].le(0.0).to_numpy(dtype=int)
        + 15 * rows["taker_imbalance_decay"].lt(0.0).to_numpy(dtype=int)
        + 10 * rows["opposite_body_ratio"].ge(0.60).to_numpy(dtype=int)
        + 10 * rows["volume_over_4"].to_numpy(dtype=int)
        + 10 * rows["structure_deterioration"].gt(0.0).to_numpy(dtype=int)
        + 5 * rows["btc_opposes_position"].gt(0.0).to_numpy(dtype=int)
    ).astype(float)


def _pipeline(model: Any, *, scale: bool) -> Pipeline:
    steps: list[tuple[str, Any]] = [("imputer", SimpleImputer(strategy="median"))]
    if scale:
        steps.append(("scale", StandardScaler()))
    steps.append(("model", model))
    return Pipeline(steps)


def _fit_models(
    train: pd.DataFrame, config: dict[str, Any], *, side: str
) -> tuple[dict[str, Any], dict[str, Any]]:
    ordered = train.sort_values("evaluation_timestamp_ms").copy()
    cutoff = ordered["evaluation_timestamp_ms"].quantile(
        float(config["models"]["temporal_fit_fraction"])
    )
    fit = ordered.loc[ordered["evaluation_timestamp_ms"].le(cutoff)]
    calibration = ordered.loc[ordered["evaluation_timestamp_ms"].gt(cutoff)]
    counts = fit.groupby("position_episode_id")["position_episode_id"].transform("size")
    weights = 1.0 / counts.to_numpy(dtype=float)
    x_fit = fit.loc[:, FEATURE_COLUMNS].replace([np.inf, -np.inf], np.nan)
    x_cal = calibration.loc[:, FEATURE_COLUMNS].replace([np.inf, -np.inf], np.nan)
    y_primary = fit["target_giveback_before_new_extreme"].astype(int)
    y_cal_primary = calibration["target_giveback_before_new_extreme"].astype(int)
    y_hazard = fit["target_giveback_025_atr_next_1"].astype(int)
    y_cal_hazard = calibration["target_giveback_025_atr_next_1"].astype(int)
    definitions = {
        "LOGISTIC_L2": (
            _pipeline(LogisticRegression(
                C=float(config["models"]["logistic_l2"]["C"]),
                max_iter=int(config["models"]["logistic_l2"]["max_iter"]),
                random_state=182001,
            ), scale=True), y_primary, y_cal_primary,
        ),
        "HIST_GRADIENT_BOOSTING": (
            _pipeline(HistGradientBoostingClassifier(
                learning_rate=float(config["models"]["hist_gradient_boosting"]["learning_rate"]),
                max_iter=int(config["models"]["hist_gradient_boosting"]["max_iter"]),
                max_depth=int(config["models"]["hist_gradient_boosting"]["max_depth"]),
                l2_regularization=float(config["models"]["hist_gradient_boosting"]["l2_regularization"]),
                random_state=182001,
            ), scale=False), y_primary, y_cal_primary,
        ),
        "RANDOM_FOREST": (
            _pipeline(RandomForestClassifier(
                n_estimators=int(config["models"]["random_forest"]["n_estimators"]),
                max_depth=int(config["models"]["random_forest"]["max_depth"]),
                min_samples_leaf=int(config["models"]["random_forest"]["min_samples_leaf"]),
                max_features=str(config["models"]["random_forest"]["max_features"]),
                n_jobs=-1,
                random_state=182001,
            ), scale=False), y_primary, y_cal_primary,
        ),
        "DISCRETE_TIME_HAZARD_LOGISTIC": (
            _pipeline(LogisticRegression(
                C=float(config["models"]["logistic_l2"]["C"]),
                max_iter=int(config["models"]["logistic_l2"]["max_iter"]),
                random_state=182001,
            ), scale=True), y_hazard, y_cal_hazard,
        ),
    }
    models: dict[str, Any] = {}
    metadata: dict[str, Any] = {}
    for name, (base, y_fit, y_cal) in definitions.items():
        base.fit(x_fit, y_fit, model__sample_weight=weights)
        calibrated = CalibratedClassifierCV(FrozenEstimator(base), method="sigmoid")
        calibrated.fit(x_cal, y_cal)
        probability = calibrated.predict_proba(x_cal)[:, 1]
        metadata[name] = {
            "side": side,
            "fit_rows": int(len(fit)),
            "calibration_rows": int(len(calibration)),
            "positive_rate_fit": float(y_fit.mean()),
            "positive_rate_calibration": float(y_cal.mean()),
            "brier_calibration": float(brier_score_loss(y_cal, probability)),
            "log_loss_calibration": float(log_loss(y_cal, probability)),
            "roc_auc_calibration": float(roc_auc_score(y_cal, probability)),
            "pr_auc_calibration": float(average_precision_score(y_cal, probability)),
        }
        models[name] = calibrated
    return models, metadata


def _prediction_metrics(target: pd.Series, probability: np.ndarray) -> dict[str, float]:
    y = target.astype(int).to_numpy()
    return {
        "events": int(len(y)),
        "positive_rate": float(y.mean()),
        "brier": float(brier_score_loss(y, probability)),
        "log_loss": float(log_loss(y, probability)),
        "roc_auc": float(roc_auc_score(y, probability)),
        "pr_auc": float(average_precision_score(y, probability)),
    }


def _requested_exit_map(
    rows: pd.DataFrame, values: np.ndarray, threshold: float
) -> dict[str, int]:
    selected = rows.assign(_value=values).loc[lambda frame: frame["_value"].ge(threshold)]
    first = selected.sort_values("bar_index").groupby(
        "position_episode_id", sort=False
    )["bar_index"].first()
    return {str(key): int(value) for key, value in first.items()}


def _paired_bootstrap(
    challenger: pd.DataFrame,
    baseline: pd.DataFrame,
    *,
    repetitions: int,
    seed: int,
) -> dict[str, Any]:
    joined = challenger.merge(
        baseline[["position_episode_id", "net_return", "profit_capture_ratio"]],
        on="position_episode_id", suffixes=("_w2", "_baseline"), validate="one_to_one",
    )
    joined["day"] = pd.to_datetime(
        joined["entry_timestamp_ms"], unit="ms", utc=True
    ).dt.floor("1D")
    daily = joined.groupby("day", sort=True).agg(
        capture_delta=("profit_capture_ratio_w2", "median"),
        baseline_capture=("profit_capture_ratio_baseline", "median"),
        net_w2=("net_return_w2", "mean"),
        net_baseline=("net_return_baseline", "mean"),
    )
    capture_delta = (
        daily["capture_delta"] - daily["baseline_capture"]
    ).to_numpy(dtype=float)
    net_delta = (daily["net_w2"] - daily["net_baseline"]).to_numpy(dtype=float)
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(daily), size=(repetitions, len(daily)))
    capture_samples = capture_delta[indices].mean(axis=1)
    net_samples = net_delta[indices].mean(axis=1)
    return {
        "cluster_days": int(len(daily)),
        "median_capture_delta": float(
            challenger["profit_capture_ratio"].median()
            - baseline["profit_capture_ratio"].median()
        ),
        "mean_net_delta": float(
            challenger["net_return"].mean() - baseline["net_return"].mean()
        ),
        "capture_delta_ci_95": np.quantile(capture_samples, [0.025, 0.975]).tolist(),
        "net_delta_ci_95": np.quantile(net_samples, [0.025, 0.975]).tolist(),
        "p_capture_delta_le_zero": float((capture_samples <= 0.0).mean()),
        "posterior_probability_capture_positive": float((capture_samples > 0.0).mean()),
        "posterior_probability_net_positive": float((net_samples > 0.0).mean()),
    }


def _temporal_fold_count(
    challenger: pd.DataFrame, baseline: pd.DataFrame
) -> dict[str, Any]:
    joined = challenger.merge(
        baseline[["position_episode_id", "net_return", "profit_capture_ratio"]],
        on="position_episode_id", suffixes=("_w2", "_baseline"), validate="one_to_one",
    ).sort_values("entry_timestamp_ms")
    folds = []
    for index, positions in enumerate(np.array_split(np.arange(len(joined)), 4), start=1):
        rows = joined.iloc[positions]
        capture = float(
            rows["profit_capture_ratio_w2"].median()
            - rows["profit_capture_ratio_baseline"].median()
        )
        net = float((rows["net_return_w2"] - rows["net_return_baseline"]).mean())
        folds.append({
            "fold": index, "episodes": int(len(rows)),
            "capture_delta": capture, "net_delta": net,
            "positive": capture > 0.0 and net >= -0.0002,
        })
    return {
        "classification": "FROZEN_POLICY_TEMPORAL_VALIDATION_NOT_REFIT_WALK_FORWARD",
        "folds": folds,
        "positive_folds": sum(row["positive"] for row in folds),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-root", type=Path,
        default=Path("data/momentum_exhaustion_w2/dataset_train_validation_01"),
    )
    parser.add_argument(
        "--config", type=Path,
        default=Path("config/experiments/aegis_momentum_exhaustion_w2.yaml"),
    )
    parser.add_argument(
        "--output", type=Path,
        default=Path("data/momentum_exhaustion_w2/evaluation_train_validation_01.json"),
    )
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    resolve = lambda value: value if value.is_absolute() else root / value
    dataset_root = resolve(args.dataset_root)
    config_path = resolve(args.config)
    output = resolve(args.output)
    config = yaml.safe_load(config_path.read_text())
    manifest_path = dataset_root / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if manifest["final_holdout_state"] != "SEALED" or manifest["final_holdout_outcomes_read"]:
        raise RuntimeError("AEGIS_W2_HOLDOUT_NOT_SEALED")
    episodes = pd.concat([
        pd.read_parquet(path) for path in sorted(dataset_root.glob("*_episodes.parquet"))
    ], ignore_index=True)
    cost = float(config["economics"]["base_round_trip_cost_bps"])
    baseline_defs = _baseline_definitions(config)
    all_results: dict[str, Any] = {}
    pvalues: dict[str, float] = {}
    model_metadata: dict[str, Any] = {}

    decision_columns = [
        "position_episode_id", "partition", "symbol", "side",
        "evaluation_timestamp_ms", "bar_index", "volume_over_4",
        "target_giveback_before_new_extreme",
        "target_giveback_025_atr_next_1",
        *(_gate_key(float(gate)) for gate in config["profit_activation_gates_atr"]),
        *FEATURE_COLUMNS,
    ]
    for side in ("LONG", "SHORT"):
        decisions = pd.concat([
            pd.read_parquet(path, columns=decision_columns).loc[lambda frame: frame["side"].eq(side)]
            for path in sorted(dataset_root.glob("*_decisions.parquet"))
        ], ignore_index=True)
        train_decisions = decisions.loc[decisions["partition"].eq("TRAIN")].copy()
        validation_decisions = decisions.loc[decisions["partition"].eq("VALIDATION")].copy()
        models, fit_metadata = _fit_models(train_decisions, config, side=side)
        model_metadata.update({f"{side}:{key}": value for key, value in fit_metadata.items()})
        predictions: dict[str, tuple[np.ndarray, np.ndarray]] = {
            "INTERPRETABLE_SCORE": (
                _interpretable_score(train_decisions),
                _interpretable_score(validation_decisions),
            )
        }
        prediction_validation = {}
        for name, model in models.items():
            train_probability = model.predict_proba(
                train_decisions.loc[:, FEATURE_COLUMNS].replace([np.inf, -np.inf], np.nan)
            )[:, 1]
            validation_probability = model.predict_proba(
                validation_decisions.loc[:, FEATURE_COLUMNS].replace([np.inf, -np.inf], np.nan)
            )[:, 1]
            predictions[name] = (train_probability, validation_probability)
            target = (
                validation_decisions["target_giveback_025_atr_next_1"]
                if name == "DISCRETE_TIME_HAZARD_LOGISTIC"
                else validation_decisions["target_giveback_before_new_extreme"]
            )
            prediction_validation[name] = _prediction_metrics(target, validation_probability)
        model_metadata[f"{side}:VALIDATION"] = prediction_validation

        for gate in map(float, config["profit_activation_gates_atr"]):
            identity = f"{side}:GATE_{int(gate * 100):03d}"
            side_episodes = episodes.loc[
                episodes["side"].eq(side)
                & episodes.apply(lambda row: _eligible_before_hard_stop(row, gate), axis=1)
            ]
            train_episodes = side_episodes.loc[side_episodes["partition"].eq("TRAIN")]
            validation_episodes = side_episodes.loc[side_episodes["partition"].eq("VALIDATION")]
            train_baselines = {}
            validation_baselines = {}
            for definition in baseline_defs:
                key = f"{definition[0]}:{definition[1]:g}"
                train_rows = _simulate_baseline(
                    train_episodes, definition, gate=gate, cost_bps=cost
                )
                validation_rows = _simulate_baseline(
                    validation_episodes, definition, gate=gate, cost_bps=cost
                )
                train_baselines[key] = {"rows": train_rows, "metrics": _metrics(train_rows)}
                validation_baselines[key] = {"rows": validation_rows, "metrics": _metrics(validation_rows)}
            best_baseline_key = max(
                train_baselines,
                key=lambda key: (
                    train_baselines[key]["metrics"]["net_expectancy"],
                    train_baselines[key]["metrics"]["median_profit_capture_ratio"],
                    key,
                ),
            )
            best_train = train_baselines[best_baseline_key]["metrics"]
            train_gate_rows = train_decisions.loc[
                train_decisions[_gate_key(gate)]
                & train_decisions["position_episode_id"].isin(train_episodes["position_episode_id"])
            ]
            validation_gate_rows = validation_decisions.loc[
                validation_decisions[_gate_key(gate)]
                & validation_decisions["position_episode_id"].isin(validation_episodes["position_episode_id"])
            ]
            candidates = []
            for family, (train_values, validation_values) in predictions.items():
                thresholds = (
                    config["models"]["interpretable_score"]["thresholds"]
                    if family == "INTERPRETABLE_SCORE"
                    else config["models"]["policy_probability_thresholds"]
                )
                train_positions = train_decisions.index.get_indexer(train_gate_rows.index)
                validation_positions = validation_decisions.index.get_indexer(validation_gate_rows.index)
                if (train_positions < 0).any() or (validation_positions < 0).any():
                    raise RuntimeError("AEGIS_W2_PREDICTION_ALIGNMENT_FAILED")
                for threshold in map(float, thresholds):
                    train_exit = _requested_exit_map(
                        train_gate_rows, train_values[train_positions], threshold
                    )
                    validation_exit = _requested_exit_map(
                        validation_gate_rows, validation_values[validation_positions], threshold
                    )
                    train_rows = _simulate_requested_exits(
                        train_episodes, train_exit, policy=family,
                        parameter=threshold, cost_bps=cost,
                    )
                    validation_rows = _simulate_requested_exits(
                        validation_episodes, validation_exit, policy=family,
                        parameter=threshold, cost_bps=cost,
                    )
                    train_metrics = _metrics(train_rows)
                    candidates.append({
                        "family": family,
                        "threshold": threshold,
                        "train_rows": train_rows,
                        "validation_rows": validation_rows,
                        "train_metrics": train_metrics,
                        "eligible": train_metrics["net_expectancy"]
                        >= best_train["net_expectancy"] - 0.0002,
                    })
            eligible = [row for row in candidates if row["eligible"]]
            selection_pool = eligible if eligible else candidates
            selected = max(selection_pool, key=lambda row: (
                row["train_metrics"]["median_profit_capture_ratio"],
                row["train_metrics"]["net_expectancy"],
                row["family"], -row["threshold"],
            ))
            validation_metrics = _metrics(selected["validation_rows"])
            baseline_validation_rows = validation_baselines[best_baseline_key]["rows"]
            paired = _paired_bootstrap(
                selected["validation_rows"], baseline_validation_rows,
                repetitions=int(config["statistics"]["bootstrap_repetitions"]),
                seed=182001 + int(gate * 100) + (0 if side == "LONG" else 1000),
            )
            temporal = _temporal_fold_count(
                selected["validation_rows"], baseline_validation_rows
            )
            positive_symbols = 0
            for symbol in sorted(validation_episodes["symbol"].unique()):
                challenger_symbol = selected["validation_rows"].loc[
                    selected["validation_rows"]["symbol"].eq(symbol)
                ]
                baseline_symbol = baseline_validation_rows.loc[
                    baseline_validation_rows["symbol"].eq(symbol)
                ]
                if (
                    challenger_symbol["profit_capture_ratio"].median()
                    > baseline_symbol["profit_capture_ratio"].median()
                    and challenger_symbol["net_return"].mean()
                    >= baseline_symbol["net_return"].mean() - 0.0002
                ):
                    positive_symbols += 1
            stress = {}
            for stress_cost in config["economics"]["stress_round_trip_cost_bps"]:
                stress[str(stress_cost)] = float(
                    validation_metrics["gross_expectancy"] - float(stress_cost) / 10_000.0
                )
            blockers = []
            if len(validation_episodes) < int(config["statistics"]["minimum_validation_episodes_per_side"]):
                blockers.append("VALIDATION_EPISODES_INSUFFICIENT")
            if paired["median_capture_delta"] < float(config["gate"]["minimum_median_profit_capture_improvement"]):
                blockers.append("CAPTURE_EFFECT_BELOW_MINIMUM")
            if paired["capture_delta_ci_95"][0] <= 0.0:
                blockers.append("CAPTURE_CI_LOWER_NOT_POSITIVE")
            if paired["net_delta_ci_95"][0] < -0.0002:
                blockers.append("NET_NONINFERIORITY_CI_FAILED")
            if stress["20"] <= 0.0:
                blockers.append("FAILS_20BPS_COST")
            if positive_symbols < int(config["gate"]["minimum_positive_symbols"]):
                blockers.append("POSITIVE_SYMBOLS_INSUFFICIENT")
            if temporal["positive_folds"] < int(config["gate"]["minimum_positive_walk_forward_folds"]):
                blockers.append("TEMPORAL_FOLDS_INSUFFICIENT")
            blockers.append("EXPANDING_REFIT_WALK_FORWARD_NOT_RUN")
            pvalues[identity] = float(paired["p_capture_delta_le_zero"])
            all_results[identity] = {
                "state": "EVALUATED",
                "side": side,
                "gate_atr": gate,
                "train_episodes": int(len(train_episodes)),
                "validation_episodes": int(len(validation_episodes)),
                "best_train_baseline": best_baseline_key,
                "best_train_baseline_metrics": best_train,
                "validation_baseline_metrics": validation_baselines[best_baseline_key]["metrics"],
                "selected_family": selected["family"],
                "selected_threshold": selected["threshold"],
                "selected_train_eligible": selected["eligible"],
                "selected_train_metrics": selected["train_metrics"],
                "validation_metrics": validation_metrics,
                "paired_comparison": paired,
                "temporal_validation": temporal,
                "positive_symbols": positive_symbols,
                "stress_expectancy": stress,
                "gate_blockers": blockers,
                "gate_pass_before_fdr": not blockers,
                "baseline_summary": {
                    key: value["metrics"] for key, value in validation_baselines.items()
                },
            }
            print(json.dumps({"hypothesis": identity, "state": "EVALUATED"}), flush=True)
        del decisions, train_decisions, validation_decisions, models, predictions
    fdr = benjamini_hochberg(
        pvalues,
        false_discovery_rate=float(config["statistics"]["fdr_alpha"]),
    )
    passing = []
    for identity, result in all_results.items():
        result["fdr_significant"] = bool(fdr[identity])
        if not fdr[identity]:
            result["gate_blockers"].append("FDR_NOT_SIGNIFICANT")
        result["gate_pass"] = not result["gate_blockers"]
        if result["gate_pass"]:
            passing.append(identity)
    verdict = {
        "schema_version": "aegis-momentum-exhaustion-w2-evaluation-v1",
        "config_sha256": sha256_file(config_path),
        "dataset_manifest_sha256": sha256_file(manifest_path),
        "final_holdout_state": "SEALED",
        "final_holdout_outcomes_read": False,
        "results": all_results,
        "model_metadata": model_metadata,
        "fdr": fdr,
        "passing_hypotheses": passing,
        "W2_RULE_EDGE_FOUND": bool(passing),
        "W2_MODELING_JUSTIFIED": bool(passing),
        "W2_READY_FOR_SHADOW": False,
        "W2_READY_FOR_LIVE": False,
        "authenticated_requests": 0,
        "exchange_mutations": 0,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(_safe(verdict), indent=2, sort_keys=True) + "\n")
    os.chmod(output, 0o600)
    print(json.dumps({
        "output": str(output), "passing": passing, "holdout": "SEALED",
    }))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
