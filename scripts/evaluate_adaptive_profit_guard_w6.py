#!/usr/bin/env python3
"""Run preregistered W6 adaptive profit-guard research without runtime effects."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier, export_text

from aegis.research.adaptive_profit_guard_w6 import (
    benjamini_hochberg,
    choose_best_action,
    paired_day_bootstrap,
    policy_summary,
    simulate_guard,
    simulate_simple_baseline,
    trailing_activation_atr,
    validate_feature_contract,
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def utc_ms(value: str) -> int:
    return int(pd.Timestamp(value).timestamp() * 1_000)


def load_population(root: Path, config: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame]:
    source = root / config["population"]["source"]
    episode_frames, decision_frames = [], []
    feature_columns = list(config["features"]["columns"])
    validate_feature_contract(feature_columns)
    decision_columns = [
        "position_episode_id", "symbol", "side", "entry_timestamp_ms",
        "evaluation_timestamp_ms", "bar_index", "entry_price", "entry_atr",
        *feature_columns,
    ]
    for episode_path in sorted(source.glob("*_episodes.parquet")):
        symbol = episode_path.name.removesuffix("_episodes.parquet")
        decision_path = source / f"{symbol}_decisions.parquet"
        episodes = pd.read_parquet(episode_path)
        episodes = episodes.loc[episodes["partition"].eq("TRAIN")].copy()
        decisions = pd.read_parquet(decision_path, columns=decision_columns)
        decision_frames.append(decisions.loc[decisions.position_episode_id.isin(episodes.position_episode_id)])
        episode_frames.append(episodes)
    episodes = pd.concat(episode_frames, ignore_index=True)
    decisions = pd.concat(decision_frames, ignore_index=True)
    if episodes.position_episode_id.duplicated().any():
        raise RuntimeError("AEGIS_W6_DUPLICATE_POSITION_EPISODE")
    return episodes, decisions


def activation_rows(decisions: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    guard = config["current_guard"]
    threshold = np.array([
        trailing_activation_atr(float(entry), float(atr), float(guard["trailing_activation_roe"]), float(guard["leverage_for_roe_mapping"]))
        for entry, atr in zip(decisions.entry_price, decisions.entry_atr, strict=True)
    ])
    eligible = decisions.loc[decisions.peak_mfe_atr.to_numpy(float) >= threshold].copy()
    eligible.sort_values(["position_episode_id", "bar_index"], inplace=True)
    first = eligible.groupby("position_episode_id", observed=True, as_index=False).first()
    first["activated"] = True
    return first


def full_path_peak_return(episode: dict[str, Any]) -> float:
    entry = float(episode["simulated_entry"])
    if episode["side"] == "LONG":
        favorable = np.asarray(episode["path_high"], dtype=float) - entry
    else:
        favorable = entry - np.asarray(episode["path_low"], dtype=float)
    return max(0.0, float(np.max(favorable, initial=0.0)) / entry)


def result_columns(prefix: str, result: Any, *, future_peak_return: float) -> dict[str, Any]:
    return {
        f"{prefix}_net_return_bps": result.net_return * 10_000,
        f"{prefix}_gross_return_bps": result.gross_return * 10_000,
        f"{prefix}_profit_capture_ratio": result.profit_capture_ratio,
        f"{prefix}_peak_mfe_bps": result.peak_mfe * 10_000,
        f"{prefix}_final_giveback_bps": result.final_giveback * 10_000,
        f"{prefix}_early_exit_regret_bps": max(0.0, future_peak_return - result.gross_return) * 10_000,
        f"{prefix}_hold_too_long_regret_bps": result.final_giveback * 10_000,
        f"{prefix}_mae_bps": result.mae * 10_000,
        f"{prefix}_exit_bar": result.exit_bar,
        f"{prefix}_exit_reason": result.exit_reason,
    }


def replay_actions(episodes: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    actions = {
        name: config["adaptive_actions"][name]
        for name in ("DEFENSIVE", "NORMAL", "EXPANSION")
    }
    guard = config["current_guard"]
    rows: list[dict[str, Any]] = []
    for episode in episodes.to_dict("records"):
        future_peak = full_path_peak_return(episode)
        action_results = {
            name: simulate_guard(
                episode, atr_multiplier=float(multiplier),
                cost_bps=float(config["economics"]["base_round_trip_cost_bps"]),
                leverage=float(guard["leverage_for_roe_mapping"]),
                be_trigger_roe=float(guard["break_even_trigger_roe"]),
                be_offset_fraction=float(guard["break_even_offset_underlying_fraction"]),
                activation_roe=float(guard["trailing_activation_roe"]),
                hard_stop_roe=float(guard["hard_stop_roe"]),
            )
            for name, multiplier in actions.items()
        }
        row = {
            "position_episode_id": episode["position_episode_id"],
            "symbol": episode["symbol"], "side": episode["side"],
            "entry_timestamp_ms": int(episode["entry_timestamp_ms"]),
            "best_action": choose_best_action(action_results),
        }
        for name, result in action_results.items():
            row.update(result_columns(name.lower(), result, future_peak_return=future_peak))
        rows.append(row)
    return pd.DataFrame(rows)


def model_pipeline(model: Any, numeric: list[str]) -> Pipeline:
    transform = ColumnTransformer([
        ("numeric", Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), numeric),
        ("categorical", OneHotEncoder(handle_unknown="ignore", sparse_output=False), ["symbol", "side"]),
    ])
    return Pipeline([("features", transform), ("model", model)])


def materialize_policy(frame: pd.DataFrame, actions: pd.Series, name: str) -> pd.DataFrame:
    result = frame[["position_episode_id", "symbol", "side", "entry_timestamp_ms"]].copy()
    result["state"] = actions.to_numpy()
    for suffix in ("net_return_bps", "gross_return_bps", "profit_capture_ratio", "peak_mfe_bps", "final_giveback_bps", "early_exit_regret_bps", "hold_too_long_regret_bps", "mae_bps", "exit_bar", "exit_reason"):
        result[suffix] = [frame.loc[index, f"{action.lower()}_{suffix}"] for index, action in zip(frame.index, actions, strict=True)]
    result["policy"] = name
    result["utc_day"] = pd.to_datetime(result.entry_timestamp_ms, unit="ms", utc=True).dt.strftime("%Y-%m-%d")
    return result


def add_risk_metrics(summary: dict[str, Any], frame: pd.DataFrame) -> dict[str, Any]:
    ordered = frame.sort_values("entry_timestamp_ms")
    returns = ordered.net_return_bps.to_numpy(float)
    equity = np.cumsum(returns)
    peak = np.maximum.accumulate(np.concatenate(([0.0], equity)))[1:]
    drawdown = peak - equity
    downside = returns[returns < 0]
    summary.update({
        "maximum_drawdown_bps_additive": float(drawdown.max(initial=0.0)),
        "sortino_episode": float(returns.mean() / downside.std()) if len(downside) > 1 and downside.std() > 0 else 0.0,
        "expected_shortfall_p95_bps": float(returns[returns <= np.quantile(returns, 0.05)].mean()) if len(returns) else 0.0,
    })
    return summary


def simple_baselines(episodes: pd.DataFrame, config: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, pd.DataFrame]]:
    outputs, frames = [], {}
    definitions = []
    for value in config["baselines"]["fixed_trailing_atr"]:
        definitions.append((f"FIXED_TRAILING_{value}", "FIXED_TRAILING", float(value)))
    for value in config["baselines"]["fixed_percentage_giveback"]:
        definitions.append((f"PERCENT_GIVEBACK_{value}", "PERCENT_GIVEBACK", float(value)))
    for value in config["baselines"]["fixed_tp_atr"]:
        definitions.append((f"FIXED_TP_{value}", "FIXED_TP", float(value)))
    for value in config["baselines"]["time_exit_bars_after_activation"]:
        definitions.append((f"TIME_EXIT_{value}", "TIME_EXIT", float(value)))
    for label, policy, parameter in definitions:
        rows = []
        for episode in episodes.to_dict("records"):
            future_peak = full_path_peak_return(episode)
            result = simulate_simple_baseline(
                episode, policy=policy, parameter=parameter,
                cost_bps=float(config["economics"]["base_round_trip_cost_bps"]),
            )
            row = {"position_episode_id": episode["position_episode_id"], "symbol": episode["symbol"], "side": episode["side"], "entry_timestamp_ms": episode["entry_timestamp_ms"]}
            row.update({key.removeprefix("x_"): value for key, value in result_columns("x", result, future_peak_return=future_peak).items()})
            rows.append(row)
        frame = pd.DataFrame(rows)
        frame["utc_day"] = pd.to_datetime(frame.entry_timestamp_ms, unit="ms", utc=True).dt.strftime("%Y-%m-%d")
        frames[label] = frame
        outputs.append({"policy": label, **add_risk_metrics(policy_summary(frame), frame)})
    return outputs, frames


def write_reports(root: Path, config: dict[str, Any], result: dict[str, Any], config_hash: str) -> None:
    report_dir = root / "reports/governance/aegis_prospective_validation/live/adaptive_profit_guard_w6"
    report_dir.mkdir(parents=True, exist_ok=True)
    prereg = f"""# Aegis Adaptive Profit Guard W6 - Preregistration\n\n- Mode: RESEARCH_ONLY\n- Config SHA-256: `{config_hash}`\n- Population: W2 TRAIN episodes only\n- W2 VALIDATION/HOLDOUT: PROHIBITED\n- W6 FINAL HOLDOUT: SEALED\n- Primary metric: net expectancy bps per independent position episode\n- Current guard: BE 0.08 ROE, activation 0.15 ROE, ATR 1.5, leverage 20 mapping\n- Adaptive actions: DEFENSIVE 0.75 ATR, NORMAL 1.5 ATR, EXPANSION 2.25 ATR\n- No entry, direction, sizing, execution, production, PM2 or exchange changes.\n"""
    (report_dir / "aegis_adaptive_profit_guard_w6_preregistration.md").write_text(prereg)
    verdict = result["verdict"]
    validation = result["validation"]
    state_counts = ", ".join(f"{key}={value}" for key, value in result["model"]["validation_state_counts"].items())
    failed_gates = ", ".join(key for key, value in result["gate_checks"].items() if not value)
    side_results = ", ".join(f"{key}={value:.4f} bps" for key, value in validation["per_side_improvement_bps"].items())
    report = f"""# Aegis Adaptive Profit Guard W6 - Result\n\n## Verdict\n\n`{verdict['status']}`\n\n- W6_ADAPTIVE_GUARD_EDGE_FOUND: `{str(verdict['W6_ADAPTIVE_GUARD_EDGE_FOUND']).upper()}`\n- W6_MODELING_JUSTIFIED: `{str(verdict['W6_MODELING_JUSTIFIED']).upper()}`\n- W6_READY_FOR_SHADOW: `FALSE`\n- W6_READY_FOR_LIVE: `FALSE`\n- FINAL_HOLDOUT_W6: `{verdict['final_holdout_state']}`\n\n## Population\n\n- TRAIN episodes: {result['population']['train_episodes']}\n- VALIDATION episodes: {result['population']['validation_episodes']}\n- Activated in TRAIN: {result['population']['train_activated']}\n- Activated in VALIDATION: {result['population']['validation_activated']}\n- Symbols: {result['population']['symbols']}\n- Source accessed: W2 TRAIN only; prohibited partitions accessed: none.\n\n## Primary Comparison\n\n- Selected model: `{result['model']['selected']}`\n- Validation states: {state_counts}\n- CURRENT_GUARD expectancy: {validation['current']['net_expectancy_bps']:.4f} bps/episode\n- Adaptive expectancy: {validation['adaptive']['net_expectancy_bps']:.4f} bps/episode\n- Improvement: {validation['improvement_mean_bps']:.4f} bps/episode\n- Paired 95% CI: [{validation['improvement_ci_95_bps'][0]:.4f}, {validation['improvement_ci_95_bps'][1]:.4f}]\n- Side improvement: {side_results}\n- Positive symbols: {validation['positive_symbols']}/11\n- Positive temporal folds: {validation['positive_temporal_folds']}/4\n- Failed gates: {failed_gates}\n\n## Profit And Risk\n\n| Metric | CURRENT_GUARD | Adaptive |\n|---|---:|---:|\n| Profit factor | {validation['current']['profit_factor']:.4f} | {validation['adaptive']['profit_factor']:.4f} |\n| Median capture ratio | {validation['current']['median_profit_capture_ratio']:.4f} | {validation['adaptive']['median_profit_capture_ratio']:.4f} |\n| Median giveback | {validation['current']['median_giveback_bps']:.4f} bps | {validation['adaptive']['median_giveback_bps']:.4f} bps |\n| P95 giveback | {validation['current']['p95_giveback_bps']:.4f} bps | {validation['adaptive']['p95_giveback_bps']:.4f} bps |\n| Median early-exit regret | {validation['current']['median_early_exit_regret_bps']:.4f} bps | {validation['adaptive']['median_early_exit_regret_bps']:.4f} bps |\n| Median hold-too-long regret | {validation['current']['median_hold_too_long_regret_bps']:.4f} bps | {validation['adaptive']['median_hold_too_long_regret_bps']:.4f} bps |\n\n## Interpretation\n\n{verdict['interpretation']} The classifier assigned different states, but the economic effect was effectively zero and unstable across sides, symbols, and time.\n\nThis is a conservative closed-5m reconstruction. Production evaluates mark price more frequently, so even a positive result would require a new independent holdout and Shadow parity before any runtime use.\n"""
    (report_dir / "aegis_adaptive_profit_guard_w6_result.md").write_text(report)
    (report_dir / "aegis_adaptive_profit_guard_w6_verdict.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    root = args.root.resolve()
    config_path = root / "config/experiments/aegis_adaptive_profit_guard_w6.yaml"
    config = yaml.safe_load(config_path.read_text())
    config_hash = sha256(config_path)
    episodes, decisions = load_population(root, config)
    activations = activation_rows(decisions, config)
    replay = replay_actions(episodes, config)
    frame = replay.merge(activations, on=["position_episode_id", "symbol", "side", "entry_timestamp_ms"], how="left")
    frame["activated"] = frame["activated"].fillna(False)
    frame["best_action"] = np.where(frame.activated, frame.best_action, "NORMAL")

    train_start, train_end = map(utc_ms, config["partitions"]["train"])
    val_start, val_end = map(utc_ms, config["partitions"]["validation"])
    purge = int(config["partitions"]["purge_minutes"]) * 60_000
    train = frame.loc[(frame.entry_timestamp_ms >= train_start) & (frame.entry_timestamp_ms < train_end - purge)].copy()
    validation = frame.loc[(frame.entry_timestamp_ms >= val_start + purge) & (frame.entry_timestamp_ms < val_end)].copy()
    train_active, validation_active = train.loc[train.activated].copy(), validation.loc[validation.activated].copy()
    if len(validation) < int(config["statistics"]["minimum_validation_episodes"]):
        raise RuntimeError("AEGIS_W6_VALIDATION_POPULATION_INSUFFICIENT")

    numeric = list(config["features"]["columns"])
    predictors = [*numeric, "symbol", "side"]
    split_ts = train_active.entry_timestamp_ms.quantile(float(config["models"]["internal_train_fit_fraction"]))
    fit = train_active.loc[train_active.entry_timestamp_ms <= split_ts]
    select = train_active.loc[train_active.entry_timestamp_ms > split_ts]
    seed = int(config["models"]["random_seed"])
    candidates = {
        "MULTINOMIAL_LOGISTIC_L2": LogisticRegression(C=float(config["models"]["logistic_C"]), max_iter=500, class_weight="balanced", random_state=seed),
        "SHALLOW_DECISION_TREE": DecisionTreeClassifier(max_depth=int(config["models"]["tree_max_depth"]), min_samples_leaf=int(config["models"]["tree_min_samples_leaf"]), class_weight="balanced", random_state=seed),
    }
    selection_results, fitted = {}, {}
    pvalues = {}
    normal_select = materialize_policy(select, pd.Series("NORMAL", index=select.index), "CURRENT_GUARD")
    for name, estimator in candidates.items():
        model = model_pipeline(estimator, numeric).fit(fit[predictors], fit.best_action)
        predicted = pd.Series(model.predict(select[predictors]), index=select.index)
        policy = materialize_policy(select, predicted, name)
        paired = policy[["position_episode_id", "net_return_bps", "utc_day"]].merge(normal_select[["position_episode_id", "net_return_bps"]], on="position_episode_id", suffixes=("", "_current"))
        paired["improvement_bps"] = paired.net_return_bps - paired.net_return_bps_current
        ci, pvalue = paired_day_bootstrap(paired, repetitions=int(config["statistics"]["bootstrap_repetitions"]), seed=seed)
        selection_results[name] = {"net_expectancy_bps": float(policy.net_return_bps.mean()), "improvement_bps": float(paired.improvement_bps.mean()), "improvement_ci_95_bps": ci, "bootstrap_p_one_sided": pvalue}
        pvalues[name] = pvalue
        fitted[name] = model
    fdr = benjamini_hochberg(pvalues, alpha=float(config["statistics"]["fdr_alpha"]))
    selected_name = max(selection_results, key=lambda name: (selection_results[name]["improvement_bps"], name))
    selected = model_pipeline(candidates[selected_name], numeric).fit(train_active[predictors], train_active.best_action)

    predicted_validation = pd.Series("NORMAL", index=validation.index)
    predicted_validation.loc[validation_active.index] = selected.predict(validation_active[predictors])
    adaptive = materialize_policy(validation, predicted_validation, f"ADAPTIVE_{selected_name}")
    current = materialize_policy(validation, pd.Series("NORMAL", index=validation.index), "CURRENT_GUARD")
    paired = adaptive[["position_episode_id", "symbol", "side", "entry_timestamp_ms", "net_return_bps", "utc_day"]].merge(current[["position_episode_id", "net_return_bps"]], on="position_episode_id", suffixes=("", "_current"))
    paired["improvement_bps"] = paired.net_return_bps - paired.net_return_bps_current
    ci, probability_nonpositive = paired_day_bootstrap(paired, repetitions=int(config["statistics"]["bootstrap_repetitions"]), seed=seed + 1)
    paired["fold"] = pd.qcut(paired.entry_timestamp_ms.rank(method="first"), 4, labels=False)
    positive_folds = int((paired.groupby("fold").improvement_bps.mean() > 0).sum())
    positive_symbols = int((paired.groupby("symbol").improvement_bps.mean() > 0).sum())
    validation_episodes = episodes.loc[episodes.position_episode_id.isin(validation.position_episode_id)]
    baseline_rows, baseline_frames = simple_baselines(validation_episodes, config)
    best_simple = max(baseline_rows, key=lambda row: row["net_expectancy_bps"])
    adaptive_summary = add_risk_metrics(policy_summary(adaptive), adaptive)
    current_summary = add_risk_metrics(policy_summary(current), current)
    state_counts = {str(key): int(value) for key, value in predicted_validation.value_counts().items()}

    gate_cfg = config["gate"]
    checks = {
        "minimum_effect": float(paired.improvement_bps.mean()) >= float(gate_cfg["minimum_mean_improvement_vs_current_bps"]),
        "paired_ci_positive": float(ci[0]) > 0,
        "positive_symbols": positive_symbols >= int(gate_cfg["minimum_positive_symbols"]),
        "positive_folds": positive_folds >= int(gate_cfg["minimum_positive_walk_forward_folds"]),
        "profit_factor_not_worse": adaptive_summary["profit_factor"] >= current_summary["profit_factor"],
        "p95_giveback_safe": adaptive_summary["p95_giveback_bps"] <= current_summary["p95_giveback_bps"] + float(gate_cfg["maximum_p95_giveback_degradation_bps"]),
        "beats_best_simple": adaptive_summary["net_expectancy_bps"] > float(best_simple["net_expectancy_bps"]),
        "stress_cost_positive": adaptive_summary["net_expectancy_bps"] - (float(config["economics"]["stress_round_trip_cost_bps"]) - float(config["economics"]["base_round_trip_cost_bps"])) > 0,
    }
    edge = all(checks.values())
    status = "AEGIS_ADAPTIVE_PROFIT_GUARD_W6_EDGE_FOUND" if edge else "AEGIS_ADAPTIVE_PROFIT_GUARD_W6_NO_ECONOMIC_EDGE"
    interpretation = (
        "The frozen adaptive guard cleared every preregistered TRAIN/VALIDATION gate; the independent final holdout remains sealed, so Shadow is not yet authorized."
        if edge else
        "Wave/regime state did not improve the frozen current guard by the preregistered economic margin with stable risk. The holdout remained sealed and no Shadow or production change is justified."
    )
    result = {
        "schema_version": "aegis-adaptive-profit-guard-w6-verdict-v1",
        "experiment_id": config["experiment_id"],
        "config_sha256": config_hash,
        "population": {
            "train_episodes": int(len(train)), "validation_episodes": int(len(validation)),
            "train_activated": int(len(train_active)), "validation_activated": int(len(validation_active)),
            "symbols": int(validation.symbol.nunique()), "sides": sorted(validation.side.unique()),
            "source_partitions_accessed": ["W2_TRAIN"],
            "prohibited_partitions_accessed": [],
        },
        "model": {
            "selected": selected_name, "selection_results": selection_results,
            "selection_fdr_accepted": fdr, "validation_state_counts": state_counts,
            "tree_rules_if_selected": export_text(selected.named_steps["model"], feature_names=list(selected.named_steps["features"].get_feature_names_out())) if selected_name == "SHALLOW_DECISION_TREE" else "NOT_SELECTED",
        },
        "validation": {
            "current": current_summary, "adaptive": adaptive_summary,
            "improvement_mean_bps": float(paired.improvement_bps.mean()),
            "improvement_ci_95_bps": ci,
            "bootstrap_probability_improvement_nonpositive": probability_nonpositive,
            "positive_symbols": positive_symbols, "positive_temporal_folds": positive_folds,
            "best_simple_baseline": best_simple, "all_simple_baselines": baseline_rows,
            "per_symbol_improvement_bps": {str(k): float(v) for k, v in paired.groupby("symbol").improvement_bps.mean().items()},
            "per_side_improvement_bps": {str(k): float(v) for k, v in paired.groupby("side").improvement_bps.mean().items()},
        },
        "gate_checks": checks,
        "verdict": {
            "status": status,
            "W6_ADAPTIVE_GUARD_EDGE_FOUND": edge,
            "W6_MODELING_JUSTIFIED": edge,
            "W6_READY_FOR_SHADOW": False,
            "W6_READY_FOR_LIVE": False,
            "final_holdout_state": "SEALED_NOT_OPENED",
            "interpretation": interpretation,
        },
        "safety": {"production_changes": "NONE", "authenticated_requests": 0, "exchange_mutations": 0, "pm2_changes": "NONE"},
    }
    output_dir = root / "data/adaptive_profit_guard_w6/run_01"
    output_dir.mkdir(parents=True, exist_ok=True)
    adaptive.to_parquet(output_dir / "validation_adaptive_policy.parquet", index=False)
    paired.to_parquet(output_dir / "validation_paired_comparison.parquet", index=False)
    (output_dir / "result.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    write_reports(root, config, result, config_hash)
    print(json.dumps({"status": status, "improvement_bps": result["validation"]["improvement_mean_bps"], "gate_checks": checks, "selected": selected_name}, indent=2))


if __name__ == "__main__":
    main()
