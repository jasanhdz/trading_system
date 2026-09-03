#!/usr/bin/env python3
"""Evaluate frozen dynamic early-path confirmation for Aegis W12."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import balanced_accuracy_score, brier_score_loss, log_loss
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from aegis.research.early_path_confirmation_w12 import w12_feature_groups


def _classifier(c: float, seed: int) -> Pipeline:
    return Pipeline([
        ("impute", SimpleImputer(strategy="median", add_indicator=True)),
        ("scale", StandardScaler()),
        ("model", LogisticRegression(C=c, class_weight="balanced", max_iter=3000, random_state=seed)),
    ])


def _regressor(alpha: float) -> Pipeline:
    return Pipeline([
        ("impute", SimpleImputer(strategy="median", add_indicator=True)),
        ("scale", StandardScaler()),
        ("model", Ridge(alpha=alpha)),
    ])


def fit_ablation(train: pd.DataFrame, features: list[str], config: dict) -> tuple[Pipeline, Pipeline]:
    classifier = _classifier(float(config["models"]["logistic_c"]), int(config["seed"]))
    regressor = _regressor(float(config["models"]["ridge_alpha"]))
    classifier.fit(train[features], train["remaining_positive_label"])
    regressor.fit(train[features], train["remaining_net_return_bps"])
    return classifier, regressor


def dynamic_decisions(
    frame: pd.DataFrame, classifier: Pipeline, regressor: Pipeline,
    features: list[str], config: dict,
) -> tuple[pd.DataFrame, dict[str, float]]:
    working = frame.copy()
    working["probability_positive_remaining"] = classifier.predict_proba(working[features])[:, 1]
    working["predicted_remaining_net_bps"] = regressor.predict(working[features])
    enter_threshold = float(config["models"]["enter_probability_threshold"])
    cancel_threshold = float(config["models"]["cancel_probability_threshold"])
    minimum_net = float(config["models"]["minimum_predicted_remaining_net_bps"])
    decisions = []
    for episode_id, group in working.groupby("original_live_signal_id", sort=True):
        states = group.sort_values("state_index")
        selected = None
        reason = "CANCEL_NO_CONFIRMATION"
        for _, state in states.iterrows():
            probability = float(state["probability_positive_remaining"])
            predicted_net = float(state["predicted_remaining_net_bps"])
            if probability >= enter_threshold and predicted_net >= minimum_net:
                selected = state
                break
            if int(state["state_index"]) == 1 and probability <= cancel_threshold:
                reason = "CANCEL_EARLY_INVALIDATION"
                break
        if selected is None:
            decisions.append({
                "original_live_signal_id": episode_id, "executed": False,
                "selected_state_index": 0, "decision": reason, "reason": reason,
                "selected_gross_bps": 0.0, "selected_mfe_bps": math.nan,
                "selected_mae_bps": math.nan, "confirmation_elapsed_seconds": math.nan,
                "missed_mfe_before_entry_bps": math.nan, "entry_price_improvement_bps": math.nan,
            })
        else:
            decisions.append({
                "original_live_signal_id": episode_id, "executed": True,
                "selected_state_index": int(selected["state_index"]),
                "decision": f"ENTER_AT_STATE_{int(selected['state_index'])}", "reason": "EARLY_PATH_CONFIRMED",
                "selected_gross_bps": float(selected["remaining_gross_return_bps"]),
                "selected_mfe_bps": float(selected["remaining_mfe_bps"]),
                "selected_mae_bps": float(selected["remaining_mae_bps"]),
                "confirmation_elapsed_seconds": float(selected["early_elapsed_seconds"]),
                "missed_mfe_before_entry_bps": float(selected["missed_mfe_before_entry_bps"]),
                "entry_price_improvement_bps": float(selected["entry_price_improvement_bps"]),
            })
    labels = working["remaining_positive_label"].to_numpy(int)
    probabilities = working["probability_positive_remaining"].to_numpy(float)
    prediction = probabilities >= 0.5
    model_metrics = {
        "balanced_accuracy": float(balanced_accuracy_score(labels, prediction)),
        "brier_score": float(brier_score_loss(labels, probabilities)),
        "log_loss": float(log_loss(labels, probabilities, labels=[0, 1])),
        "positive_label_rate": float(labels.mean()),
    }
    return pd.DataFrame(decisions), model_metrics


def fixed_decisions(w11: pd.DataFrame, episode_ids: set[str], delay: int) -> pd.DataFrame:
    selected = w11.loc[
        w11["live_signal_episode_id"].isin(episode_ids) & w11["delay_minutes"].eq(delay)
    ].copy()
    return pd.DataFrame({
        "original_live_signal_id": selected["live_signal_episode_id"],
        "executed": True,
        "selected_state_index": delay,
        "decision": "ENTER_NOW" if delay == 0 else f"WAIT_{delay}M_FIXED",
        "reason": "BASELINE",
        "selected_gross_bps": selected["gross_return_bps"],
        "selected_mfe_bps": selected["mfe_bps"],
        "selected_mae_bps": selected["mae_bps"],
        "confirmation_elapsed_seconds": float(delay * 60),
        "missed_mfe_before_entry_bps": np.nan,
        "entry_price_improvement_bps": np.nan,
    }).reset_index(drop=True)


def frozen_w11_decisions(
    w11_decisions: pd.DataFrame, w11_dataset: pd.DataFrame, episode_ids: set[str]
) -> pd.DataFrame:
    decisions = w11_decisions.loc[w11_decisions["live_signal_episode_id"].isin(episode_ids)].copy()
    outcomes = w11_dataset[[
        "live_signal_episode_id", "delay_minutes", "gross_return_bps", "mfe_bps", "mae_bps"
    ]]
    decisions = decisions.merge(
        outcomes, left_on=["live_signal_episode_id", "selected_delay"],
        right_on=["live_signal_episode_id", "delay_minutes"], how="left", validate="one_to_one",
    )
    return pd.DataFrame({
        "original_live_signal_id": decisions["live_signal_episode_id"],
        "executed": decisions["executed"].astype(bool),
        "selected_state_index": decisions["selected_delay"].astype(int),
        "decision": decisions["decision"], "reason": decisions["reason"].fillna(""),
        "selected_gross_bps": decisions["gross_return_bps"],
        "selected_mfe_bps": decisions["mfe_bps"], "selected_mae_bps": decisions["mae_bps"],
        "confirmation_elapsed_seconds": decisions["selected_delay"] * 60.0,
        "missed_mfe_before_entry_bps": np.nan, "entry_price_improvement_bps": np.nan,
    })


def policy_metrics(
    decisions: pd.DataFrame, metadata: pd.DataFrame, *, cost_bps: float,
    baseline_net: pd.Series,
) -> tuple[dict[str, Any], pd.Series]:
    result = decisions.set_index("original_live_signal_id").join(metadata.set_index("original_live_signal_id"), how="left")
    executed = result["executed"].astype(bool)
    gross = result["selected_gross_bps"].where(executed, 0.0)
    net = (gross - cost_bps).where(executed, 0.0)
    positive, negative = net.clip(lower=0.0).sum(), -net.clip(upper=0.0).sum()
    good = result["entry_class"].eq("GOOD_CLEAN_ENTRY")
    bad = result["entry_class"].eq("BAD_ENTRY")
    mixed = result["entry_class"].eq("MIXED_OR_EXIT_DEPENDENT")
    symbols = result.loc[executed, "symbol"].value_counts()
    baseline_mae = float(metadata["baseline_mae_bps"].median())
    selected_mae = float(result.loc[executed, "selected_mae_bps"].median()) if executed.any() else math.nan
    metrics = {
        "episodes": int(len(result)), "executed": int(executed.sum()), "cancelled": int((~executed).sum()),
        "execution_rate": float(executed.mean()), "cancellation_rate": float((~executed).mean()),
        "gross_bps_per_original_signal": float(gross.mean()), "net_bps_per_original_signal": float(net.mean()),
        "gross_bps_per_executed_trade": float(gross.loc[executed].mean()) if executed.any() else math.nan,
        "net_bps_per_executed_trade": float(net.loc[executed].mean()) if executed.any() else math.nan,
        "improvement_vs_enter_now_bps": float((net - baseline_net).mean()),
        "profit_factor": float(positive / negative) if negative > 0 else math.inf,
        "win_rate": float(net.loc[executed].gt(0.0).mean()) if executed.any() else math.nan,
        "good_retention": float(executed.loc[good].mean()) if good.any() else math.nan,
        "bad_avoidance": float((~executed.loc[bad]).mean()) if bad.any() else math.nan,
        "mixed_retention": float(executed.loc[mixed].mean()) if mixed.any() else math.nan,
        "median_mfe_bps": float(result.loc[executed, "selected_mfe_bps"].median()) if executed.any() else math.nan,
        "median_mae_bps": selected_mae,
        "median_mae_reduction_vs_enter_now": (baseline_mae - selected_mae) / max(baseline_mae, 1e-9) if executed.any() else math.nan,
        "median_confirmation_seconds": float(result.loc[executed, "confirmation_elapsed_seconds"].median()) if executed.any() else math.nan,
        "median_missed_mfe_bps": float(result.loc[executed, "missed_mfe_before_entry_bps"].median()) if executed.any() else math.nan,
        "median_entry_price_improvement_bps": float(result.loc[executed, "entry_price_improvement_bps"].median()) if executed.any() else math.nan,
        "confirmation_too_late_rate": float((result.loc[executed, "missed_mfe_before_entry_bps"].ge(30.0) & net.loc[executed].le(0.0)).mean()) if executed.any() else math.nan,
        "actions": result["decision"].value_counts().sort_index().astype(int).to_dict(),
        "symbols_executed": int(len(symbols)),
        "maximum_single_symbol_fraction": float(symbols.iloc[0] / executed.sum()) if executed.any() else 1.0,
        "long_executed": int((executed & result["side"].eq("LONG")).sum()),
        "short_executed": int((executed & result["side"].eq("SHORT")).sum()),
    }
    return metrics, net


def bootstrap_differences(
    policy: pd.Series, references: dict[str, pd.Series], dates: pd.Series, config: dict
) -> dict[str, Any]:
    rng = np.random.default_rng(int(config["seed"]))
    samples_n = int(config["validation_gates"]["bootstrap_samples"])
    result = {}
    unique_dates = dates.unique()
    for name, reference in references.items():
        difference = (policy - reference).to_numpy(float)
        episode_samples = np.empty(samples_n)
        block_samples = np.empty(samples_n)
        grouped = {date: difference[dates.eq(date).to_numpy()] for date in unique_dates}
        for index in range(samples_n):
            draw = rng.integers(0, len(difference), len(difference))
            episode_samples[index] = difference[draw].mean()
            selected_dates = rng.choice(unique_dates, len(unique_dates), replace=True)
            block_samples[index] = np.concatenate([grouped[date] for date in selected_dates]).mean()
        result[name] = {
            "mean_bps": float(difference.mean()),
            "episode_ci95": [float(value) for value in np.quantile(episode_samples, [0.025, 0.975])],
            "temporal_block_ci95": [float(value) for value in np.quantile(block_samples, [0.025, 0.975])],
            "probability_positive": float((block_samples > 0.0).mean()),
        }
    return result


def benjamini_hochberg(p_values: dict[str, float]) -> dict[str, float]:
    ordered = sorted(p_values, key=p_values.get)
    count = len(ordered)
    adjusted = {}
    running = 1.0
    for rank_from_end, name in enumerate(reversed(ordered), start=1):
        rank = count - rank_from_end + 1
        running = min(running, p_values[name] * count / rank)
        adjusted[name] = float(min(1.0, running))
    return adjusted


def segment_results(decisions: pd.DataFrame, metadata: pd.DataFrame, cost_bps: float, column: str) -> list[dict[str, Any]]:
    result = decisions.set_index("original_live_signal_id").join(metadata.set_index("original_live_signal_id"), how="left")
    if column == "signal_date":
        result["signal_date"] = pd.to_datetime(result["signal_timestamp"], utc=True).dt.strftime("%Y-%m-%d")
    rows = []
    for value, group in result.groupby(column, sort=True):
        executed = group["executed"].astype(bool)
        net = (group["selected_gross_bps"] - cost_bps).where(executed, 0.0)
        rows.append({
            column: str(value), "episodes": int(len(group)), "executed": int(executed.sum()),
            "net_bps_per_signal": float(net.mean()),
            "net_bps_per_trade": float(net.loc[executed].mean()) if executed.any() else math.nan,
        })
    return rows


def early_path_class_diagnostics(validation: pd.DataFrame) -> list[dict[str, Any]]:
    rows = []
    for (state, label), group in validation.groupby(["state_index", "entry_class"], sort=True):
        rows.append({
            "state_index": int(state), "entry_class": str(label), "episodes": int(len(group)),
            "median_directional_return_bps": float(group["early_directional_return_bps"].median()),
            "median_favorable_excursion_bps": float(group["early_favorable_excursion_bps"].median()),
            "median_adverse_excursion_bps": float(group["early_adverse_excursion_bps"].median()),
            "median_path_efficiency": float(group["early_path_efficiency"].median()),
            "median_directional_taker_imbalance": float(group["early_directional_taker_imbalance"].median()),
            "positive_remaining_rate": float(group["remaining_positive_label"].mean()),
        })
    return rows


def _safe(value: Any) -> Any:
    if isinstance(value, dict): return {str(key): _safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)): return [_safe(item) for item in value]
    if isinstance(value, (np.integer,)): return int(value)
    if isinstance(value, (np.floating, float)): return float(value) if math.isfinite(float(value)) else None
    if isinstance(value, (np.bool_,)): return bool(value)
    return value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("config/experiments/aegis_early_path_confirmation_w12.yaml"))
    parser.add_argument("--dataset", type=Path, default=Path("data/early_path_confirmation_w12/run_01/w12_early_path_train_validation.parquet"))
    parser.add_argument("--out-dir", type=Path, default=Path("reports/governance/aegis_prospective_validation/live/early_path_confirmation_w12"))
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    frame = pd.read_parquet(args.dataset)
    train = frame.loc[frame["w12_split"].eq("W12_TRAIN")].copy()
    validation = frame.loc[frame["w12_split"].eq("W12_VALIDATION")].copy()
    feature_groups = w12_feature_groups(list(frame.columns))
    w11_dataset = pd.read_parquet(config["sources"]["w11_dataset"])
    w11_decision_log = pd.read_csv(config["sources"]["w11_validation_decisions"])
    validation_ids = set(validation["original_live_signal_id"].unique())
    metadata = validation.loc[validation["state_index"].eq(1), [
        "original_live_signal_id", "entry_class", "symbol", "side", "signal_timestamp"
    ]].copy()
    baseline_rows = w11_dataset.loc[
        w11_dataset["live_signal_episode_id"].isin(validation_ids) & w11_dataset["delay_minutes"].eq(0)
    ][["live_signal_episode_id", "mae_bps"]].rename(columns={
        "live_signal_episode_id": "original_live_signal_id", "mae_bps": "baseline_mae_bps"
    })
    metadata = metadata.merge(baseline_rows, on="original_live_signal_id", how="left", validate="one_to_one")

    baseline_decisions = {f"WAIT_{delay}M_FIXED" if delay else "ENTER_NOW": fixed_decisions(w11_dataset, validation_ids, delay) for delay in (0, 1, 2, 3)}
    baseline_decisions["W11_FROZEN"] = frozen_w11_decisions(w11_decision_log, w11_dataset, validation_ids)
    cost = float(config["outcome"]["baseline_cost_bps"])
    baseline_metrics = {}
    baseline_nets = {}
    enter_now_net = None
    for name, decisions in baseline_decisions.items():
        reference = pd.Series(0.0, index=decisions["original_live_signal_id"])
        metrics, net = policy_metrics(decisions, metadata, cost_bps=cost, baseline_net=reference)
        net.index = decisions["original_live_signal_id"]
        baseline_metrics[name] = metrics
        baseline_nets[name] = net
        if name == "ENTER_NOW": enter_now_net = net
    assert enter_now_net is not None
    for name in baseline_metrics:
        baseline_metrics[name]["improvement_vs_enter_now_bps"] = float((baseline_nets[name] - enter_now_net).mean())

    ablation_metrics, ablation_models, ablation_decisions, ablation_nets = {}, {}, {}, {}
    for name in config["models"]["feature_ablations"]:
        features = feature_groups[name]
        classifier, regressor = fit_ablation(train, features, config)
        decisions, model_metrics = dynamic_decisions(validation, classifier, regressor, features, config)
        metrics, net = policy_metrics(decisions, metadata, cost_bps=cost, baseline_net=enter_now_net)
        net.index = decisions["original_live_signal_id"]
        ablation_metrics[name] = metrics
        ablation_models[name] = model_metrics
        ablation_decisions[name] = decisions
        ablation_nets[name] = net
    primary_name = "FULL_W12"
    primary_metrics = ablation_metrics[primary_name]
    primary_net = ablation_nets[primary_name]
    stress_metrics, _ = policy_metrics(
        ablation_decisions[primary_name], metadata,
        cost_bps=float(config["outcome"]["stress_cost_bps"]), baseline_net=enter_now_net,
    )
    dates = pd.to_datetime(metadata.set_index("original_live_signal_id").loc[primary_net.index, "signal_timestamp"], utc=True).dt.strftime("%Y-%m-%d")
    references = {
        "ENTER_NOW": enter_now_net,
        "WAIT_2M_FIXED": baseline_nets["WAIT_2M_FIXED"],
        "W11_FROZEN": baseline_nets["W11_FROZEN"],
    }
    bootstrap = bootstrap_differences(primary_net, references, dates, config)
    ablation_bootstrap = {
        name: bootstrap_differences(net, {"ENTER_NOW": enter_now_net}, dates, config)["ENTER_NOW"]
        for name, net in ablation_nets.items()
    }
    p_values = {name: 1.0 - values["probability_positive"] for name, values in ablation_bootstrap.items()}
    fdr = benjamini_hochberg(p_values)
    gate = config["validation_gates"]
    gates = {
        "minimum_episodes": primary_metrics["episodes"] >= int(gate["minimum_episodes"]),
        "minimum_executed": primary_metrics["executed"] >= int(gate["minimum_executed"]),
        "execution_rate": float(gate["minimum_execution_rate"]) <= primary_metrics["execution_rate"] <= float(gate["maximum_execution_rate"]),
        "positive_per_trade": primary_metrics["net_bps_per_executed_trade"] >= float(gate["minimum_net_bps_per_executed_trade"]),
        "positive_per_signal": primary_metrics["net_bps_per_original_signal"] > float(gate["minimum_net_bps_per_original_signal"]),
        "beats_enter_now": bootstrap["ENTER_NOW"]["mean_bps"] >= float(gate["minimum_improvement_vs_enter_now_bps"]),
        "beats_wait_2m": bootstrap["WAIT_2M_FIXED"]["mean_bps"] >= float(gate["minimum_improvement_vs_wait_2m_bps"]),
        "beats_w11": bootstrap["W11_FROZEN"]["mean_bps"] >= float(gate["minimum_improvement_vs_w11_bps"]),
        "good_retention": primary_metrics["good_retention"] >= float(gate["minimum_good_retention"]),
        "bad_avoidance": primary_metrics["bad_avoidance"] >= float(gate["minimum_bad_avoidance"]),
        "mae_reduction": primary_metrics["median_mae_reduction_vs_enter_now"] >= float(gate["minimum_mae_reduction"]),
        "symbol_breadth": primary_metrics["symbols_executed"] >= int(gate["minimum_symbols_executed"]),
        "concentration": primary_metrics["maximum_single_symbol_fraction"] <= float(gate["maximum_single_symbol_fraction"]),
        "bootstrap_enter_now": bootstrap["ENTER_NOW"]["temporal_block_ci95"][0] > 0.0,
        "bootstrap_wait_2m": bootstrap["WAIT_2M_FIXED"]["temporal_block_ci95"][0] > 0.0,
        "bootstrap_w11": bootstrap["W11_FROZEN"]["temporal_block_ci95"][0] > 0.0,
        "stress_cost": stress_metrics["net_bps_per_original_signal"] > 0.0,
    }
    passed = all(gates.values())
    short_only = validation.loc[validation["state_index"].eq(1), "side"].nunique() == 1 and validation["side"].iloc[0] == "SHORT"
    confirmation_values = ablation_decisions[primary_name].loc[
        ablation_decisions[primary_name]["executed"], "confirmation_elapsed_seconds"
    ].dropna()
    verdict = {
        "schema_version": "aegis-early-path-confirmation-w12-verdict-v1",
        "status": "AEGIS_W12_EARLY_PATH_CONFIRMATION_EDGE_FOUND" if passed else "AEGIS_W12_NO_ROBUST_EARLY_PATH_EDGE",
        "dataset": {"train_episodes": int(train["original_live_signal_id"].nunique()), "validation_episodes": int(validation["original_live_signal_id"].nunique()), "states": int(len(frame))},
        "resolution": {"core": "CLOSED_1M", "subminute_available": False, "optional_l2_overlap": "NONE", "states": ["FULL_POST_SIGNAL_BAR_1", "FULL_POST_SIGNAL_BAR_2"]},
        "holdouts": {"W12_FINAL_HOLDOUT": "SEALED_NOT_OPENED", "W11_AUGUST_HOLDOUT": "SEALED_NOT_OPENED", "opened": False},
        "feature_counts": {name: len(values) for name, values in feature_groups.items()},
        "model_metrics": ablation_models,
        "baseline_metrics": baseline_metrics,
        "ablation_metrics": ablation_metrics,
        "primary_stress_metrics": stress_metrics,
        "bootstrap": bootstrap,
        "ablation_bootstrap": ablation_bootstrap,
        "multiple_comparisons": {"method": "BENJAMINI_HOCHBERG", "raw_p_values": p_values, "adjusted_p_values": fdr},
        "early_path_class_diagnostics": early_path_class_diagnostics(validation),
        "primary_per_symbol": segment_results(ablation_decisions[primary_name], metadata, cost, "symbol"),
        "primary_per_date": segment_results(ablation_decisions[primary_name], metadata, cost, "signal_date"),
        "confirmation_time_seconds": {
            "q25": float(confirmation_values.quantile(0.25)) if len(confirmation_values) else math.nan,
            "median": float(confirmation_values.median()) if len(confirmation_values) else math.nan,
            "q75": float(confirmation_values.quantile(0.75)) if len(confirmation_values) else math.nan,
        },
        "validation_gates": gates,
        "flags": {
            "W12_EARLY_PATH_INFORMATION_FOUND": max(
                metrics["balanced_accuracy"] for metrics in ablation_models.values()
            ) >= 0.55,
            "W12_DYNAMIC_CONFIRMATION_VALUE_FOUND": bootstrap["WAIT_2M_FIXED"]["mean_bps"] >= 2.0 and bootstrap["WAIT_2M_FIXED"]["temporal_block_ci95"][0] > 0.0,
            "W12_REMAINING_EDGE_FOUND": primary_metrics["net_bps_per_executed_trade"] >= 2.0,
            "W12_EXECUTED_TRADE_EDGE_FOUND": primary_metrics["net_bps_per_executed_trade"] > 0.0,
            "W12_PER_SIGNAL_EDGE_FOUND": primary_metrics["net_bps_per_original_signal"] > 0.0,
            "W12_COST_GATE_PASSED": stress_metrics["net_bps_per_original_signal"] > 0.0,
            "W12_SHORT_ONLY_EVIDENCE": short_only,
            "W12_MODELING_JUSTIFIED": passed,
            "W12_READY_FOR_PROSPECTIVE_OBSERVATION": passed,
            "W12_READY_FOR_SHADOW": False,
            "W12_READY_FOR_LIVE": False,
        },
        "restrictions_verified": config["restrictions"],
        "limitations": [
            "No sub-minute data overlap exists for the Live signals; 15s/30s/90s states were not fabricated.",
            "W12 validation contains only SHORT signals, so LONG transfer is untested.",
            "The W12 and W11 holdouts remain sealed because validation is the only promotion gate used here.",
        ],
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    output_decisions = ablation_decisions[primary_name].merge(metadata, on="original_live_signal_id", how="left")
    output_decisions.to_csv(args.out_dir / "w12_validation_decisions.csv", index=False)
    (args.out_dir / "aegis_early_path_confirmation_w12_verdict.json").write_text(
        json.dumps(_safe(verdict), indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8"
    )
    (args.out_dir / "aegis_early_path_confirmation_w12_result.md").write_text(_report(verdict), encoding="utf-8")
    print(json.dumps(_safe(verdict), indent=2, sort_keys=True, allow_nan=False))


def _report(verdict: dict[str, Any]) -> str:
    def fmt(value: float) -> str:
        return f"{value:.2f}" if math.isfinite(value) else "N/A"

    primary = verdict["ablation_metrics"]["FULL_W12"]
    stress = verdict["primary_stress_metrics"]
    baseline_rows = "\n".join(
        f"| {name} | {metrics['executed']} | {metrics['net_bps_per_original_signal']:.2f} | {metrics['net_bps_per_executed_trade']:.2f} | {metrics['median_mae_bps']:.1f} |"
        for name, metrics in verdict["baseline_metrics"].items()
    )
    ablation_rows = "\n".join(
        f"| {name} | {metrics['executed']} | {metrics['net_bps_per_original_signal']:.2f} | {metrics['net_bps_per_executed_trade']:.2f} | {metrics['good_retention']:.1%} | {metrics['bad_avoidance']:.1%} | {verdict['model_metrics'][name]['balanced_accuracy']:.3f} |"
        for name, metrics in verdict["ablation_metrics"].items()
    )
    comparison_rows = "\n".join(
        f"| {name} | {values['mean_bps']:+.2f} | [{values['temporal_block_ci95'][0]:.2f}, {values['temporal_block_ci95'][1]:.2f}] |"
        for name, values in verdict["bootstrap"].items()
    )
    diagnostic_rows = "\n".join(
        f"| {row['state_index']} | {row['entry_class']} | {row['episodes']} | {row['median_directional_return_bps']:.1f} | {row['median_favorable_excursion_bps']:.1f} | {row['median_adverse_excursion_bps']:.1f} | {row['median_directional_taker_imbalance']:+.3f} | {row['positive_remaining_rate']:.1%} |"
        for row in verdict["early_path_class_diagnostics"]
    )
    symbol_rows = "\n".join(
        f"| {row['symbol']} | {row['episodes']} | {row['executed']} | {fmt(row['net_bps_per_signal'])} | {fmt(row['net_bps_per_trade'])} |"
        for row in verdict["primary_per_symbol"]
    )
    gates = "\n".join(f"- `{name}`: `{str(value).upper()}`" for name, value in verdict["validation_gates"].items())
    flags = "\n".join(f"- `{name} = {str(value).upper()}`" for name, value in verdict["flags"].items())
    return f"""# Aegis W12 Early Path Confirmation - Result

## Verdict

`{verdict['status']}`

- TRAIN: {verdict['dataset']['train_episodes']} signals.
- VALIDATION: {verdict['dataset']['validation_episodes']} signals.
- Resolution: closed `1m`; no causal sub-minute/L2 overlap.
- W12 and W11 holdouts: `SEALED_NOT_OPENED`.
- W12 executed: {primary['executed']}/{primary['episodes']} ({primary['execution_rate']:.1%}).
- Net: {primary['net_bps_per_original_signal']:.2f} bps/original signal and {primary['net_bps_per_executed_trade']:.2f} bps/executed trade.
- Stress 20 bps: {stress['net_bps_per_original_signal']:.2f} bps/original signal.
- Median confirmation: {primary['median_confirmation_seconds']:.0f}s.
- Median missed MFE: {primary['median_missed_mfe_bps']:.1f} bps.

## Baselines

| Policy | Executed | Net/signal | Net/trade | Median MAE |
|---|---:|---:|---:|---:|
{baseline_rows}

## Ablations

| Features | Executed | Net/signal | Net/trade | GOOD retained | BAD avoided | Balanced accuracy |
|---|---:|---:|---:|---:|---:|---:|
{ablation_rows}

## Primary bootstrap comparisons

| Reference | Improvement bps/signal | Temporal-block 95% CI |
|---|---:|---:|
{comparison_rows}

## Early-path diagnostics

| State | Historical class | N | Directional return | Early MFE | Early MAE | Taker alignment | Positive remaining |
|---:|---|---:|---:|---:|---:|---:|---:|
{diagnostic_rows}

Historical class is diagnostic only and was not a model target.

## Symbols

| Symbol | Signals | Executed | Net/signal | Net/trade |
|---|---:|---:|---:|---:|
{symbol_rows}

## Primary behavior

- Actions: `{primary['actions']}`.
- GOOD retained: {primary['good_retention']:.1%}; BAD avoided: {primary['bad_avoidance']:.1%}; MIXED retained: {primary['mixed_retention']:.1%}.
- Median MFE/MAE: {primary['median_mfe_bps']:.1f}/{primary['median_mae_bps']:.1f} bps.
- Median entry-price improvement: {primary['median_entry_price_improvement_bps']:.1f} bps.
- LONG/SHORT executed: {primary['long_executed']}/{primary['short_executed']}.

## Gates

{gates}

## Flags

{flags}

No production, TypeScript, Brain, guards, leverage, PM2, exchange, authenticated
API, orders, Shadow, Live, or financial state were modified.
"""


if __name__ == "__main__":
    main()
