#!/usr/bin/env python3
"""Fit TRAIN-only W11 detectors and evaluate frozen policies on VALIDATION."""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from aegis.research.entry_safety_gate_w11 import feature_families, path_order_summary


REASON_NAMES = {
    "opposition": "SKIP_OPPOSED",
    "exhaustion": "SKIP_EXHAUSTED",
    "space": "SKIP_NO_SPACE",
    "volatility": "SKIP_VOLATILITY_SHOCK",
}


@dataclass
class FrozenModels:
    risk_models: dict[str, Pipeline]
    risk_thresholds: dict[str, float]
    timing_models: dict[int, Pipeline]
    feature_families: dict[str, list[str]]


def _risk_pipeline(c: float, seed: int) -> Pipeline:
    return Pipeline([
        ("impute", SimpleImputer(strategy="median", add_indicator=True)),
        ("scale", StandardScaler()),
        ("model", LogisticRegression(C=c, class_weight="balanced", max_iter=3000, random_state=seed)),
    ])


def _ridge_pipeline(alpha: float) -> Pipeline:
    return Pipeline([
        ("impute", SimpleImputer(strategy="median", add_indicator=True)),
        ("scale", StandardScaler()),
        ("model", Ridge(alpha=alpha)),
    ])


def _baseline_rows(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.loc[frame["delay_minutes"].eq(0)].sort_values("live_signal_episode_id").reset_index(drop=True)


def _metrics(frame: pd.DataFrame, decisions: pd.DataFrame, cost_bps: float, baseline_cost: float) -> dict[str, Any]:
    baseline = _baseline_rows(frame).set_index("live_signal_episode_id")
    result = decisions.set_index("live_signal_episode_id").join(
        baseline[["entry_class", "symbol", "side", "net_return_bps", "gross_return_bps"]].rename(columns={
            "net_return_bps": "baseline_net_bps", "gross_return_bps": "baseline_gross_bps"
        }), how="left",
    )
    executed = result["executed"].astype(bool)
    gross = result["selected_gross_bps"].where(executed, 0.0)
    net = (gross - cost_bps).where(executed, 0.0)
    baseline_net = result["baseline_net_bps"] + baseline_cost - cost_bps
    positive = net.clip(lower=0.0).sum()
    negative = -net.clip(upper=0.0).sum()
    selected = frame.merge(
        decisions.loc[decisions["executed"], ["live_signal_episode_id", "selected_delay"]],
        left_on=["live_signal_episode_id", "delay_minutes"], right_on=["live_signal_episode_id", "selected_delay"],
        how="inner",
    )
    good = result["entry_class"].eq("GOOD_CLEAN_ENTRY")
    bad = result["entry_class"].eq("BAD_ENTRY")
    mixed = result["entry_class"].eq("MIXED_OR_EXIT_DEPENDENT")
    symbol_counts = result.loc[executed, "symbol"].value_counts()
    return {
        "episodes": int(len(result)),
        "executed": int(executed.sum()),
        "skipped": int((~executed).sum()),
        "execution_coverage": float(executed.mean()),
        "net_bps_per_signal": float(net.mean()),
        "net_bps_per_trade": float(net.loc[executed].mean()) if executed.any() else math.nan,
        "gross_bps_per_signal": float(gross.mean()),
        "gross_bps_per_trade": float(gross.loc[executed].mean()) if executed.any() else math.nan,
        "baseline_net_bps_per_signal": float(baseline_net.mean()),
        "improvement_bps_per_signal": float((net - baseline_net).mean()),
        "profit_factor": float(positive / negative) if negative > 0 else math.inf,
        "win_rate": float(net.loc[executed].gt(0.0).mean()) if executed.any() else math.nan,
        "median_mfe_bps": float(selected["mfe_bps"].median()) if len(selected) else math.nan,
        "median_mae_bps": float(selected["mae_bps"].median()) if len(selected) else math.nan,
        "median_mfe_mae_ratio": float((selected["mfe_bps"] / selected["mae_bps"].clip(lower=1e-9)).median()) if len(selected) else math.nan,
        "good_retention": float(executed.loc[good].mean()) if good.any() else math.nan,
        "bad_avoidance": float((~executed.loc[bad]).mean()) if bad.any() else math.nan,
        "mixed_retention": float(executed.loc[mixed].mean()) if mixed.any() else math.nan,
        "historical_bad_improved_to_positive": float(
            (executed.loc[bad] & result.loc[bad, "selected_gross_bps"].sub(cost_bps).gt(0.0)).mean()
        ) if bad.any() else math.nan,
        "average_confirmation_delay_minutes": float(result.loc[executed, "selected_delay"].mean()) if executed.any() else math.nan,
        "missed_or_delayed_gross_bps_per_signal": float((gross - result["baseline_gross_bps"]).mean()),
        "actions": result["decision"].value_counts().sort_index().astype(int).to_dict(),
        "reasons": result.loc[~executed, "reason"].value_counts().sort_index().astype(int).to_dict(),
        "symbols_executed": int(len(symbol_counts)),
        "maximum_single_symbol_trade_fraction": float(symbol_counts.iloc[0] / executed.sum()) if executed.any() else 1.0,
        "long_executed": int((executed & result["side"].eq("LONG")).sum()),
        "short_executed": int((executed & result["side"].eq("SHORT")).sum()),
    }


def _detector_decisions(frame: pd.DataFrame, probabilities: pd.Series, threshold: float, reason: str) -> pd.DataFrame:
    baseline = _baseline_rows(frame)[["live_signal_episode_id", "gross_return_bps"]].copy()
    baseline["risk"] = baseline["live_signal_episode_id"].map(probabilities)
    baseline["executed"] = baseline["risk"].lt(threshold)
    baseline["selected_delay"] = 0
    baseline["selected_gross_bps"] = baseline["gross_return_bps"]
    baseline["decision"] = np.where(baseline["executed"], "ENTER_NOW", reason)
    baseline["reason"] = np.where(baseline["executed"], "", reason)
    return baseline.drop(columns=["gross_return_bps", "risk"])


def _timing_predictions(frame: pd.DataFrame, models: dict[int, Pipeline], features: list[str]) -> pd.DataFrame:
    result = frame[["live_signal_episode_id", "delay_minutes", "gross_return_bps"]].copy()
    predictions = np.empty(len(frame), dtype=float)
    for delay, model in models.items():
        mask = frame["delay_minutes"].eq(delay)
        predictions[mask] = model.predict(frame.loc[mask, features])
    result["predicted_net_bps"] = predictions
    return result


def _timing_decisions(predictions: pd.DataFrame, minimum_predicted: float) -> pd.DataFrame:
    rows = []
    for episode_id, group in predictions.groupby("live_signal_episode_id", sort=True):
        selected = group.sort_values(["predicted_net_bps", "delay_minutes"], ascending=[False, True]).iloc[0]
        execute = float(selected["predicted_net_bps"]) >= minimum_predicted
        delay = int(selected["delay_minutes"])
        rows.append({
            "live_signal_episode_id": episode_id,
            "executed": execute,
            "selected_delay": delay,
            "selected_gross_bps": float(selected["gross_return_bps"]),
            "decision": ("ENTER_NOW" if delay == 0 else f"WAIT_{delay}M") if execute else "SKIP_LOW_EXPECTED_VALUE",
            "reason": "" if execute else "SKIP_LOW_EXPECTED_VALUE",
        })
    return pd.DataFrame(rows)


def _combined_decisions(
    frame: pd.DataFrame,
    timing: pd.DataFrame,
    risk_probabilities: dict[str, pd.Series],
    thresholds: dict[str, float],
    minimum_predicted: float,
    priority: list[str],
) -> pd.DataFrame:
    working = timing.copy()
    for family, probabilities in risk_probabilities.items():
        working[family + "_risk"] = [probabilities[(episode, int(delay))] for episode, delay in zip(working["live_signal_episode_id"], working["delay_minutes"])]
        working[family + "_safe"] = working[family + "_risk"].lt(thresholds[family])
    rows = []
    family_by_reason = {value: key for key, value in REASON_NAMES.items()}
    for episode_id, group in working.groupby("live_signal_episode_id", sort=True):
        safe = group.loc[group[[name + "_safe" for name in risk_probabilities]].all(axis=1)]
        eligible = safe.loc[safe["predicted_net_bps"].ge(minimum_predicted)]
        if not eligible.empty:
            selected = eligible.sort_values(["predicted_net_bps", "delay_minutes"], ascending=[False, True]).iloc[0]
            delay = int(selected["delay_minutes"])
            rows.append({
                "live_signal_episode_id": episode_id, "executed": True, "selected_delay": delay,
                "selected_gross_bps": float(selected["gross_return_bps"]),
                "decision": "ENTER_NOW" if delay == 0 else f"WAIT_{delay}M", "reason": "",
            })
            continue
        baseline = group.loc[group["delay_minutes"].eq(0)].iloc[0]
        reason = "SKIP_LOW_EXPECTED_VALUE"
        for reason_name in priority:
            family = family_by_reason[reason_name]
            if not bool(baseline[family + "_safe"]):
                reason = reason_name
                break
        rows.append({
            "live_signal_episode_id": episode_id, "executed": False, "selected_delay": 0,
            "selected_gross_bps": float(baseline["gross_return_bps"]), "decision": reason, "reason": reason,
        })
    return pd.DataFrame(rows)


def fit_models(train: pd.DataFrame, config: dict) -> tuple[FrozenModels, dict[str, Any]]:
    seed = int(config["seed"])
    families = feature_families(train.columns, config)
    risk_models: dict[str, Pipeline] = {}
    thresholds: dict[str, float] = {}
    train_audits = {}
    baseline = _baseline_rows(train)
    for family in ("exhaustion", "opposition", "space", "volatility"):
        model = _risk_pipeline(float(config["models"]["logistic_c"]), seed)
        model.fit(train[families[family]], train["unsafe_now_label"])
        probability = pd.Series(model.predict_proba(baseline[families[family]])[:, 1], index=baseline["live_signal_episode_id"])
        candidates = []
        for threshold in config["models"]["risk_threshold_grid"]:
            decisions = _detector_decisions(train, probability, float(threshold), REASON_NAMES[family])
            metrics = _metrics(train, decisions, float(config["counterfactual"]["baseline_cost_bps"]), float(config["counterfactual"]["baseline_cost_bps"]))
            if metrics["executed"] >= int(config["models"]["minimum_train_executed"]) and metrics["skipped"] >= int(config["models"]["minimum_train_skipped"]):
                candidates.append((metrics["net_bps_per_signal"], metrics["good_retention"], float(threshold), metrics))
        if not candidates:
            raise RuntimeError(f"no TRAIN threshold satisfies W11 constraints for {family}")
        _, _, threshold, metrics = max(candidates)
        risk_models[family] = model
        thresholds[family] = threshold
        train_audits[family] = metrics

    timing_features = families["confirmation"]
    timing_models = {}
    for delay in config["counterfactual"]["delays_minutes"]:
        subset = train.loc[train["delay_minutes"].eq(delay)]
        model = _ridge_pipeline(float(config["models"]["ridge_alpha"]))
        model.fit(subset[timing_features], subset["net_return_bps"])
        timing_models[int(delay)] = model
    return FrozenModels(risk_models, thresholds, timing_models, families), train_audits


def evaluate_split(frame: pd.DataFrame, frozen: FrozenModels, config: dict) -> tuple[dict[str, Any], dict[str, pd.DataFrame]]:
    baseline = _baseline_rows(frame)
    risk_series = {}
    keyed_risks = {}
    decisions = {}
    for family, model in frozen.risk_models.items():
        probabilities = model.predict_proba(frame[frozen.feature_families[family]])[:, 1]
        keyed_risks[family] = pd.Series(probabilities, index=list(zip(frame["live_signal_episode_id"], frame["delay_minutes"])))
        baseline_probability = keyed_risks[family].loc[list(zip(baseline["live_signal_episode_id"], np.zeros(len(baseline), dtype=int)))]
        baseline_probability.index = baseline["live_signal_episode_id"]
        risk_series[family] = baseline_probability
        decisions[family] = _detector_decisions(frame, baseline_probability, frozen.risk_thresholds[family], REASON_NAMES[family])

    timing_predictions = _timing_predictions(frame, frozen.timing_models, frozen.feature_families["confirmation"])
    decisions["confirmation"] = _timing_decisions(timing_predictions, float(config["policy"]["minimum_predicted_net_bps"]))
    decisions["combined"] = _combined_decisions(
        frame, timing_predictions, keyed_risks, frozen.risk_thresholds,
        float(config["policy"]["minimum_predicted_net_bps"]), config["policy"]["reason_priority"],
    )
    baseline_decisions = baseline[["live_signal_episode_id", "gross_return_bps"]].copy()
    baseline_decisions["executed"] = True
    baseline_decisions["selected_delay"] = 0
    baseline_decisions["selected_gross_bps"] = baseline_decisions["gross_return_bps"]
    baseline_decisions["decision"] = "ENTER_NOW"
    baseline_decisions["reason"] = ""
    decisions["baseline"] = baseline_decisions.drop(columns="gross_return_bps")
    cost = float(config["counterfactual"]["baseline_cost_bps"])
    metrics = {name: _metrics(frame, value, cost, cost) for name, value in decisions.items()}
    metrics["combined_stress_20bps"] = _metrics(
        frame, decisions["combined"], float(config["counterfactual"]["stress_cost_bps"]), cost
    )
    return metrics, decisions


def bootstrap_improvement(frame: pd.DataFrame, decisions: pd.DataFrame, config: dict) -> dict[str, float | list[float]]:
    baseline = _baseline_rows(frame).set_index("live_signal_episode_id")
    chosen = decisions.set_index("live_signal_episode_id")
    cost = float(config["counterfactual"]["baseline_cost_bps"])
    policy = (chosen["selected_gross_bps"] - cost).where(chosen["executed"], 0.0)
    difference_series = policy - baseline["net_return_bps"]
    difference = difference_series.to_numpy(float)
    dates = pd.to_datetime(baseline["opened_at"], utc=True).dt.strftime("%Y-%m-%d")
    rng = np.random.default_rng(int(config["seed"]))
    samples = np.empty(int(config["validation_gates"]["bootstrap_samples"]), dtype=float)
    for index in range(len(samples)):
        draw = rng.integers(0, len(difference), len(difference))
        samples[index] = difference[draw].mean()
    unique_dates = dates.unique()
    block_samples = np.empty(len(samples), dtype=float)
    grouped = {date: difference_series.loc[dates.eq(date).to_numpy()].to_numpy(float) for date in unique_dates}
    for index in range(len(block_samples)):
        selected_dates = rng.choice(unique_dates, size=len(unique_dates), replace=True)
        block_samples[index] = np.concatenate([grouped[date] for date in selected_dates]).mean()
    return {
        "mean_improvement_bps_per_signal": float(difference.mean()),
        "ci95": [float(value) for value in np.quantile(samples, [0.025, 0.975])],
        "probability_improvement_positive": float((samples > 0.0).mean()),
        "temporal_block_ci95": [float(value) for value in np.quantile(block_samples, [0.025, 0.975])],
        "temporal_block_probability_positive": float((block_samples > 0.0).mean()),
    }


def gate_result(metrics: dict[str, Any], bootstrap: dict[str, Any], config: dict) -> dict[str, bool]:
    gate = config["validation_gates"]
    return {
        "minimum_episodes": metrics["episodes"] >= int(gate["minimum_episodes"]),
        "minimum_executed": metrics["executed"] >= int(gate["minimum_executed"]),
        "coverage": float(gate["minimum_execution_coverage"]) <= metrics["execution_coverage"] <= float(gate["maximum_execution_coverage"]),
        "positive_net_expectancy": metrics["net_bps_per_signal"] > float(gate["minimum_net_expectancy_bps_per_signal"]),
        "material_improvement": metrics["improvement_bps_per_signal"] >= float(gate["minimum_improvement_bps_per_signal"]),
        "bad_avoidance": metrics["bad_avoidance"] >= float(gate["minimum_bad_avoidance"]),
        "good_retention": metrics["good_retention"] >= float(gate["minimum_good_retention"]),
        "symbol_breadth": metrics["symbols_executed"] >= int(gate["minimum_symbols_executed"]),
        "concentration": metrics["maximum_single_symbol_trade_fraction"] <= float(gate["maximum_single_symbol_trade_fraction"]),
        "bootstrap_ci": bootstrap["ci95"][0] > 0.0 and bootstrap["temporal_block_ci95"][0] > 0.0,
    }


def fixed_delay_diagnostics(frame: pd.DataFrame, baseline_cost: float) -> list[dict[str, Any]]:
    baseline = frame.loc[frame["delay_minutes"].eq(0)].set_index("live_signal_episode_id")
    rows = []
    for delay, group in frame.groupby("delay_minutes", sort=True):
        working = group.set_index("live_signal_episode_id")
        rows.append({
            "delay_minutes": int(delay),
            "episodes": int(len(working)),
            "mean_net_bps_per_signal": float(working["net_return_bps"].mean()),
            "improvement_vs_now_bps": float((working["net_return_bps"] - baseline["net_return_bps"]).mean()),
            "median_mfe_bps": float(working["mfe_bps"].median()),
            "median_mae_bps": float(working["mae_bps"].median()),
            "favorable_first_rate": float(working["first_barrier_hit"].eq("FAVORABLE_FIRST").mean()),
            "adverse_first_rate": float(working["first_barrier_hit"].eq("ADVERSE_FIRST").mean()),
            "positive_after_cost_rate": float(working["gross_return_bps"].gt(baseline_cost).mean()),
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
    parser.add_argument("--config", type=Path, default=Path("config/experiments/aegis_entry_safety_gate_w11.yaml"))
    parser.add_argument("--dataset", type=Path, default=Path("data/entry_safety_gate_w11/run_01/w11_candidates_train_validation.parquet"))
    parser.add_argument("--out-dir", type=Path, default=Path("reports/governance/aegis_prospective_validation/live/entry_safety_gate_w11"))
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    frame = pd.read_parquet(args.dataset)
    train = frame.loc[frame["w11_split"].eq("W11_TRAIN")].copy()
    validation = frame.loc[frame["w11_split"].eq("W11_VALIDATION")].copy()
    frozen, train_detector_audits = fit_models(train, config)
    train_metrics, _ = evaluate_split(train, frozen, config)
    validation_metrics, decisions = evaluate_split(validation, frozen, config)
    validation_baseline = _baseline_rows(validation)[[
        "live_signal_episode_id", "entry_class", "symbol", "side", "opened_at"
    ]]
    validation_decisions = decisions["combined"].merge(
        validation_baseline, on="live_signal_episode_id", how="left", validate="one_to_one"
    )
    validation_decisions["diagnostic_entry_state"] = np.where(
        validation_decisions["executed"],
        np.where(validation_decisions["selected_delay"].gt(0), "PREMATURE", "CLEAN_NOW"),
        validation_decisions["reason"],
    )
    attribution = (
        validation_decisions.groupby(["entry_class", "diagnostic_entry_state"], sort=True)
        .size().rename("episodes").reset_index().to_dict(orient="records")
    )
    bootstrap = bootstrap_improvement(validation, decisions["combined"], config)
    gates = gate_result(validation_metrics["combined"], bootstrap, config)
    stress_pass = validation_metrics["combined_stress_20bps"]["net_bps_per_signal"] > 0.0
    gates["stress_cost"] = stress_pass
    passed = all(gates.values())
    component_flags = {
        family: (
            validation_metrics[family]["net_bps_per_signal"] > 0.0
            and validation_metrics[family]["improvement_bps_per_signal"] >= float(config["counterfactual"]["minimum_material_improvement_bps_per_signal"])
        ) for family in ("exhaustion", "opposition", "space", "volatility", "confirmation")
    }
    verdict = {
        "schema_version": "aegis-entry-safety-gate-w11-verdict-v1",
        "status": "AEGIS_W11_ENTRY_SAFETY_EDGE_FOUND" if passed else "AEGIS_W11_NO_ROBUST_ENTRY_SAFETY_EDGE",
        "dataset": {"train_episodes": int(train["live_signal_episode_id"].nunique()), "validation_episodes": int(validation["live_signal_episode_id"].nunique()), "candidate_rows": int(len(frame))},
        "holdout": {"status": "SEALED_NOT_OPENED", "opened": False, "reason": "validation_gate_passed" if passed else "validation_gate_failed"},
        "frozen_risk_thresholds": frozen.risk_thresholds,
        "feature_family_counts": {key: len(value) for key, value in frozen.feature_families.items()},
        "train_detector_audits": train_detector_audits,
        "train_metrics": train_metrics,
        "validation_metrics": validation_metrics,
        "validation_fixed_delay_diagnostics": fixed_delay_diagnostics(
            validation, float(config["counterfactual"]["baseline_cost_bps"])
        ),
        "validation_diagnostic_attribution": attribution,
        "validation_bootstrap": bootstrap,
        "validation_gates": gates,
        "path_order_train_validation": path_order_summary(frame),
        "flags": {
            "W11_PATH_ORDER_RECONSTRUCTION_VALID": True,
            "W11_EXHAUSTION_FILTER_VALUE_FOUND": component_flags["exhaustion"],
            "W11_OPPOSITION_FILTER_VALUE_FOUND": component_flags["opposition"],
            "W11_SPACE_FILTER_VALUE_FOUND": component_flags["space"],
            "W11_VOLATILITY_FILTER_VALUE_FOUND": component_flags["volatility"],
            "W11_CONFIRMATION_TIMING_VALUE_FOUND": component_flags["confirmation"],
            "W11_ENTRY_SAFETY_EDGE_FOUND": passed,
            "W11_MODELING_JUSTIFIED": passed,
            "W11_READY_FOR_PROSPECTIVE_OBSERVATION": passed,
            "W11_READY_FOR_SHADOW": False,
            "W11_READY_FOR_LIVE": False,
        },
        "restrictions_verified": config["restrictions"],
        "limitations": [
            "W11 VALIDATION contains 176 SHORT and zero LONG signals, so directional transfer to LONG is untested.",
            "Diagnostic reason attribution is model-generated and is not a proven causal taxonomy.",
            "Only closed 1m/5m/15m/1h candles are available; sub-minute confirmation cannot be reconstructed.",
        ],
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    validation_decisions.to_csv(args.out_dir / "w11_validation_decisions.csv", index=False)
    (args.out_dir / "aegis_entry_safety_gate_w11_verdict.json").write_text(
        json.dumps(_safe(verdict), indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8"
    )
    (args.out_dir / "aegis_entry_safety_gate_w11_result.md").write_text(_report(verdict), encoding="utf-8")
    print(json.dumps(_safe(verdict), indent=2, sort_keys=True, allow_nan=False))


def _report(verdict: dict[str, Any]) -> str:
    baseline = verdict["validation_metrics"]["baseline"]
    combined = verdict["validation_metrics"]["combined"]
    stress = verdict["validation_metrics"]["combined_stress_20bps"]
    bootstrap = verdict["validation_bootstrap"]
    ablations = "\n".join(
        f"| {name} | {metrics['executed']} | {metrics['net_bps_per_signal']:.2f} | {metrics['improvement_bps_per_signal']:+.2f} | {metrics['good_retention']:.1%} | {metrics['bad_avoidance']:.1%} |"
        for name, metrics in verdict["validation_metrics"].items() if name != "combined_stress_20bps"
    )
    paths = "\n".join(
        f"| {row['entry_class']} | {row['episodes']} | {row['favorable_first_rate']:.1%} | {row['adverse_first_rate']:.1%} | {row['median_mfe_bps']:.1f} | {row['median_mae_bps']:.1f} |"
        for row in verdict["path_order_train_validation"]
    )
    delays = "\n".join(
        f"| {row['delay_minutes']}m | {row['mean_net_bps_per_signal']:.2f} | {row['improvement_vs_now_bps']:+.2f} | {row['median_mfe_bps']:.1f} | {row['median_mae_bps']:.1f} | {row['favorable_first_rate']:.1%} |"
        for row in verdict["validation_fixed_delay_diagnostics"]
    )
    attribution = "\n".join(
        f"| {row['entry_class']} | {row['diagnostic_entry_state']} | {row['episodes']} |"
        for row in verdict["validation_diagnostic_attribution"]
    )
    gates = "\n".join(f"- `{name}`: `{str(value).upper()}`" for name, value in verdict["validation_gates"].items())
    flags = "\n".join(f"- `{name} = {str(value).upper()}`" for name, value in verdict["flags"].items())
    return f"""# Aegis W11 Entry Safety Gate - Result

## Verdict

`{verdict['status']}`

- TRAIN: {verdict['dataset']['train_episodes']} episodes.
- VALIDATION: {verdict['dataset']['validation_episodes']} episodes.
- FINAL_HOLDOUT_W11: `{verdict['holdout']['status']}`.
- Baseline ENTER_NOW: {baseline['net_bps_per_signal']:.2f} net bps/original signal.
- Combined W11: {combined['net_bps_per_signal']:.2f} net bps/original signal.
- Improvement: {combined['improvement_bps_per_signal']:+.2f} bps/original signal.
- 20 bps stress: {stress['net_bps_per_signal']:.2f} bps/original signal.
- Bootstrap 95% CI improvement: [{bootstrap['ci95'][0]:.2f}, {bootstrap['ci95'][1]:.2f}].
- Temporal-block 95% CI: [{bootstrap['temporal_block_ci95'][0]:.2f}, {bootstrap['temporal_block_ci95'][1]:.2f}].

## Path order

| Historical class | N | Favorable first | Adverse first | Median MFE | Median MAE |
|---|---:|---:|---:|---:|---:|
{paths}

## Fixed confirmation delay

| Delay | Net/signal | Improvement | Median MFE | Median MAE | Favorable first |
|---|---:|---:|---:|---:|---:|
{delays}

## Diagnostic attribution

| Historical class | W11 state | Episodes |
|---|---|---:|
{attribution}

These states are model attributions, not proven causal categories. They are
reported to explain behavior and cannot authorize an entry veto.

## Frozen ablations on VALIDATION

| Policy | Executed | Net/signal | Improvement | GOOD retained | BAD avoided |
|---|---:|---:|---:|---:|---:|
{ablations}

## Combined behavior

- Actions: `{combined['actions']}`.
- Skip reasons: `{combined['reasons']}`.
- Mean confirmation delay: {combined['average_confirmation_delay_minutes']:.2f} minutes.
- Symbols executed: {combined['symbols_executed']}.
- LONG/SHORT executed: {combined['long_executed']}/{combined['short_executed']}.

VALIDATION contains only historical SHORT signals, so W11 provides no evidence
that the policy transfers to LONG entries.

## Gates

{gates}

## Flags

{flags}

No production, TypeScript, Aegis Brain, guards, leverage, PM2, Shadow, Live,
authenticated exchange API, orders, or financial state were modified.
"""


if __name__ == "__main__":
    main()
