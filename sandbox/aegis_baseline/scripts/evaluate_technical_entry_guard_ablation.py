#!/usr/bin/env python3
"""TRAIN-selected factor ablations and breakout confirmation audit."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
import yaml

from aegis.research.technical_entry_guard_ablation import add_policy_columns


def _ratio(a: int, b: int) -> float:
    return float(a / b) if b else math.nan


def metrics(frame: pd.DataFrame, execute: pd.Series, stress_cost: float) -> dict[str, Any]:
    execute = execute.astype(bool)
    net = pd.to_numeric(frame["net_return_bps"], errors="coerce")
    gross = pd.to_numeric(frame["gross_return_bps"], errors="coerce")
    policy = net.where(execute, 0.0)
    executed = net.loc[execute]
    good = frame["entry_class"].eq("GOOD_CLEAN_ENTRY")
    bad = frame["entry_class"].eq("BAD_ENTRY")
    gains = float(executed.loc[executed > 0].sum())
    losses = float(-executed.loc[executed < 0].sum())
    return {
        "episodes": int(len(frame)),
        "executed": int(execute.sum()),
        "execution_rate": float(execute.mean()),
        "net_bps_per_signal": float(policy.mean()),
        "net_bps_per_trade": float(executed.mean()) if len(executed) else math.nan,
        "stress_net_bps_per_signal": float((gross - stress_cost).where(execute, 0.0).mean()),
        "improvement_bps_per_signal": float(policy.mean() - net.mean()),
        "good_retained": _ratio(int((good & execute).sum()), int(good.sum())),
        "bad_avoided": _ratio(int((bad & ~execute).sum()), int(bad.sum())),
        "winning_outcomes_sacrificed": int((~execute & net.gt(0)).sum()),
        "losing_outcomes_avoided": int((~execute & net.lt(0)).sum()),
        "profit_factor": float(gains / losses) if losses else math.nan,
        "median_mfe_bps": float(frame.loc[execute, "mfe_bps"].median()) if execute.any() else math.nan,
        "median_mae_bps": float(frame.loc[execute, "mae_bps"].median()) if execute.any() else math.nan,
        "symbols": int(frame.loc[execute, "symbol"].nunique()),
    }


def eligible(value: dict[str, Any], selection: dict[str, Any]) -> bool:
    return (
        value["executed"] >= int(selection["minimum_train_executed"])
        and float(selection["minimum_train_execution_rate"]) <= value["execution_rate"] <= float(selection["maximum_train_execution_rate"])
        and value["good_retained"] >= float(selection["minimum_train_good_retention"])
        and value["bad_avoided"] >= float(selection["minimum_train_bad_avoidance"])
    )


def breakout_rows(frame: pd.DataFrame, predicate: Callable[[pd.Series], bool]) -> tuple[pd.DataFrame, pd.Series]:
    rows: list[pd.Series] = []
    execution: list[bool] = []
    for _, group in frame.sort_values("delay_minutes").groupby("live_signal_episode_id", sort=False):
        matches = group.loc[group.apply(predicate, axis=1)]
        if matches.empty:
            rows.append(group.loc[group["delay_minutes"].eq(0)].iloc[0])
            execution.append(False)
        else:
            rows.append(matches.iloc[0])
            execution.append(True)
    result = pd.DataFrame(rows).reset_index(drop=True)
    return result, pd.Series(execution, index=result.index)


BREAKOUTS: dict[str, Callable[[pd.Series], bool]] = {
    "breakout_1m": lambda row: float(row["dir1m__aligned_breakout"]) >= 1.0,
    "breakout_5m": lambda row: float(row["dir5m__aligned_breakout"]) >= 1.0,
    "breakout_any": lambda row: max(float(row["dir1m__aligned_breakout"]), float(row["dir5m__aligned_breakout"])) >= 1.0,
    "breakout_any_with_flow": lambda row: (
        max(float(row["dir1m__aligned_breakout"]), float(row["dir5m__aligned_breakout"])) >= 1.0
        and float(row["dir1m__taker_imbalance"]) > 0.0
    ),
}


def bootstrap(frame: pd.DataFrame, execute: pd.Series, samples: int, seed: int) -> dict[str, float]:
    baseline = frame["net_return_bps"].to_numpy(float)
    policy = frame["net_return_bps"].where(execute, 0.0).to_numpy(float)
    delta = policy - baseline
    rng = np.random.default_rng(seed)
    estimates = np.empty(samples)
    for start in range(0, samples, 500):
        count = min(500, samples - start)
        indices = rng.integers(0, len(delta), size=(count, len(delta)))
        estimates[start:start + count] = delta[indices].mean(axis=1)
    low, high = np.quantile(estimates, [0.025, 0.975])
    return {
        "mean": float(delta.mean()),
        "ci95_low": float(low),
        "ci95_high": float(high),
        "p_one_sided": float((1 + np.sum(estimates <= 0.0)) / (samples + 1)),
    }


def bh_adjust(values: dict[str, float]) -> dict[str, float]:
    ordered = sorted(values, key=values.get)
    n = len(ordered)
    adjusted: dict[str, float] = {}
    running = 1.0
    for rank in range(n, 0, -1):
        name = ordered[rank - 1]
        running = min(running, values[name] * n / rank)
        adjusted[name] = float(min(1.0, running))
    return adjusted


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_safe(item) for item in value]
    if isinstance(value, (float, np.floating)) and not math.isfinite(float(value)):
        return None
    return value


def table(rows: dict[str, dict[str, Any]]) -> list[str]:
    output = [
        "| Policy | Executed | Net/signal | Net/trade | Improvement | GOOD retained | BAD avoided |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name, value in rows.items():
        output.append(
            f"| {name} | {value['executed']} | {value['net_bps_per_signal']:.2f} | "
            f"{value['net_bps_per_trade']:.2f} | {value['improvement_bps_per_signal']:+.2f} | "
            f"{value['good_retained']:.1%} | {value['bad_avoided']:.1%} |"
        )
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("config/experiments/aegis_technical_entry_guard_ablation.yaml"))
    parser.add_argument("--out-dir", type=Path, default=Path("reports/governance/aegis_prospective_validation/live/technical_entry_guard_ablation"))
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    data = pd.read_parquet(config["sources"]["dataset"])
    timestamps = pd.to_datetime(data["decision_timestamp"], utc=True, format="mixed")
    if timestamps.ge(pd.Timestamp(config["sources"]["sealed_holdout_start"])).any():
        raise RuntimeError("sealed holdout loaded")
    baseline = data.loc[data["delay_minutes"].eq(0)].copy().reset_index(drop=True)
    baseline = add_policy_columns(baseline, config["thresholds"])
    train = baseline.loc[baseline["w11_split"].eq(config["sources"]["train_split"])].copy()
    validation = baseline.loc[baseline["w11_split"].eq(config["sources"]["validation_split"])].copy()
    policy_names = [name for name in baseline.columns if name in {
        "ema_only", "rsi_only", "structure_only", "rsi_and_structure", "rsi_and_extension",
        "ema_and_structure", "ema_and_price", "ema_and_flow", "price_and_flow",
        "ema_price_flow", "late_and_structure", "volatility_only",
    }]
    stress = float(config["validation"]["stress_cost_bps"])
    train_t0 = {name: metrics(train, ~train[name].astype(bool), stress) for name in policy_names}
    qualified_t0 = [name for name in policy_names if eligible(train_t0[name], config["selection"])]
    selected_t0 = sorted(qualified_t0, key=lambda name: train_t0[name]["net_bps_per_signal"], reverse=True)[
        :int(config["selection"]["maximum_selected_t0_policies"])
    ]

    train_all = data.loc[data["w11_split"].eq(config["sources"]["train_split"])].copy()
    validation_all = data.loc[data["w11_split"].eq(config["sources"]["validation_split"])].copy()
    train_breakout: dict[str, dict[str, Any]] = {}
    breakout_train_cache: dict[str, tuple[pd.DataFrame, pd.Series]] = {}
    for name, predicate in BREAKOUTS.items():
        selected_rows, execute = breakout_rows(train_all, predicate)
        breakout_train_cache[name] = (selected_rows, execute)
        train_breakout[name] = metrics(selected_rows, execute, stress)
    qualified_breakout = [name for name in BREAKOUTS if eligible(train_breakout[name], config["selection"])]
    selected_breakout = sorted(
        qualified_breakout, key=lambda name: train_breakout[name]["net_bps_per_signal"], reverse=True
    )[:int(config["selection"]["maximum_selected_breakout_policies"])]

    frozen = {"selected_t0": selected_t0, "selected_breakout": selected_breakout, "selection_source": "W11_TRAIN_ONLY"}
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "train_selected_candidates.json").write_text(json.dumps(frozen, indent=2) + "\n", encoding="utf-8")

    validation_t0 = {name: metrics(validation, ~validation[name].astype(bool), stress) for name in policy_names}
    validation_breakout: dict[str, dict[str, Any]] = {}
    validation_breakout_cache: dict[str, tuple[pd.DataFrame, pd.Series]] = {}
    for name, predicate in BREAKOUTS.items():
        selected_rows, execute = breakout_rows(validation_all, predicate)
        validation_breakout_cache[name] = (selected_rows, execute)
        validation_breakout[name] = metrics(selected_rows, execute, stress)

    selected_results: dict[str, dict[str, Any]] = {}
    p_values: dict[str, float] = {}
    for name in selected_t0:
        execute = ~validation[name].astype(bool)
        boot = bootstrap(validation, execute, int(config["validation"]["bootstrap_samples"]), int(config["seed"]))
        selected_results[name] = {"family": "T0_FILTER", "metrics": validation_t0[name], "bootstrap": boot}
        p_values[name] = boot["p_one_sided"]
    for name in selected_breakout:
        selected_rows, execute = validation_breakout_cache[name]
        boot = bootstrap(selected_rows, execute, int(config["validation"]["bootstrap_samples"]), int(config["seed"]))
        selected_results[name] = {"family": "PENDING_BREAKOUT", "metrics": validation_breakout[name], "bootstrap": boot}
        p_values[name] = boot["p_one_sided"]
    adjusted = bh_adjust(p_values) if p_values else {}

    gate_config = config["validation"]
    passing: list[str] = []
    for name, result in selected_results.items():
        value, boot = result["metrics"], result["bootstrap"]
        gates = {
            "minimum_executed": value["executed"] >= int(gate_config["minimum_validation_executed"]),
            "execution_rate": value["execution_rate"] >= float(gate_config["minimum_validation_execution_rate"]),
            "positive_per_signal": value["net_bps_per_signal"] > float(gate_config["minimum_net_bps_per_signal"]),
            "positive_per_trade": value["net_bps_per_trade"] > float(gate_config["minimum_net_bps_per_trade"]),
            "material_improvement": value["improvement_bps_per_signal"] >= float(gate_config["minimum_improvement_bps_per_signal"]),
            "good_retention": value["good_retained"] >= float(gate_config["minimum_good_retention"]),
            "bad_avoidance": value["bad_avoided"] >= float(gate_config["minimum_bad_avoidance"]),
            "stress_positive": value["stress_net_bps_per_signal"] > 0.0,
            "bootstrap_ci": boot["ci95_low"] > 0.0,
            "fdr": adjusted[name] <= 0.05,
        }
        result["fdr_q"] = adjusted[name]
        result["gates"] = gates
        if all(gates.values()):
            passing.append(name)

    status = "AEGIS_TECHNICAL_ENTRY_GUARD_ABLATION_EDGE_FOUND" if passing else "AEGIS_TECHNICAL_ENTRY_GUARD_ABLATION_NO_EDGE"
    verdict = {
        "schema_version": "aegis-technical-entry-guard-ablation-verdict-v1",
        "status": status,
        "dataset": {"train": len(train), "validation": len(validation), "holdout": config["sources"]["sealed_holdout_status"]},
        "frozen_selection": frozen,
        "train_t0": train_t0,
        "train_breakout": train_breakout,
        "validation_t0_diagnostic_all": validation_t0,
        "validation_breakout_diagnostic_all": validation_breakout,
        "selected_validation_results": selected_results,
        "passing_candidates": passing,
        "flags": {
            "TECHNICAL_FACTOR_EDGE_FOUND": bool(passing),
            "BREAKOUT_CONFIRMATION_EDGE_FOUND": any(selected_results[name]["family"] == "PENDING_BREAKOUT" for name in passing),
            "READY_FOR_PROSPECTIVE_OBSERVATION": bool(passing),
            "READY_FOR_LIVE": False,
        },
        "restrictions_verified": config["restrictions"],
    }
    lines = [
        "# Technical Entry Guard Factor Ablation - Result", "", "## Verdict", "", f"`{status}`", "",
        f"- TRAIN: {len(train)}; VALIDATION: {len(validation)}.",
        f"- TRAIN-selected T0 policies: `{selected_t0}`.",
        f"- TRAIN-selected breakout policies: `{selected_breakout}`.",
        f"- Passing candidates: `{passing}`.", "", "## T0 Policies - TRAIN", "",
        *table(train_t0), "", "## T0 Policies - VALIDATION (diagnostic; only TRAIN-selected candidates are inferential)", "",
        *table(validation_t0), "", "## Breakout Policies - TRAIN", "", *table(train_breakout), "",
        "## Breakout Policies - VALIDATION", "", *table(validation_breakout), "", "## Selected Candidate Gates", "",
    ]
    for name, result in selected_results.items():
        lines.append(f"### {name}")
        lines.append("")
        lines.append(
            f"Net/signal {result['metrics']['net_bps_per_signal']:.2f}; net/trade "
            f"{result['metrics']['net_bps_per_trade']:.2f}; bootstrap CI "
            f"[{result['bootstrap']['ci95_low']:.2f}, {result['bootstrap']['ci95_high']:.2f}]; FDR q={result['fdr_q']:.4f}."
        )
        lines.append("")
        for gate, passed in result["gates"].items():
            lines.append(f"- `{gate}`: `{str(passed).upper()}`")
        lines.append("")
    lines.extend([
        "## Integrity", "",
        "This is retrospective and cannot authorize Live. W11 FINAL_HOLDOUT remains sealed. No production, TypeScript, PM2, account, order, or financial behavior was changed.", "",
    ])
    safe = json_safe(verdict)
    (args.out_dir / "aegis_technical_entry_guard_ablation_verdict.json").write_text(
        json.dumps(safe, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8"
    )
    (args.out_dir / "aegis_technical_entry_guard_ablation_result.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"status": status, "selected": frozen, "passing": passing}, indent=2))


if __name__ == "__main__":
    main()

