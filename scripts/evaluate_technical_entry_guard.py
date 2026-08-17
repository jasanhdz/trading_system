#!/usr/bin/env python3
"""Evaluate the frozen technical entry guard without touching Live systems."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from aegis.research.technical_entry_guard import apply_guard


POLICIES = {
    "exhaustion_only": "skip_exhausted",
    "opposition_only": "skip_opposed",
    "space_only": "skip_no_space",
    "volatility_only": "skip_volatility_shock",
}


def _safe_ratio(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator else math.nan


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, (float, np.floating)) and not math.isfinite(float(value)):
        return None
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def _metrics(frame: pd.DataFrame, execute: pd.Series, stress_cost_bps: float) -> dict[str, Any]:
    execute = execute.astype(bool)
    returns = pd.to_numeric(frame["net_return_bps"], errors="coerce")
    gross = pd.to_numeric(frame["gross_return_bps"], errors="coerce")
    realized = pd.to_numeric(frame["pnl_usdt"], errors="coerce").fillna(0.0)
    policy = returns.where(execute, 0.0)
    stress = (gross - stress_cost_bps).where(execute, 0.0)
    executed_returns = returns.loc[execute]
    positive = float(executed_returns.loc[executed_returns > 0].sum())
    negative = float(-executed_returns.loc[executed_returns < 0].sum())
    good = frame["entry_class"].eq("GOOD_CLEAN_ENTRY")
    bad = frame["entry_class"].eq("BAD_ENTRY")
    mixed = frame["entry_class"].eq("MIXED_OR_EXIT_DEPENDENT")
    skipped = ~execute
    return {
        "episodes": int(len(frame)),
        "executed": int(execute.sum()),
        "skipped": int(skipped.sum()),
        "execution_rate": float(execute.mean()),
        "net_bps_per_original_signal": float(policy.mean()),
        "net_bps_per_executed_trade": float(executed_returns.mean()) if execute.any() else math.nan,
        "stress_net_bps_per_original_signal": float(stress.mean()),
        "gross_bps_per_original_signal": float(gross.where(execute, 0.0).mean()),
        "profit_factor": _safe_ratio(positive, negative),
        "win_rate_executed": float(executed_returns.gt(0).mean()) if execute.any() else math.nan,
        "median_mfe_bps": float(frame.loc[execute, "mfe_bps"].median()) if execute.any() else math.nan,
        "median_mae_bps": float(frame.loc[execute, "mae_bps"].median()) if execute.any() else math.nan,
        "good_retained": _safe_ratio(int((good & execute).sum()), int(good.sum())),
        "bad_avoided": _safe_ratio(int((bad & skipped).sum()), int(bad.sum())),
        "mixed_retained": _safe_ratio(int((mixed & execute).sum()), int(mixed.sum())),
        "winning_entries_sacrificed": int((skipped & returns.gt(0)).sum()),
        "losing_entries_avoided": int((skipped & returns.lt(0)).sum()),
        "historical_pnl_usdt_retained": float(realized.where(execute, 0.0).sum()),
        "historical_loss_usdt_avoided": float(-realized.where(skipped & realized.lt(0), 0.0).sum()),
        "historical_profit_usdt_sacrificed": float(realized.where(skipped & realized.gt(0), 0.0).sum()),
        "symbols_executed": int(frame.loc[execute, "symbol"].nunique()),
        "long_executed": int((execute & frame["side"].eq("LONG")).sum()),
        "short_executed": int((execute & frame["side"].eq("SHORT")).sum()),
    }


def _bootstrap_improvement(
    baseline: np.ndarray, policy: np.ndarray, samples: int, seed: int
) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    delta = np.asarray(policy, dtype=float) - np.asarray(baseline, dtype=float)
    n = len(delta)
    estimates = np.empty(samples, dtype=float)
    for start in range(0, samples, 500):
        count = min(500, samples - start)
        indices = rng.integers(0, n, size=(count, n))
        estimates[start:start + count] = delta[indices].mean(axis=1)
    low, high = np.quantile(estimates, [0.025, 0.975])
    return {
        "mean_improvement_bps": float(delta.mean()),
        "ci95_low_bps": float(low),
        "ci95_high_bps": float(high),
        "probability_improvement_positive": float((estimates > 0).mean()),
    }


def _per_symbol(frame: pd.DataFrame, execute: pd.Series, stress_cost_bps: float) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for symbol, group in frame.groupby("symbol", sort=True):
        metrics = _metrics(group, execute.loc[group.index], stress_cost_bps)
        rows.append({"symbol": str(symbol), **metrics})
    return rows


def _markdown(verdict: dict[str, Any]) -> str:
    validation = verdict["validation"]["combined"]
    baseline = verdict["validation"]["baseline"]
    bootstrap = verdict["validation"]["bootstrap"]
    pooled = verdict["all_auditable_operations"]
    gates = verdict["gates"]
    lines = [
        "# Aegis Technical Entry Guard - Retrospective Result",
        "",
        "## Verdict",
        "",
        f"`{verdict['status']}`",
        "",
        f"- Auditable operations: {verdict['dataset']['episodes']} of {verdict['dataset']['paired_live_operations']} paired Live operations.",
        f"- W11 FINAL_HOLDOUT excluded and sealed: {verdict['dataset']['sealed_holdout_episodes']} operations.",
        f"- VALIDATION baseline: {baseline['net_bps_per_original_signal']:.2f} net bps/original signal.",
        f"- Guard: {validation['net_bps_per_original_signal']:.2f} net bps/original signal.",
        f"- Improvement: {bootstrap['mean_improvement_bps']:+.2f} bps, 95% CI [{bootstrap['ci95_low_bps']:.2f}, {bootstrap['ci95_high_bps']:.2f}].",
        f"- Executed: {validation['executed']}/{validation['episodes']} ({validation['execution_rate']:.1%}).",
        f"- GOOD retained: {validation['good_retained']:.1%}; BAD avoided: {validation['bad_avoided']:.1%}.",
        f"- Net per executed trade: {validation['net_bps_per_executed_trade']:.2f} bps.",
        f"- Stress net per signal: {validation['stress_net_bps_per_original_signal']:.2f} bps.",
        "",
        "## All Auditable Operations",
        "",
        f"- Frozen 60m baseline: {pooled['baseline']['net_bps_per_original_signal']:.2f} bps/signal.",
        f"- Frozen 60m guard: {pooled['combined']['net_bps_per_original_signal']:.2f} bps/signal.",
        f"- Logged PnL baseline: ${pooled['baseline']['historical_pnl_usdt_retained']:.2f}.",
        f"- Logged PnL retained by guard: ${pooled['combined']['historical_pnl_usdt_retained']:.2f}.",
        f"- Guard executions: {pooled['combined']['executed']}/{pooled['combined']['episodes']} "
        f"({pooled['combined']['execution_rate']:.1%}).",
        f"- Losing outcomes avoided: {pooled['combined']['losing_entries_avoided']}; "
        f"winning outcomes sacrificed: {pooled['combined']['winning_entries_sacrificed']}.",
        f"- Logged loss avoided: ${pooled['combined']['historical_loss_usdt_avoided']:.2f}; "
        f"logged profit sacrificed: ${pooled['combined']['historical_profit_usdt_sacrificed']:.2f}.",
        "",
        "| Split | Baseline net/signal | Guard net/signal | Guard net/trade | Execution |",
        "|---|---:|---:|---:|---:|",
        f"| TRAIN | {verdict['train']['baseline']['net_bps_per_original_signal']:.2f} | "
        f"{verdict['train']['combined']['net_bps_per_original_signal']:.2f} | "
        f"{verdict['train']['combined']['net_bps_per_executed_trade']:.2f} | "
        f"{verdict['train']['combined']['execution_rate']:.1%} |",
        f"| VALIDATION | {baseline['net_bps_per_original_signal']:.2f} | "
        f"{validation['net_bps_per_original_signal']:.2f} | "
        f"{validation['net_bps_per_executed_trade']:.2f} | {validation['execution_rate']:.1%} |",
        "",
        "## What Was Avoided",
        "",
        f"- Losing 60m outcomes avoided: {validation['losing_entries_avoided']}.",
        f"- Winning 60m outcomes sacrificed: {validation['winning_entries_sacrificed']}.",
        f"- Logged historical loss avoided: ${validation['historical_loss_usdt_avoided']:.2f}.",
        f"- Logged historical profit sacrificed: ${validation['historical_profit_usdt_sacrificed']:.2f}.",
        "",
        "## Validation Ablations",
        "",
        "| Policy | Executed | Net/signal | Net/trade | GOOD retained | BAD avoided |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name, metrics in verdict["validation"]["policies"].items():
        lines.append(
            f"| {name} | {metrics['executed']} | {metrics['net_bps_per_original_signal']:.2f} | "
            f"{metrics['net_bps_per_executed_trade']:.2f} | {metrics['good_retained']:.1%} | {metrics['bad_avoided']:.1%} |"
        )
    lines.extend(["", "## Skip Reasons", ""])
    for reason, count in verdict["validation"]["skip_reasons"].items():
        lines.append(f"- `{reason}`: {count}")
    lines.extend(["", "## Gates", ""])
    for name, passed in gates.items():
        lines.append(f"- `{name}`: `{str(passed).upper()}`")
    lines.extend([
        "",
        "## Interpretation",
        "",
        verdict["interpretation"],
        "",
        "This is a retrospective audit on previously reported W11 TRAIN/VALIDATION data, not fresh prospective evidence. "
        "No production guard, TypeScript logic, PM2 service, order, account, or sealed holdout was changed.",
    ])
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("config/experiments/aegis_technical_entry_guard_audit.yaml"))
    parser.add_argument(
        "--out-dir", type=Path,
        default=Path("reports/governance/aegis_prospective_validation/live/technical_entry_guard_audit"),
    )
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    source = Path(config["sources"]["w11_dataset"])
    frame = pd.read_parquet(source)
    frame = frame.loc[frame["delay_minutes"].eq(int(config["sources"]["evaluated_delay_minutes"]))].copy()
    frame["decision_timestamp"] = pd.to_datetime(frame["decision_timestamp"], utc=True, format="mixed")
    sealed_start = pd.Timestamp(config["sources"]["sealed_holdout_start"])
    if frame["decision_timestamp"].ge(sealed_start).any():
        raise RuntimeError("sealed W11 holdout rows were loaded")
    evaluated = apply_guard(frame, config["guard"])
    stress_cost = float(config["costs"]["stress_bps"])

    splits: dict[str, Any] = {}
    for split in ("W11_TRAIN", "W11_VALIDATION"):
        current = evaluated.loc[evaluated["w11_split"].eq(split)].copy()
        baseline_execute = pd.Series(True, index=current.index)
        combined_execute = current["action"].eq("ENTER")
        policies = {
            "baseline": _metrics(current, baseline_execute, stress_cost),
            "combined": _metrics(current, combined_execute, stress_cost),
        }
        for name, column in POLICIES.items():
            policies[name] = _metrics(current, ~current[column].astype(bool), stress_cost)
        baseline_returns = current["net_return_bps"].to_numpy(float)
        policy_returns = current["net_return_bps"].where(combined_execute, 0.0).to_numpy(float)
        splits[split] = {
            "baseline": policies["baseline"],
            "combined": policies["combined"],
            "policies": policies,
            "bootstrap": _bootstrap_improvement(
                baseline_returns, policy_returns,
                int(config["validation_gates"]["bootstrap_samples"]), int(config["seed"]),
            ),
            "skip_reasons": current.loc[~combined_execute, "reason"].value_counts().sort_index().astype(int).to_dict(),
            "per_symbol": _per_symbol(current, combined_execute, stress_cost),
        }

    validation = splits["W11_VALIDATION"]
    all_baseline_execute = pd.Series(True, index=evaluated.index)
    all_combined_execute = evaluated["action"].eq("ENTER")
    all_operations = {
        "baseline": _metrics(evaluated, all_baseline_execute, stress_cost),
        "combined": _metrics(evaluated, all_combined_execute, stress_cost),
        "bootstrap": _bootstrap_improvement(
            evaluated["net_return_bps"].to_numpy(float),
            evaluated["net_return_bps"].where(all_combined_execute, 0.0).to_numpy(float),
            int(config["validation_gates"]["bootstrap_samples"]), int(config["seed"]),
        ),
    }
    metrics = validation["combined"]
    bootstrap = validation["bootstrap"]
    gates_config = config["validation_gates"]
    gates = {
        "minimum_episodes": metrics["episodes"] >= int(gates_config["minimum_validation_episodes"]),
        "minimum_executed": metrics["executed"] >= int(gates_config["minimum_executed"]),
        "execution_rate": float(gates_config["minimum_execution_rate"]) <= metrics["execution_rate"] <= float(gates_config["maximum_execution_rate"]),
        "positive_net_expectancy": metrics["net_bps_per_original_signal"] > 0.0,
        "material_improvement": bootstrap["mean_improvement_bps"] >= float(gates_config["minimum_improvement_bps_per_original_signal"]),
        "good_retention": metrics["good_retained"] >= float(gates_config["minimum_good_retention"]),
        "bad_avoidance": metrics["bad_avoided"] >= float(gates_config["minimum_bad_avoidance"]),
        "symbol_breadth": metrics["symbols_executed"] >= int(gates_config["minimum_symbols_executed"]),
        "bootstrap_ci": bootstrap["ci95_low_bps"] > 0.0,
        "stress_cost": metrics["stress_net_bps_per_original_signal"] > 0.0,
    }
    passed = all(gates.values())
    status = "AEGIS_TECHNICAL_ENTRY_GUARD_RETROSPECTIVE_PASS" if passed else "AEGIS_TECHNICAL_ENTRY_GUARD_NO_ROBUST_EDGE"
    interpretation = (
        "The frozen guard passed every retrospective gate, but Live activation still requires fresh prospective observation."
        if passed else
        "The frozen technical conjunction did not produce robust positive economics. Its diagnostic warnings must not be promoted to hard Live vetoes."
    )
    verdict = {
        "schema_version": "aegis-technical-entry-guard-audit-verdict-v1",
        "status": status,
        "dataset": {
            "paired_live_operations": 718,
            "episodes": int(len(evaluated)),
            "train_episodes": int(evaluated["w11_split"].eq("W11_TRAIN").sum()),
            "validation_episodes": int(evaluated["w11_split"].eq("W11_VALIDATION").sum()),
            "sealed_holdout_episodes": 105,
            "holdout_status": config["sources"]["sealed_holdout_status"],
        },
        "frozen_guard": config["guard"],
        "train": splits["W11_TRAIN"],
        "validation": splits["W11_VALIDATION"],
        "all_auditable_operations": all_operations,
        "gates": gates,
        "interpretation": interpretation,
        "flags": {
            "TECHNICAL_ENTRY_GUARD_EDGE_FOUND": bool(passed),
            "TECHNICAL_ENTRY_GUARD_READY_FOR_PROSPECTIVE_OBSERVATION": bool(passed),
            "TECHNICAL_ENTRY_GUARD_READY_FOR_LIVE": False,
        },
        "restrictions_verified": config["restrictions"],
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    decisions = evaluated[[
        "live_signal_episode_id", "decision_timestamp", "symbol", "side", "w11_split", "entry_class",
        "net_return_bps", "pnl_usdt", "action", "reason", "risk_count", "rsi_exhausted",
        "ema_extended", "mature_move", "opposition_votes",
    ]]
    decisions.to_csv(args.out_dir / "technical_entry_guard_decisions.csv", index=False)
    safe_verdict = _json_safe(verdict)
    (args.out_dir / "aegis_technical_entry_guard_verdict.json").write_text(
        json.dumps(safe_verdict, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8"
    )
    (args.out_dir / "aegis_technical_entry_guard_result.md").write_text(_markdown(verdict), encoding="utf-8")
    print(json.dumps({"status": status, "validation": metrics, "gates": gates}, indent=2))


if __name__ == "__main__":
    main()
