#!/usr/bin/env python3
"""Build and evaluate the frozen W14 contextual regime policy."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from aegis.research.contextual_regime_arbitration_w14 import add_context_columns, choose_episode_decision
from aegis.research.live_entry_multitimeframe import add_directional_context, attach_features


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, (float, np.floating)) and not math.isfinite(float(value)):
        return None
    if isinstance(value, (np.integer,)):
        return int(value)
    return value


def load_candles(directory: Path) -> dict[str, pd.DataFrame]:
    return {
        path.name.split("_1m.parquet")[0]: pd.read_parquet(path)
        for path in sorted(directory.glob("*_1m.parquet"))
    }


def add_wider_context(
    frame: pd.DataFrame, candles: dict[str, pd.DataFrame], timeframes: list[int]
) -> pd.DataFrame:
    entries = frame.copy()
    entries["opened_at"] = entries["decision_timestamp"]
    enriched = add_directional_context(attach_features(entries, candles, timeframes), timeframes)
    btc = frame.copy()
    btc["opened_at"] = btc["decision_timestamp"]
    btc["symbol"] = "BTCUSDT"
    btc = add_directional_context(attach_features(btc, candles, timeframes), timeframes)
    prefixes = tuple([f"tf{tf}m__" for tf in timeframes] + [f"dir{tf}m__" for tf in timeframes])
    btc_columns = [name for name in btc.columns if name.startswith(prefixes)]
    btc = btc[["live_signal_episode_id", "delay_minutes", *btc_columns]].rename(
        columns={name: f"btc_{name}" for name in btc_columns}
    )
    return enriched.merge(
        btc, on=["live_signal_episode_id", "delay_minutes"], how="left", validate="one_to_one"
    )


def _ratio(a: int, b: int) -> float:
    return float(a / b) if b else math.nan


def policy_metrics(frame: pd.DataFrame, execute: pd.Series, stress_cost: float) -> dict[str, Any]:
    execute = execute.astype(bool)
    net = pd.to_numeric(frame["net_return_bps"], errors="coerce")
    gross = pd.to_numeric(frame["gross_return_bps"], errors="coerce")
    selected = net.loc[execute]
    good = frame["entry_class"].eq("GOOD_CLEAN_ENTRY")
    bad = frame["entry_class"].eq("BAD_ENTRY")
    gains = float(selected.loc[selected > 0].sum())
    losses = float(-selected.loc[selected < 0].sum())
    return {
        "episodes": int(len(frame)),
        "executed": int(execute.sum()),
        "execution_rate": float(execute.mean()),
        "net_bps_per_signal": float(net.where(execute, 0.0).mean()),
        "net_bps_per_trade": float(selected.mean()) if len(selected) else math.nan,
        "gross_bps_per_signal": float(gross.where(execute, 0.0).mean()),
        "gross_bps_per_trade": float(gross.loc[execute].mean()) if execute.any() else math.nan,
        "gross_directional_win_rate": float(gross.loc[execute].gt(0.0).mean()) if execute.any() else math.nan,
        "mfe_greater_than_mae_rate": float(frame.loc[execute, "mfe_bps"].gt(frame.loc[execute, "mae_bps"]).mean()) if execute.any() else math.nan,
        "stress_net_bps_per_signal": float((gross - stress_cost).where(execute, 0.0).mean()),
        "improvement_bps_per_signal": float(net.where(execute, 0.0).mean() - net.mean()),
        "good_retained": _ratio(int((good & execute).sum()), int(good.sum())),
        "bad_avoided": _ratio(int((bad & ~execute).sum()), int(bad.sum())),
        "mixed_retained": _ratio(int((frame["entry_class"].eq("MIXED_OR_EXIT_DEPENDENT") & execute).sum()), int(frame["entry_class"].eq("MIXED_OR_EXIT_DEPENDENT").sum())),
        "winning_outcomes_sacrificed": int((~execute & net.gt(0)).sum()),
        "losing_outcomes_avoided": int((~execute & net.lt(0)).sum()),
        "median_mfe_bps": float(frame.loc[execute, "mfe_bps"].median()) if execute.any() else math.nan,
        "median_mae_bps": float(frame.loc[execute, "mae_bps"].median()) if execute.any() else math.nan,
        "profit_factor": float(gains / losses) if losses else math.nan,
        "symbols": int(frame.loc[execute, "symbol"].nunique()),
        "long_executed": int((execute & frame["side"].eq("LONG")).sum()),
        "short_executed": int((execute & frame["side"].eq("SHORT")).sum()),
        "average_delay_minutes": float(frame.loc[execute, "delay_minutes"].mean()) if execute.any() else math.nan,
    }


def state_atlas(frame: pd.DataFrame) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for state, group in frame.groupby("context_state", sort=True):
        rows.append({
            "state": str(state),
            "episodes": int(len(group)),
            "good_rate": float(group["entry_class"].eq("GOOD_CLEAN_ENTRY").mean()),
            "bad_rate": float(group["entry_class"].eq("BAD_ENTRY").mean()),
            "mean_net_bps": float(group["net_return_bps"].mean()),
            "median_net_bps": float(group["net_return_bps"].median()),
            "mean_gross_bps": float(group["gross_return_bps"].mean()),
            "median_gross_bps": float(group["gross_return_bps"].median()),
            "gross_directional_win_rate": float(group["gross_return_bps"].gt(0.0).mean()),
            "mfe_greater_than_mae_rate": float(group["mfe_bps"].gt(group["mae_bps"]).mean()),
            "median_mfe_bps": float(group["mfe_bps"].median()),
            "median_mae_bps": float(group["mae_bps"].median()),
            "favorable_first_rate": float(group["first_barrier_hit"].eq("FAVORABLE_FIRST").mean()),
            "adverse_first_rate": float(group["first_barrier_hit"].eq("ADVERSE_FIRST").mean()),
            "symbols": int(group["symbol"].nunique()),
            "long": int(group["side"].eq("LONG").sum()),
            "short": int(group["side"].eq("SHORT").sum()),
        })
    return rows


def choose_context_policy(
    frame: pd.DataFrame, thresholds: dict[str, Any], policy: dict[str, Any]
) -> tuple[pd.DataFrame, pd.Series]:
    selected: list[pd.Series] = []
    execute: list[bool] = []
    actions: list[str] = []
    initial_states: list[str] = []
    for _, episode in frame.groupby("live_signal_episode_id", sort=False):
        row, should_execute, action, state = choose_episode_decision(episode, thresholds, policy)
        selected.append(row)
        execute.append(should_execute)
        actions.append(action)
        initial_states.append(state)
    result = pd.DataFrame(selected).reset_index(drop=True)
    result["w14_action"] = actions
    result["initial_context_state"] = initial_states
    return result, pd.Series(execute, index=result.index)


def fixed_delay(frame: pd.DataFrame, delay: int) -> tuple[pd.DataFrame, pd.Series]:
    selected = frame.loc[frame["delay_minutes"].eq(delay)].copy().reset_index(drop=True)
    return selected, pd.Series(True, index=selected.index)


def static_context_policy(
    frame: pd.DataFrame, enter_states: set[str]
) -> tuple[pd.DataFrame, pd.Series]:
    selected = frame.loc[frame["delay_minutes"].eq(0)].copy().reset_index(drop=True)
    return selected, selected["context_state"].isin(enter_states)


def bootstrap_improvement(
    baseline: pd.DataFrame, selected: pd.DataFrame, execute: pd.Series, samples: int, seed: int
) -> dict[str, float]:
    base = baseline.sort_values("live_signal_episode_id").set_index("live_signal_episode_id")["net_return_bps"]
    policy = selected.assign(execute=execute.to_numpy()).sort_values("live_signal_episode_id").set_index("live_signal_episode_id")
    policy_values = policy["net_return_bps"].where(policy["execute"], 0.0).reindex(base.index)
    delta = policy_values.to_numpy(float) - base.to_numpy(float)
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
        "probability_positive": float((estimates > 0).mean()),
    }


def metric_table(values: dict[str, dict[str, Any]]) -> list[str]:
    rows = [
        "| Policy | Executed | Net/signal | Net/trade | Improvement | GOOD retained | BAD avoided | Delay |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, value in values.items():
        rows.append(
            f"| {name} | {value['executed']} | {value['net_bps_per_signal']:.2f} | "
            f"{value['net_bps_per_trade']:.2f} | {value['improvement_bps_per_signal']:+.2f} | "
            f"{value['good_retained']:.1%} | {value['bad_avoided']:.1%} | {value['average_delay_minutes']:.2f}m |"
        )
    return rows


def atlas_table(values: list[dict[str, Any]]) -> list[str]:
    rows = [
        "| State | N | GOOD | BAD | Mean net | Median MFE | Median MAE | Fav first |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for value in values:
        rows.append(
            f"| {value['state']} | {value['episodes']} | {value['good_rate']:.1%} | "
            f"{value['bad_rate']:.1%} | {value['mean_net_bps']:.2f} | {value['median_mfe_bps']:.2f} | "
            f"{value['median_mae_bps']:.2f} | {value['favorable_first_rate']:.1%} |"
        )
    return rows


def directional_atlas_table(values: list[dict[str, Any]]) -> list[str]:
    rows = [
        "| State | N | Gross mean | Gross median | Direction correct | MFE > MAE | Median MFE | Median MAE |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for value in values:
        rows.append(
            f"| {value['state']} | {value['episodes']} | {value['mean_gross_bps']:.2f} | "
            f"{value['median_gross_bps']:.2f} | {value['gross_directional_win_rate']:.1%} | "
            f"{value['mfe_greater_than_mae_rate']:.1%} | {value['median_mfe_bps']:.2f} | {value['median_mae_bps']:.2f} |"
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("config/experiments/aegis_contextual_regime_arbitration_w14.yaml"))
    parser.add_argument("--out-dir", type=Path, default=Path("reports/governance/aegis_prospective_validation/live/contextual_regime_arbitration_w14"))
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    data = pd.read_parquet(config["sources"]["w11_dataset"])
    timestamps = pd.to_datetime(data["decision_timestamp"], utc=True, format="mixed")
    if timestamps.ge(pd.Timestamp(config["sources"]["sealed_holdout_start"])).any():
        raise RuntimeError("sealed W11 holdout rows loaded")
    candles = load_candles(Path(config["sources"]["candle_dir"]))
    enriched = add_wider_context(data, candles, [int(value) for value in config["sources"]["added_timeframes_minutes"]])
    enriched = add_context_columns(enriched, config["thresholds"])
    baseline_rows = enriched.loc[enriched["delay_minutes"].eq(0)].copy()
    coverage_columns = [
        "dir240m__ema25_slope_atr", "dir240m__ema99_extension_atr", "dir1440m__return_1_bps",
        "dir1440m__rsi6_remaining_room", "dir1440m__ema25_slope_atr",
    ]
    coverage = {name: float(baseline_rows[name].notna().mean()) for name in coverage_columns}
    stress = float(config["economics"]["stress_cost_bps"])
    results: dict[str, Any] = {}
    decisions_by_split: list[pd.DataFrame] = []
    for split in ("W11_TRAIN", "W11_VALIDATION"):
        all_rows = enriched.loc[enriched["w11_split"].eq(split)].copy()
        initial = all_rows.loc[all_rows["delay_minutes"].eq(0)].copy().reset_index(drop=True)
        context_rows, context_execute = choose_context_policy(all_rows, config["thresholds"], config["policy"])
        context_rows["executed"] = context_execute.to_numpy()
        decisions_by_split.append(context_rows)
        static_rows, static_execute = static_context_policy(initial, set(config["policy"]["enter_states"]))
        policies: dict[str, tuple[pd.DataFrame, pd.Series]] = {
            "ENTER_NOW": (initial, pd.Series(True, index=initial.index)),
            "WAIT_1M": fixed_delay(all_rows, 1),
            "WAIT_2M": fixed_delay(all_rows, 2),
            "WAIT_3M": fixed_delay(all_rows, 3),
            "STATIC_CONTEXT": (static_rows, static_execute),
            "W14_CONTEXTUAL": (context_rows, context_execute),
        }
        policy_values = {name: policy_metrics(rows, execute, stress) for name, (rows, execute) in policies.items()}
        results[split] = {
            "atlas": state_atlas(initial),
            "policies": policy_values,
            "bootstrap": bootstrap_improvement(
                initial, context_rows, context_execute,
                int(config["validation_gates"]["bootstrap_samples"]), int(config["seed"]),
            ),
            "actions": context_rows["w14_action"].value_counts().sort_index().astype(int).to_dict(),
        }

    validation = results["W11_VALIDATION"]
    metrics = validation["policies"]["W14_CONTEXTUAL"]
    boot = validation["bootstrap"]
    gates_config = config["validation_gates"]
    gates = {
        "minimum_episodes": metrics["episodes"] >= int(gates_config["minimum_episodes"]),
        "minimum_executed": metrics["executed"] >= int(gates_config["minimum_executed"]),
        "execution_rate": float(gates_config["minimum_execution_rate"]) <= metrics["execution_rate"] <= float(gates_config["maximum_execution_rate"]),
        "positive_per_signal": metrics["net_bps_per_signal"] > float(gates_config["minimum_net_bps_per_signal"]),
        "positive_per_trade": metrics["net_bps_per_trade"] > float(gates_config["minimum_net_bps_per_trade"]),
        "material_improvement": metrics["improvement_bps_per_signal"] >= float(config["economics"]["minimum_material_improvement_bps_per_signal"]),
        "good_retention": metrics["good_retained"] >= float(gates_config["minimum_good_retention"]),
        "bad_avoidance": metrics["bad_avoided"] >= float(gates_config["minimum_bad_avoidance"]),
        "symbols": metrics["symbols"] >= int(gates_config["minimum_symbols"]),
        "stress_positive": metrics["stress_net_bps_per_signal"] > 0.0,
        "bootstrap_ci": boot["ci95_low"] > 0.0,
    }
    passed = all(gates.values())
    status = "AEGIS_W14_CONTEXTUAL_REGIME_EDGE_FOUND" if passed else "AEGIS_W14_NO_ROBUST_CONTEXTUAL_REGIME_EDGE"
    verdict = {
        "schema_version": "aegis-contextual-regime-arbitration-w14-verdict-v1",
        "status": status,
        "dataset": {
            "episodes": int(baseline_rows["live_signal_episode_id"].nunique()),
            "train": int(baseline_rows["w11_split"].eq("W11_TRAIN").sum()),
            "validation": int(baseline_rows["w11_split"].eq("W11_VALIDATION").sum()),
            "holdout": config["sources"]["sealed_holdout_status"],
        },
        "higher_timeframe_coverage": coverage,
        "train": results["W11_TRAIN"],
        "validation": results["W11_VALIDATION"],
        "gates": gates,
        "flags": {
            "W14_CONTEXT_STATES_INFORMATION_FOUND": False,
            "W14_CONTEXTUAL_POLICY_EDGE_FOUND": bool(passed),
            "W14_MODELING_JUSTIFIED": bool(passed),
            "W14_READY_FOR_PROSPECTIVE_OBSERVATION": bool(passed),
            "W14_READY_FOR_LIVE": False,
        },
        "restrictions_verified": config["restrictions"],
    }
    # Descriptive state information requires at least one repeated state's mean
    # to differ materially, but it never overrides the economic policy gate.
    informative = any(
        value["episodes"] >= 20 and abs(value["mean_net_bps"] - results["W11_VALIDATION"]["policies"]["ENTER_NOW"]["net_bps_per_signal"]) >= 5.0
        for value in validation["atlas"]
    )
    verdict["flags"]["W14_CONTEXT_STATES_INFORMATION_FOUND"] = bool(informative)
    transition = next((row for row in validation["atlas"] if row["state"] == "REGIME_TRANSITION"), None)
    shock = next((row for row in validation["atlas"] if row["state"] == "SHOCK_OR_UNCERTAIN"), None)
    verdict["flags"]["W14_ZERO_COST_POLICY_DIRECTIONAL_MAJORITY_FOUND"] = bool(
        metrics["gross_bps_per_trade"] > 0.0
        and metrics["gross_directional_win_rate"] > 0.50
        and metrics["mfe_greater_than_mae_rate"] > 0.50
    )
    verdict["flags"]["W14_REGIME_TRANSITION_DESCRIPTIVE_SIGNAL"] = bool(
        transition
        and transition["episodes"] >= 20
        and transition["mean_gross_bps"] > 0.0
        and transition["gross_directional_win_rate"] >= 0.55
    )
    verdict["flags"]["W14_SHOCK_DANGER_INFORMATION_FOUND"] = bool(
        shock
        and shock["episodes"] >= 20
        and shock["gross_directional_win_rate"] <= 0.30
        and shock["median_mae_bps"] > 2.0 * shock["median_mfe_bps"]
    )
    args.out_dir.mkdir(parents=True, exist_ok=True)
    decisions = pd.concat(decisions_by_split, ignore_index=True)
    decisions[[
        "live_signal_episode_id", "decision_timestamp", "symbol", "side", "w11_split", "entry_class",
        "initial_context_state", "w14_action", "executed", "delay_minutes", "net_return_bps", "mfe_bps", "mae_bps",
    ]].to_csv(args.out_dir / "w14_contextual_decisions.csv", index=False)
    lines = [
        "# Aegis W14 Contextual Regime Arbitration - Result", "", "## Verdict", "", f"`{status}`", "",
        f"- Episodes: {verdict['dataset']['episodes']} (TRAIN {verdict['dataset']['train']}; VALIDATION {verdict['dataset']['validation']}).",
        f"- Holdout: `{verdict['dataset']['holdout']}`.",
        f"- 4h EMA25 slope coverage: {coverage['dir240m__ema25_slope_atr']:.1%}.",
        f"- 1d return coverage: {coverage['dir1440m__return_1_bps']:.1%}; 1d EMA25 slope coverage: {coverage['dir1440m__ema25_slope_atr']:.1%}.",
        "", "## TRAIN Atlas", "", *atlas_table(results["W11_TRAIN"]["atlas"]),
        "", "## VALIDATION Atlas", "", *atlas_table(validation["atlas"]),
        "", "## VALIDATION Directional Safety (Zero Cost)", "", *directional_atlas_table(validation["atlas"]),
        "",
        f"The W14 policy itself executed {metrics['executed']} signals with gross mean "
        f"{metrics['gross_bps_per_trade']:.2f} bps/trade, directional win rate "
        f"{metrics['gross_directional_win_rate']:.1%}, and MFE > MAE in "
        f"{metrics['mfe_greater_than_mae_rate']:.1%}. These figures subtract no fees or commissions.",
        "", "## TRAIN Policies", "", *metric_table(results["W11_TRAIN"]["policies"]),
        "", "## VALIDATION Policies", "", *metric_table(validation["policies"]),
        "", "## VALIDATION Actions", "",
    ]
    for action, count in validation["actions"].items():
        lines.append(f"- `{action}`: {count}")
    lines.extend([
        "", "## Gates", "",
        *[f"- `{name}`: `{str(value).upper()}`" for name, value in gates.items()],
        "", "## Interpretation", "",
        "The taxonomy is useful only if its fixed decisions produce positive economics. Descriptive differences between states do not authorize adding narrative exceptions after losses.",
        "", "No production, TypeScript, PM2, account, order, guard, leverage, or sealed holdout was modified.", "",
    ])
    safe = _json_safe(verdict)
    (args.out_dir / "aegis_contextual_regime_arbitration_w14_verdict.json").write_text(
        json.dumps(safe, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8"
    )
    (args.out_dir / "aegis_contextual_regime_arbitration_w14_result.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"status": status, "coverage": coverage, "validation": metrics, "gates": gates}, indent=2))


if __name__ == "__main__":
    main()
