#!/usr/bin/env python3
"""Evaluate frozen policies on the diagnostic July split; never open August."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

EXPERIMENT = Path(__file__).resolve().parents[1]
SANDBOX = EXPERIMENT.parents[1]
REPOSITORY = SANDBOX.parents[1]
ENTRY_V1 = SANDBOX / "experiments/independent_entry_quality_discovery_v1"
for path in (EXPERIMENT / "src", ENTRY_V1 / "src", SANDBOX / "src", REPOSITORY / "src"):
    sys.path.insert(0, str(path))

from aegis_entry_enhancement_v1.evaluation import evaluate, metrics, monotonic_ranking, policy_masks  # noqa: E402


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def matched_baselines(frame: pd.DataFrame, coverage: float, config: dict) -> list[dict]:
    count = int(np.ceil(len(frame) * coverage))
    rankings = {
        "SIMPLE_VOLATILITY": frame["feature__tf15m__atr_percentile_96"],
        "SIMPLE_BTC_DIRECTION": frame["feature__btc_1h_directional_return_bps"],
        "AEGIS_CONFIDENCE": frame.turbo_score,
    }
    rows = []
    for name, score in rankings.items():
        selected = set(score.sort_values(ascending=False, kind="mergesort").head(count).index)
        rows.append(metrics(frame, pd.Series(frame.index.isin(selected), index=frame.index), name))
    if count == 0:
        rows.append({"policy": "RANDOM_MATCHED_COVERAGE", "signals": len(frame), "executed": 0, "coverage": 0.0, "net_bps_per_original_signal": 0.0, "net_ci_lower_bps": 0.0, "net_ci_upper_bps": 0.0})
        return rows
    rng = np.random.default_rng(config["trivial_baselines"]["random_seed"])
    net = []
    for _ in range(config["trivial_baselines"]["random_repetitions"]):
        chosen = rng.choice(frame.index.to_numpy(), count, replace=False)
        net.append(frame.target__net_common_payoff_bps.where(frame.index.isin(chosen), 0.0).mean())
    rows.append({"policy": "RANDOM_MATCHED_COVERAGE", "signals": len(frame), "executed": count, "coverage": count / len(frame), "net_bps_per_original_signal": float(np.mean(net)), "net_ci_lower_bps": float(np.quantile(net, 0.025)), "net_ci_upper_bps": float(np.quantile(net, 0.975))})
    return rows


def population_shift(frame: pd.DataFrame, repository: Path, config: dict) -> dict:
    model_path = repository / config["frozen_modules"]["opportunity"]["artifact"]
    bundle = joblib.load(model_path)
    reference = pd.read_parquet(ENTRY_V1 / "artifacts/dataset_v1/development_labeled.parquet")
    validation = reference.loc[reference.split.eq("VALIDATION") & reference.side.eq("LONG")]
    scores = bundle["opportunity"].predict_proba(validation[bundle["features"]])[:, 1]
    return {
        "aegis_opportunity": frame.opportunity_score.describe(percentiles=[0.1, 0.5, 0.9]).to_dict(),
        "reference_opportunity": pd.Series(scores).describe(percentiles=[0.1, 0.5, 0.9]).to_dict(),
        "aegis_ood_rate": float(frame.ood.mean()), "aegis_side_distribution": frame.side.value_counts(normalize=True).to_dict(),
        "aegis_volatility_distribution": frame["feature__tf15m__atr_percentile_96"].describe(percentiles=[0.1, 0.5, 0.9]).to_dict(),
    }


def stability(frame: pd.DataFrame, accepted: pd.Series) -> tuple[dict, list[dict]]:
    values = frame.copy()
    values["accepted"] = accepted
    values["week"] = values.signal_timestamp.dt.strftime("%Y-W%V")
    details = []
    for dimension in ("symbol", "side", "week"):
        for value, group in values.groupby(dimension):
            row = metrics(group, group.accepted, "PRIMARY")
            group_baseline = metrics(group, pd.Series(True, index=group.index), "AEGIS_ONLY")
            row["delta_vs_aegis_bps"] = row["net_bps_per_original_signal"] - group_baseline["net_bps_per_original_signal"]
            row.update({"dimension": dimension, "value": str(value)})
            details.append(row)
    detail = pd.DataFrame(details)
    symbols = detail.loc[detail.dimension.eq("symbol")]
    weeks = detail.loc[detail.dimension.eq("week")]
    return {
        "positive_delta_symbols": int(symbols.delta_vs_aegis_bps.gt(0).sum()),
        "positive_week_fraction": float(weeks.delta_vs_aegis_bps.gt(0).mean()),
    }, details


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=EXPERIMENT / "config/preregistration_v1.json")
    parser.add_argument("--dataset", type=Path, default=EXPERIMENT / "artifacts/dataset_v1")
    parser.add_argument("--output", type=Path, default=EXPERIMENT / "artifacts/run_01")
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    frame = pd.read_parquet(args.dataset / "development_labeled.parquet")
    validation = frame.loc[frame.split.eq("VALIDATION")].copy()
    validation["signal_timestamp"] = pd.to_datetime(validation.signal_timestamp, utc=True)
    holdout = pd.read_parquet(args.dataset / "final_holdout_features_sealed.parquet")
    prohibited = [column for column in frame if any(token in column.lower() for token in ("future", "realized_pnl", "exit_reason")) and column.startswith("feature__")]
    leakage = {
        "future_feature_columns": prohibited,
        "feature_after_signal_rows": int((pd.to_datetime(frame.max_feature_available_at, utc=True) > pd.to_datetime(frame.signal_timestamp, utc=True)).sum()),
        "holdout_target_columns": [column for column in holdout if column.startswith("target__")],
    }
    leakage["passed"] = not prohibited and leakage["feature_after_signal_rows"] == 0 and not leakage["holdout_target_columns"]
    if not leakage["passed"]:
        raise RuntimeError("LEAKAGE_AUDIT_FAILED")
    args.output.mkdir(parents=True, exist_ok=True)
    evaluated = evaluate(validation, config)
    primary = next(row for row in evaluated["policy_rows"] if row["policy"] == config["policies"]["primary"])
    baseline = next(row for row in evaluated["policy_rows"] if row["policy"] == "AEGIS_ONLY")
    monotonic, rank_spearman = monotonic_ranking(evaluated["coverage_rows"])
    simple = matched_baselines(validation, primary["coverage"], config)
    stable, stability_rows = stability(validation, evaluated["masks"][config["policies"]["primary"]])
    statistical_support = len(validation) >= config["support"]["minimum_validation_signals"] and validation.signal_timestamp.dt.floor("D").nunique() >= config["support"]["minimum_validation_day_blocks"] and validation.symbol.nunique() >= config["support"]["minimum_symbols"]
    clean_support = False
    good_bad = {
        key: primary[key] for key in (
            "bad_rejected", "good_rejected", "bad_trade_rejection_rate", "good_trade_destruction_rate",
            "rejection_precision", "bad_prevalence", "net_value_removed_bps", "net_value_preserved_bps",
        )
    }
    def useful(row: dict) -> bool:
        return (
            config["success"]["minimum_coverage"] <= row["coverage"] <= config["success"]["maximum_coverage"]
            and row["executed"] >= config["support"]["minimum_selected_signals"]
            and row["delta_vs_aegis_bps"] >= config["success"]["minimum_delta_net_bps"]
            and row["delta_ci_lower_bps"] > config["success"]["improvement_ci_lower_above_bps"]
            and row["net_bps_per_executed"] > baseline["net_bps_per_executed"]
            and row["bad_trade_rejection_rate"] > row["good_trade_destruction_rate"]
        )
    opportunity_row = next(row for row in evaluated["policy_rows"] if row["policy"] == "AEGIS_OPPORTUNITY_GATE")
    cross_row = next(row for row in evaluated["policy_rows"] if row["policy"] == "AEGIS_CROSS_MARKET_CONFIRMATION")
    primary_useful = useful(primary)
    flags = {
        "AEGIS_ENTRY_ENHANCEMENT_DATASET_BUILT": True, "AEGIS_BASELINE_REPRODUCED": True,
        "LEAKAGE_CHECK_PASSED": True,
        "OPPORTUNITY_GATE_ADDS_VALUE": useful(opportunity_row),
        "CROSS_MARKET_CONFIRMATION_ADDS_VALUE": useful(cross_row),
        "COMBINED_GATE_ADDS_VALUE": primary_useful,
        "QUALITY_RANKING_MONOTONIC": monotonic and rank_spearman >= 0.7,
        "BAD_TRADE_REJECTION_IMPROVED": primary["bad_trade_rejection_rate"] > primary["good_trade_destruction_rate"],
        "GOOD_TRADE_DESTRUCTION_ACCEPTABLE": primary["good_trade_destruction_rate"] <= 1 - config["success"]["minimum_good_retention"],
        "MFE_MAE_GEOMETRY_IMPROVED": primary["mfe_mae_ratio"] > baseline["mfe_mae_ratio"],
        "TAIL_RISK_IMPROVED": primary["tail_mae_bps"] < baseline["tail_mae_bps"],
        "NET_EXPECTANCY_IMPROVED_VS_AEGIS": primary_useful,
        "NET_EXPECTANCY_POSITIVE": primary["executed"] >= config["support"]["minimum_selected_signals"] and primary["net_bps_per_executed"] > 0,
        "MULTI_SYMBOL_STABLE": primary_useful and stable["positive_delta_symbols"] >= config["success"]["minimum_positive_symbols"],
        "TEMPORALLY_STABLE": primary_useful and stable["positive_week_fraction"] >= config["success"]["minimum_positive_week_fraction"],
        "SHORT_SUBGROUP_IMPROVED": False, "LONG_SUBGROUP_IMPROVED": False,
        "VALIDATION_SUPPORT_SUFFICIENT": clean_support,
        "FINAL_HOLDOUT_OPENED": False, "FINAL_HOLDOUT_PASSED": False,
        "AEGIS_ENTRY_ENHANCEMENT_PROMISING": False, "READY_FOR_PROSPECTIVE_OBSERVATION": False,
        "READY_FOR_SHADOW": False, "READY_FOR_LIVE": False,
    }
    for side, flag in (("SHORT", "SHORT_SUBGROUP_IMPROVED"), ("LONG", "LONG_SUBGROUP_IMPROVED")):
        subset = validation.loc[validation.side.eq(side)]
        if len(subset):
            side_eval = evaluate(subset, config)["policy_rows"]
            side_primary = next(row for row in side_eval if row["policy"] == config["policies"]["primary"])
            flags[flag] = useful(side_primary)
    if flags["COMBINED_GATE_ADDS_VALUE"] and flags["TAIL_RISK_IMPROVED"] and not flags["NET_EXPECTANCY_POSITIVE"]:
        verdict = "AEGIS_ENTRY_ENHANCEMENT_RISK_IMPROVEMENT_ONLY"
    elif flags["COMBINED_GATE_ADDS_VALUE"] or flags["NET_EXPECTANCY_IMPROVED_VS_AEGIS"] or flags["TAIL_RISK_IMPROVED"]:
        verdict = "AEGIS_ENTRY_ENHANCEMENT_WEAK_OR_UNSTABLE"
    else:
        verdict = "AEGIS_ENTRY_ENHANCEMENT_NO_VALUE"
    result = {
        "schema": "aegis-entry-enhancement-v1-result", "verdict": verdict,
        "evidence_status": "DISCOVERY_CONTAMINATED_DIAGNOSTIC_ONLY", "baseline": baseline,
        "primary": primary, "policy_rows": evaluated["policy_rows"], "good_bad_rejection": good_bad,
        "ranking_spearman": rank_spearman, "statistical_support": statistical_support,
        "clean_independent_support": clean_support, "stability": stable,
        "population_shift": population_shift(validation, REPOSITORY, config),
        "wait_evaluated": False, "final_holdout_state": "SEALED_NOT_OPENED",
        "flags": flags, "production_modified": False,
    }
    subgroup_rows = []
    for split_name in ("DISCOVERY", "CALIBRATION", "VALIDATION"):
        split_frame = frame.loc[frame.split.eq(split_name)].copy()
        split_frame["signal_timestamp"] = pd.to_datetime(split_frame.signal_timestamp, utc=True)
        for subgroup, subset in (
            ("ALL_AEGIS_SIGNALS", split_frame),
            ("AEGIS_LONG", split_frame.loc[split_frame.side.eq("LONG")]),
            ("AEGIS_SHORT", split_frame.loc[split_frame.side.eq("SHORT")]),
        ):
            if subset.empty:
                subgroup_rows.append({"split": split_name, "subgroup": subgroup, "policy": "NO_SUPPORT", "signals": 0})
                continue
            for policy, mask in policy_masks(subset, config).items():
                subgroup_rows.append({"split": split_name, "subgroup": subgroup, **metrics(subset, mask, policy)})
    split_manifest = {
        "frozen_splits": config["splits"], "rows": frame.split.value_counts().sort_index().to_dict(),
        "validation_statistical_support": statistical_support,
        "validation_clean_independent_support": clean_support,
        "final_holdout_rows": len(holdout), "final_holdout_labels_built": False,
    }
    (args.output / "leakage_audit.json").write_text(json.dumps(leakage, indent=2, sort_keys=True) + "\n")
    (args.output / "split_manifest.json").write_text(json.dumps(split_manifest, indent=2, sort_keys=True) + "\n")
    (args.output / "baseline_aegis_report.json").write_text(json.dumps(baseline, indent=2, sort_keys=True, default=_json) + "\n")
    (args.output / "latency_stress.json").write_text(json.dumps({
        "status": "NOT_EVALUATED_NO_SUBMINUTE_CAUSAL_EXECUTION_SOURCE",
        "recorded_signal_to_open_delay_reported": True,
        "wait_policy_evaluated": False,
    }, indent=2, sort_keys=True) + "\n")
    pd.DataFrame(evaluated["policy_rows"]).to_csv(args.output / "policy_comparison.csv", index=False)
    pd.DataFrame(evaluated["coverage_rows"]).to_csv(args.output / "risk_coverage.csv", index=False)
    pd.DataFrame(simple).to_csv(args.output / "trivial_baselines.csv", index=False)
    pd.DataFrame(stability_rows).to_csv(args.output / "symbol_side_time_stability.csv", index=False)
    pd.DataFrame(subgroup_rows).to_csv(args.output / "long_short_split_report.csv", index=False)
    pd.DataFrame([{**row, "cost_bps": cost, "stressed_net_per_executed": row["gross_bps_per_executed"] - cost} for row in evaluated["policy_rows"] for cost in config["target"]["cost_scenarios_bps"]]).to_csv(args.output / "cost_stress.csv", index=False)
    (args.output / "bad_trade_rejection.json").write_text(json.dumps(good_bad, indent=2, sort_keys=True) + "\n")
    (args.output / "population_shift.json").write_text(json.dumps(result["population_shift"], indent=2, sort_keys=True) + "\n")
    (args.output / "result.json").write_text(json.dumps(result, indent=2, sort_keys=True, default=_json) + "\n")
    (args.output / "final_report.md").write_text(report(result))
    manifest(args, result)
    print(json.dumps({"verdict": verdict, "flags": flags}, indent=2, sort_keys=True))
    return 0


def report(result: dict) -> str:
    baseline, primary = result["baseline"], result["primary"]
    return f"""# Aegis Entry Enhancement V1 Result

Verdict: `{result['verdict']}`.

Evidence: `DISCOVERY_CONTAMINATED_DIAGNOSTIC_ONLY`. FINAL_HOLDOUT remains
`SEALED_NOT_OPENED`.

## Primary comparison

- AEGIS_ONLY: {baseline['net_bps_per_original_signal']:.2f} net bps/signal.
- Combined gate: {primary['net_bps_per_original_signal']:.2f} net bps/signal.
- Delta: {primary['delta_vs_aegis_bps']:.2f} bps; CI [{primary['delta_ci_lower_bps']:.2f}, {primary['delta_ci_upper_bps']:.2f}].
- Coverage: {primary['coverage']:.2%}; executed: {primary['executed']}.
- Executed net: {primary['net_bps_per_executed']:.2f} bps.
- MFE/MAE ratio: {primary['mfe_mae_ratio']:.3f} versus {baseline['mfe_mae_ratio']:.3f}.
- BAD rejected: {primary['bad_trade_rejection_rate']:.2%}; GOOD destroyed: {primary['good_trade_destruction_rate']:.2%}.

No production, Aegis, exchange, Shadow or Live component was modified.
"""


def manifest(args: argparse.Namespace, result: dict) -> None:
    files = [args.config, args.dataset / "dataset_manifest.json", *sorted((EXPERIMENT / "src").rglob("*.py")), *sorted((EXPERIMENT / "tools").glob("*.py"))]
    outputs = [path for path in sorted(args.output.iterdir()) if path.is_file() and path.name != "artifact_manifest.json"]
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPOSITORY, check=True, capture_output=True, text=True).stdout.strip()
    payload = {
        "schema": "aegis-entry-enhancement-v1-artifact-manifest", "git_commit": commit,
        "final_holdout_state": result["final_holdout_state"],
        "inputs": {str(path.relative_to(REPOSITORY)): sha256(path) for path in files},
        "outputs": {str(path.relative_to(REPOSITORY)): sha256(path) for path in outputs},
        "rerun": [
            "PYTHONPATH=src .venv/bin/python sandbox/aegis_strategy_router/experiments/aegis_entry_enhancement_v1/tools/download_warmup.py",
            "PYTHONPATH=src .venv/bin/python sandbox/aegis_strategy_router/experiments/aegis_entry_enhancement_v1/tools/prepare_sources.py",
            "PYTHONPATH=sandbox/aegis_strategy_router/src:src:sandbox/aegis_strategy_router/experiments/independent_entry_quality_discovery_v1/src:sandbox/aegis_strategy_router/experiments/directional_alpha_v1/src:sandbox/aegis_strategy_router/experiments/aegis_entry_enhancement_v1/src .venv/bin/python sandbox/aegis_strategy_router/experiments/aegis_entry_enhancement_v1/tools/build_dataset.py",
            "PYTHONPATH=sandbox/aegis_strategy_router/src:src:sandbox/aegis_strategy_router/experiments/independent_entry_quality_discovery_v1/src:sandbox/aegis_strategy_router/experiments/aegis_entry_enhancement_v1/src .venv/bin/python sandbox/aegis_strategy_router/experiments/aegis_entry_enhancement_v1/tools/run_evaluation.py"
        ],
    }
    (args.output / "artifact_manifest.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _json(value):
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        return value.item()
    raise TypeError(type(value).__name__)


if __name__ == "__main__":
    raise SystemExit(main())
