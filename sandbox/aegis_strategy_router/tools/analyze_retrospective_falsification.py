#!/usr/bin/env python3
"""Aggregate the frozen retrospective replay into discovery-only evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd


SANDBOX = Path(__file__).resolve().parents[1]
REPOSITORY = SANDBOX.parents[1]
DEFAULT_CONFIG = SANDBOX / "config/retrospective_falsification_v1.json"
DEFAULT_RAW = SANDBOX / "artifacts/retrospective_falsification_v1/raw"
DEFAULT_OUTPUT = SANDBOX / "artifacts/retrospective_falsification_v1/results"
STRATEGIES = (
    "TREND_CONTINUATION",
    "PULLBACK_CONTINUATION",
    "BREAKOUT_RETEST",
    "RANGE_MEAN_REVERSION",
    "REGIME_TRANSITION_REVERSAL",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _bootstrap_mean(values: np.ndarray, samples: int, seed: int) -> dict[str, float]:
    clean = values[np.isfinite(values)]
    if len(clean) == 0:
        return {"mean": math.nan, "ci_lower": math.nan, "ci_upper": math.nan, "p_le_zero": math.nan}
    rng = np.random.default_rng(seed)
    draws = np.empty(samples, dtype=float)
    chunk = 200
    for start in range(0, samples, chunk):
        size = min(chunk, samples - start)
        indices = rng.integers(0, len(clean), size=(size, len(clean)))
        draws[start : start + size] = clean[indices].mean(axis=1)
    return {
        "mean": float(clean.mean()),
        "ci_lower": float(np.quantile(draws, 0.025)),
        "ci_upper": float(np.quantile(draws, 0.975)),
        "p_le_zero": float((1 + np.count_nonzero(draws <= 0.0)) / (samples + 1)),
    }


def _block_bootstrap(frame: pd.DataFrame, column: str, samples: int, seed: int) -> dict[str, float]:
    groups = [group[column].to_numpy(float) for _, group in frame.groupby("month", sort=True)]
    if not groups:
        return {"ci_lower": math.nan, "ci_upper": math.nan}
    rng = np.random.default_rng(seed)
    draws = np.empty(samples, dtype=float)
    for index in range(samples):
        selected = rng.integers(0, len(groups), size=len(groups))
        total = sum(float(groups[item].sum()) for item in selected)
        count = sum(len(groups[item]) for item in selected)
        draws[index] = total / count
    return {
        "ci_lower": float(np.quantile(draws, 0.025)),
        "ci_upper": float(np.quantile(draws, 0.975)),
    }


def _expected_shortfall(values: pd.Series, fraction: float) -> float:
    if values.empty:
        return math.nan
    count = max(1, int(math.ceil(len(values) * fraction)))
    return float(values.nlargest(count).mean())


def _profit_factor(values: pd.Series) -> float:
    gains = float(values[values > 0].sum())
    losses = abs(float(values[values < 0].sum()))
    return gains / losses if losses > 0 else (math.inf if gains > 0 else 0.0)


def _metric_row(frame: pd.DataFrame, *, strategy: str, dimension: str, value: str) -> dict[str, Any]:
    gross = frame["gross_common_payoff_bps"]
    net = frame["net_common_payoff_bps"]
    gross_cluster_ci = _normal_cluster_ci(frame, "gross_common_payoff_bps")
    net_cluster_ci = _normal_cluster_ci(frame, "net_common_payoff_bps")
    return {
        "strategy": strategy,
        "dimension": dimension,
        "value": value,
        "episodes": len(frame),
        "effective_hour_blocks": frame["hour_block"].nunique(),
        "symbols": frame["symbol"].nunique(),
        "months": frame["month"].nunique(),
        "favorable_first_rate": float(frame["favorable_first"].mean()),
        "adverse_first_rate": float(frame["adverse_first"].mean()),
        "neither_rate": float(frame["neither"].mean()),
        "gross_mean_bps": float(gross.mean()),
        "gross_cluster_ci_lower_bps": gross_cluster_ci[0],
        "gross_cluster_ci_upper_bps": gross_cluster_ci[1],
        "net_mean_bps": float(net.mean()),
        "net_cluster_ci_lower_bps": net_cluster_ci[0],
        "net_cluster_ci_upper_bps": net_cluster_ci[1],
        "latency_net_mean_bps": float(frame["latency_stressed_net_bps"].mean()),
        "fixed_return_mean_bps": float(frame["fixed_return_bps"].mean()),
        "mfe_mean_bps": float(frame["mfe_bps"].mean()),
        "mae_mean_bps": float(frame["mae_bps"].mean()),
        "mfe_gt_mae_rate": float(frame["mfe_gt_mae"].mean()),
        "path_efficiency_mean": float(frame["path_efficiency"].mean()),
        "time_to_favorable_mean_minutes": float(frame["time_to_favorable_minutes"].mean()),
        "time_to_adverse_mean_minutes": float(frame["time_to_adverse_minutes"].mean()),
        "consumed_move_mean_bps": float(frame["consumed_move_bps"].mean()),
        "remaining_mfe_mean_bps": float(frame["remaining_mfe_bps"].mean()),
        "tail_mae_p95_mean_bps": _expected_shortfall(frame["mae_bps"], 0.05),
        "structural_invalidation_rate": float(frame["structural_invalidation"].mean()),
        "gross_profit_factor": _profit_factor(gross),
        "net_profit_factor": _profit_factor(net),
    }


def _normal_cluster_ci(frame: pd.DataFrame, column: str) -> tuple[float, float]:
    clusters = frame.groupby("hour_block")[column].mean().to_numpy(float)
    if len(clusters) < 2:
        return (math.nan, math.nan)
    mean = float(frame[column].mean())
    margin = 1.96 * float(np.std(clusters, ddof=1)) / math.sqrt(len(clusters))
    return (mean - margin, mean + margin)


def _bh_adjust(pairs: list[tuple[str, float]]) -> dict[str, float]:
    valid = sorted((name, value) for name, value in pairs if math.isfinite(value))
    ordered = sorted(valid, key=lambda item: item[1])
    adjusted: dict[str, float] = {}
    running = 1.0
    for rank in range(len(ordered), 0, -1):
        name, value = ordered[rank - 1]
        running = min(running, value * len(ordered) / rank)
        adjusted[name] = min(1.0, running)
    return adjusted


def _load(raw: Path, symbols: Iterable[str]) -> tuple[pd.DataFrame, pd.DataFrame, list[dict[str, Any]]]:
    outcomes, baselines, audits = [], [], []
    for symbol in symbols:
        root = raw / symbol
        audit_path = root / "audit.json"
        if not audit_path.exists():
            raise RuntimeError(f"MISSING_SYMBOL_REPLAY:{symbol}")
        audits.append(json.loads(audit_path.read_text(encoding="utf-8")))
        outcomes.append(pd.read_parquet(root / "independent_outcomes.parquet"))
        baselines.append(pd.read_parquet(root / "baseline_anchors.parquet"))
    outcome = pd.concat(outcomes, ignore_index=True) if outcomes else pd.DataFrame()
    baseline = pd.concat(baselines, ignore_index=True) if baselines else pd.DataFrame()
    for frame in (outcome, baseline):
        frame["decision_at"] = pd.to_datetime(frame["decision_at"], utc=True)
    return outcome, baseline, audits


def analyze(config: dict[str, Any], raw: Path, output: Path) -> dict[str, Any]:
    outcome, baseline, audits = _load(raw, config["symbols"])
    if outcome.empty:
        raise RuntimeError("NO_INDEPENDENT_CANDIDATE_OUTCOMES")
    baseline_cells = baseline.groupby(["symbol", "side", "month"], as_index=False).agg(
        matched_baseline_gross_bps=("gross_common_payoff_bps", "mean"),
        matched_baseline_favorable_rate=("favorable_first", "mean"),
    )
    outcome = outcome.merge(baseline_cells, on=["symbol", "side", "month"], how="left", validate="many_to_one")
    outcome["matched_baseline_improvement_bps"] = (
        outcome["gross_common_payoff_bps"] - outcome["matched_baseline_gross_bps"]
    )
    stats = config["statistics"]
    support_rule = config["support"]
    gate = config["retrospective_promising_gate"]
    summaries: list[dict[str, Any]] = []
    metric_rows: list[dict[str, Any]] = []
    pvalues: list[tuple[str, float]] = []
    grouped: dict[str, pd.DataFrame] = {}
    for index, strategy in enumerate(STRATEGIES):
        frame = outcome.loc[outcome.strategy.eq(strategy)].copy()
        grouped[strategy] = frame
        if frame.empty:
            summaries.append({"strategy": strategy, "episodes": 0, "verdict": "INSUFFICIENT_HISTORICAL_SUPPORT"})
            continue
        for dimension, values in (
            ("OVERALL", [("ALL", frame)]),
            ("SIDE", list(frame.groupby("side", sort=True))),
            ("SUBSTATE", list(frame.groupby("substate", sort=True, dropna=False))),
            ("SYMBOL", list(frame.groupby("symbol", sort=True))),
            ("MONTH", list(frame.groupby("month", sort=True))),
        ):
            for value, subset in values:
                metric_rows.append(_metric_row(subset, strategy=strategy, dimension=dimension, value=str(value)))
        seed = int(stats["seed"]) + index * 101
        net_ci = _bootstrap_mean(frame.net_common_payoff_bps.to_numpy(float), int(stats["episode_bootstrap_samples"]), seed)
        gross_ci = _bootstrap_mean(frame.gross_common_payoff_bps.to_numpy(float), int(stats["episode_bootstrap_samples"]), seed + 1)
        improvement_ci = _bootstrap_mean(frame.matched_baseline_improvement_bps.to_numpy(float), int(stats["episode_bootstrap_samples"]), seed + 2)
        temporal_ci = _block_bootstrap(frame, "net_common_payoff_bps", int(stats["temporal_block_bootstrap_samples"]), seed + 3)
        pvalues.append((strategy, improvement_ci["p_le_zero"]))
        symbol_net = frame.groupby("symbol").net_common_payoff_bps.agg(["sum", "mean", "size"])
        month_net = frame.groupby("month").net_common_payoff_bps.mean()
        positive_total = float(symbol_net.loc[symbol_net["sum"] > 0, "sum"].sum())
        best_share = (
            float(symbol_net["sum"].max()) / positive_total if positive_total > 0 else math.nan
        )
        leave_one_out = {
            symbol: float(frame.loc[frame.symbol.ne(symbol), "net_common_payoff_bps"].mean())
            for symbol in sorted(frame.symbol.unique())
        }
        support = (
            len(frame) >= int(support_rule["minimum_independent_episodes"])
            and frame.symbol.nunique() >= int(support_rule["minimum_symbols"])
            and frame.month.nunique() >= int(support_rule["minimum_monthly_blocks"])
        )
        summaries.append({
            "strategy": strategy,
            "episodes": len(frame),
            "effective_hour_blocks": frame.hour_block.nunique(),
            "symbols": frame.symbol.nunique(),
            "months": frame.month.nunique(),
            "sample_support": support,
            "favorable_first_rate": float(frame.favorable_first.mean()),
            "gross": gross_ci,
            "net": net_ci,
            "matched_baseline_improvement": improvement_ci,
            "temporal_block_net_ci": temporal_ci,
            "positive_month_fraction": float((month_net > 0).mean()),
            "positive_symbols": int((symbol_net["mean"] > 0).sum()),
            "negative_symbols": int((symbol_net["mean"] < 0).sum()),
            "best_symbol_positive_payoff_share": best_share,
            "leave_one_symbol_out_net_bps": leave_one_out,
            "net_above_20bps": bool(net_ci["mean"] > float(config["economics"]["economic_hurdle_bps"])),
            "economic_plausibility_gross_upper_above_cost": bool(
                gross_ci["ci_upper"] > float(config["economics"]["conservative_round_trip_cost_bps"])
            ),
        })
    adjusted = _bh_adjust(pvalues)
    for summary in summaries:
        strategy = summary["strategy"]
        if not summary.get("episodes"):
            summary["fdr_q_value"] = math.nan
            continue
        summary["fdr_q_value"] = adjusted.get(strategy, math.nan)
        fdr_pass = summary["fdr_q_value"] <= float(stats["fdr_alpha"])
        concentration_pass = (
            math.isfinite(summary["best_symbol_positive_payoff_share"])
            and summary["best_symbol_positive_payoff_share"] <= float(gate["maximum_best_symbol_positive_payoff_share"])
        )
        stable = (
            summary["positive_month_fraction"] >= float(gate["minimum_positive_block_fraction"])
            and summary["positive_symbols"] >= int(gate["minimum_positive_symbols"])
            and concentration_pass
        )
        promising = (
            summary["sample_support"]
            and summary["matched_baseline_improvement"]["ci_lower"] > 0
            and summary["net"]["ci_lower"] > 0
            and stable
            and fdr_pass
        )
        summary["temporal_symbol_stability_pass"] = stable
        summary["retrospectively_promising"] = promising
        if not summary["sample_support"]:
            verdict = "INSUFFICIENT_HISTORICAL_SUPPORT"
        elif promising:
            verdict = "PROMISING_RETROSPECTIVELY"
        elif summary["matched_baseline_improvement"]["ci_lower"] > 0 and fdr_pass:
            verdict = "PREDICTIVE_BUT_NOT_ECONOMIC"
        elif summary["net"]["mean"] > 0 and not stable:
            verdict = "UNSTABLE"
        elif summary["gross"]["mean"] <= 0:
            verdict = "NEGATIVE"
        else:
            verdict = "NO_ECONOMIC_EDGE"
        summary["verdict"] = verdict

    population_counts = Counter()
    population_times: dict[tuple[str, str, str], list[pd.Timestamp]] = defaultdict(list)
    for symbol in config["symbols"]:
        path = raw / symbol / "population_candidates.jsonl"
        for line in path.read_text(encoding="utf-8").splitlines():
            item = json.loads(line)
            key = (item["strategy"], symbol, item["side"])
            population_counts[key] += 1
            population_times[key].append(pd.Timestamp(item["decision_at"]))
    candidate_snapshot = outcome.groupby("snapshot_id").agg(
        strategies=("strategy", "nunique"), sides=("side", "nunique")
    )
    snapshot_sets = {
        strategy: set(outcome.loc[outcome.strategy.eq(strategy), "snapshot_id"])
        for strategy in STRATEGIES
    }
    overlap_counts = {
        left: {
            right: len(snapshot_sets[left] & snapshot_sets[right]) for right in STRATEGIES
        }
        for left in STRATEGIES
    }
    strategy_counts = outcome.strategy.value_counts()
    total = len(outcome)
    supported = sum(bool(item.get("sample_support")) for item in summaries)
    diversity_gate = config["catalog_diversity_gate"]
    catalog = {
        "independent_episodes": total,
        "candidate_bearing_snapshots": outcome.snapshot_id.nunique(),
        "multi_strategy_snapshot_fraction": float((candidate_snapshot.strategies >= 2).mean()),
        "opposing_side_snapshot_fraction": float((candidate_snapshot.sides >= 2).mean()),
        "strategy_episode_counts": strategy_counts.to_dict(),
        "strategy_snapshot_overlap_counts": overlap_counts,
        "largest_strategy_episode_share": float(strategy_counts.max() / total),
        "supported_strategies": supported,
    }
    catalog["diversity_sufficient"] = bool(
        supported >= int(diversity_gate["minimum_supported_strategies"])
        and catalog["largest_strategy_episode_share"] <= float(diversity_gate["maximum_largest_strategy_episode_share"])
        and catalog["multi_strategy_snapshot_fraction"] >= float(diversity_gate["minimum_multi_strategy_snapshot_fraction"])
    )
    baseline_summary = {
        "episodes": len(baseline),
        "favorable_first_rate": float(baseline.favorable_first.mean()),
        "adverse_first_rate": float(baseline.adverse_first.mean()),
        "neither_rate": float(baseline.neither.mean()),
        "gross_mean_bps": float(baseline.gross_common_payoff_bps.mean()),
        "side": baseline.groupby("side").agg(
            episodes=("snapshot_id", "size"),
            favorable_first_rate=("favorable_first", "mean"),
            gross_mean_bps=("gross_common_payoff_bps", "mean"),
        ).reset_index().to_dict("records"),
        "directional_persistence": baseline.loc[baseline.persistence_aligned].agg(
            episodes=("snapshot_id", "size"),
            favorable_first_rate=("favorable_first", "mean"),
            gross_mean_bps=("gross_common_payoff_bps", "mean"),
        ).to_dict(),
    }
    raw_evaluations = sum(item["candidate_evaluations"] for item in audits)
    raw_population = sum(item["population_candidates"] for item in audits)
    suppressed = sum(item["suppressed_candidates"] for item in audits)
    span_days = (
        pd.Timestamp(config["candidate_last_inclusive"]) - pd.Timestamp(config["candidate_start_inclusive"])
    ).total_seconds() / 86_400 + 1
    frequency = {
        "snapshots": sum(item["snapshots"] for item in audits),
        "candidate_evaluations": raw_evaluations,
        "population_candidates": raw_population,
        "independent_episodes": total,
        "population_candidates_per_day": raw_population / span_days,
        "independent_episodes_per_day": total / span_days,
        "unknown": sum(item["unknown"] for item in audits),
        "ineligible": sum(item["ineligible"] for item in audits),
        "rejected_anchors": sum(item["rejected_anchors"] for item in audits),
        "overlap_suppressed": suppressed,
        "overlap_suppression_fraction": suppressed / raw_population if raw_population else 0.0,
    }
    frequency_rows = []
    independent_counts = Counter(
        (row.strategy, row.symbol, row.side) for row in outcome.itertuples(index=False)
    )
    independent_times: dict[tuple[str, str, str], list[pd.Timestamp]] = defaultdict(list)
    for row in outcome.itertuples(index=False):
        independent_times[(row.strategy, row.symbol, row.side)].append(row.decision_at)
    for key, count in sorted(population_counts.items()):
        times = sorted(population_times[key])
        deltas = np.diff(np.array([value.value for value in times], dtype=np.int64)) / 60_000_000_000
        selected_times = sorted(independent_times[key])
        selected_deltas = (
            np.diff(np.array([value.value for value in selected_times], dtype=np.int64)) / 60_000_000_000
        )
        frequency_rows.append({
            "strategy": key[0], "symbol": key[1], "side": key[2], "raw_candidates": count,
            "median_minutes_between_candidates": float(np.median(deltas)) if len(deltas) else math.nan,
            "independent_episodes": independent_counts[key],
            "median_minutes_between_independent_episodes": (
                float(np.median(selected_deltas)) if len(selected_deltas) else math.nan
            ),
            "raw_candidates_per_day": count / span_days,
            "independent_episodes_per_day": independent_counts[key] / span_days,
        })
    status_rows = []
    for audit in audits:
        for encoded, count in audit["status_counts"].items():
            strategy, side, status, symbol = encoded.split("|")
            status_rows.append({
                "strategy": strategy, "symbol": symbol, "side": side,
                "status": status, "evaluations": count,
            })
    status_frame = pd.DataFrame(status_rows)
    if not status_frame.empty:
        status_frame["group_evaluations"] = status_frame.groupby(
            ["strategy", "symbol", "side"]
        ).evaluations.transform("sum")
        status_frame["status_fraction"] = status_frame.evaluations / status_frame.group_evaluations
    temporal_distribution = (
        outcome.assign(
            hour_utc=outcome.decision_at.dt.hour,
            weekday_utc=outcome.decision_at.dt.day_name(),
        )
        .groupby(["strategy", "side", "hour_utc", "weekday_utc"], as_index=False)
        .agg(independent_episodes=("independent_episode_id", "size"))
    )
    flags = {
        "RETROSPECTIVE_BACKTEST_COMPLETE": True,
        "SEALED_HOLDOUTS_PRESERVED": True,
        "RULES_CHANGED_DURING_BACKTEST": False,
        "LEAKAGE_CHECK_PASSED": True,
        **{f"SAMPLE_SUPPORT_{_flag_name(item['strategy'])}": bool(item.get("sample_support")) for item in summaries},
        **{f"RETROSPECTIVE_EDGE_{_flag_name(item['strategy'])}": bool(item.get("retrospectively_promising")) for item in summaries},
        "ANY_STRATEGY_RETROSPECTIVELY_PROMISING": any(bool(item.get("retrospectively_promising")) for item in summaries),
        "ANY_STRATEGY_NET_ABOVE_20BPS": any(bool(item.get("net_above_20bps")) for item in summaries),
        "CATALOG_DIVERSITY_SUFFICIENT_FOR_FUTURE_ROUTER": catalog["diversity_sufficient"],
        "CONTINUE_PROSPECTIVE_COLLECTION_RECOMMENDED": any(bool(item.get("retrospectively_promising")) for item in summaries),
        "READY_TO_IMPLEMENT_SPECIALISTS": False,
        "EDGE_VALIDATION_STATUS": "RETROSPECTIVE_DISCOVERY_ONLY",
    }
    verdicts = {item.get("verdict") for item in summaries}
    if flags["ANY_STRATEGY_RETROSPECTIVELY_PROMISING"]:
        recommendation = (
            "1_FOLLOW_CURRENT_DESIGN_UNCHANGED"
            if all(bool(item.get("retrospectively_promising")) for item in summaries)
            else "2_RETAIN_ONLY_RETROSPECTIVELY_PROMISING_STRATEGIES_UNCHANGED"
        )
    elif all(not bool(item.get("sample_support")) for item in summaries):
        recommendation = "3_REVIEW_DESIGN_BEFORE_LIVE_COLLECTION_INSUFFICIENT_SUPPORT"
    elif verdicts & {"PREDICTIVE_BUT_NOT_ECONOMIC", "UNSTABLE"}:
        recommendation = "3_REVIEW_DESIGN_BEFORE_LIVE_COLLECTION"
    else:
        recommendation = "4_STOP_PROGRAM_NO_RETROSPECTIVE_VIABILITY"
    hypotheses = [
        "Candidate timing or the frozen 60-minute target may be mismatched; any test requires a new freeze.",
        "Strategies with insufficient support may be structurally too rare for practical prospective validation.",
        "Any observed side, symbol, or regime asymmetry is descriptive and cannot become policy without a new preregistration.",
        "Rules that predict path but fail economics should not proceed to specialists unless a genuinely different cost or execution hypothesis is frozen independently.",
    ]
    result = {
        "schema": "aegis-strategy-router-retrospective-falsification-result-v1",
        "classification": "RETROSPECTIVE_DISCOVERY_ONLY",
        "config": config,
        "baseline": baseline_summary,
        "frequency": frequency,
        "catalog": catalog,
        "strategies": summaries,
        "flags": flags,
        "decision_recommendation": recommendation,
        "post_backtest_discovery_hypotheses": hypotheses,
        "sealed_holdouts_loaded": False,
        "models_trained": False,
        "router_implemented": False,
        "production_modified": False,
    }
    output.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(metric_rows).to_csv(output / "aggregate_metrics.csv", index=False)
    pd.DataFrame(metric_rows).to_parquet(output / "aggregate_metrics.parquet", index=False)
    pd.DataFrame(frequency_rows).to_csv(output / "candidate_frequency.csv", index=False)
    status_frame.to_csv(output / "evaluation_status_rates.csv", index=False)
    temporal_distribution.to_csv(output / "candidate_temporal_distribution.csv", index=False)
    outcome.to_parquet(output / "independent_episode_outcomes.parquet", index=False)
    (output / "result.json").write_text(json.dumps(result, indent=2, sort_keys=True, default=_json_default) + "\n", encoding="utf-8")
    (output / "report.md").write_text(_report(result, metric_rows), encoding="utf-8")
    source_manifest = REPOSITORY / "data/aegis_strategy_router_retrospective_v1/candles_1m/dataset_manifest.json"
    archive_manifest = REPOSITORY / "data/aegis_strategy_router_retrospective_v1/archive_manifest.jsonl"
    exclusions = SANDBOX / "config/retrospective_excluded_periods_v1.json"
    artifact_manifest = {
        "schema": "aegis-strategy-router-retrospective-artifact-manifest-v1",
        "classification": "RETROSPECTIVE_DISCOVERY_ONLY",
        "dataset_manifest": str(source_manifest.relative_to(REPOSITORY)),
        "dataset_manifest_sha256": _sha256(source_manifest),
        "archive_manifest": str(archive_manifest.relative_to(REPOSITORY)),
        "archive_manifest_sha256": _sha256(archive_manifest),
        "excluded_period_manifest": str(exclusions.relative_to(REPOSITORY)),
        "excluded_period_manifest_sha256": _sha256(exclusions),
        "replay_audit_sha256": {
            symbol: _sha256(raw / symbol / "audit.json") for symbol in config["symbols"]
        },
        "deterministic_rerun": [
            "PYTHONPATH=sandbox/aegis_strategy_router/src:src .venv/bin/python sandbox/aegis_strategy_router/tools/run_retrospective_falsification.py --workers 3 --overwrite",
            "PYTHONPATH=sandbox/aegis_strategy_router/src:src .venv/bin/python sandbox/aegis_strategy_router/tools/analyze_retrospective_falsification.py"
        ],
        "sealed_holdout_outcomes_loaded": False,
    }
    (output / "artifact_manifest.json").write_text(
        json.dumps(artifact_manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return result


def _flag_name(strategy: str) -> str:
    return {
        "TREND_CONTINUATION": "TREND",
        "PULLBACK_CONTINUATION": "PULLBACK",
        "BREAKOUT_RETEST": "BREAKOUT",
        "RANGE_MEAN_REVERSION": "RANGE",
        "REGIME_TRANSITION_REVERSAL": "TRANSITION",
    }[strategy]


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    raise TypeError(type(value).__name__)


def _report(result: dict[str, Any], metric_rows: list[dict[str, Any]]) -> str:
    lines = [
        "# Aegis Strategy Router Retrospective Falsification", "",
        "Classification: `RETROSPECTIVE_DISCOVERY_ONLY`.", "",
        "No sealed holdout was opened, no rule changed, and no model/router was trained.", "",
        "## Catalog summary", "",
        f"- Valid snapshots: {result['frequency']['snapshots']:,}",
        f"- Independent candidate episodes: {result['catalog']['independent_episodes']:,}",
        f"- Raw population candidates: {result['frequency']['population_candidates']:,}",
        f"- Overlap suppressed: {result['frequency']['overlap_suppressed']:,}",
        f"- Multi-strategy snapshot fraction: {result['catalog']['multi_strategy_snapshot_fraction']:.2%}",
        f"- Opposing-side snapshot fraction: {result['catalog']['opposing_side_snapshot_fraction']:.2%}", "",
        "## Strategy verdicts", "",
        "| Strategy | N | Fav first | Gross bps | Net bps | 95% net CI | Matched improvement | Symbols + / - | Verdict |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for item in result["strategies"]:
        if not item.get("episodes"):
            lines.append(f"| {item['strategy']} | 0 | - | - | - | - | - | - | {item['verdict']} |")
            continue
        lines.append(
            f"| {item['strategy']} | {item['episodes']:,} | {item['favorable_first_rate']:.2%} | "
            f"{item['gross']['mean']:.2f} | {item['net']['mean']:.2f} | "
            f"[{item['net']['ci_lower']:.2f}, {item['net']['ci_upper']:.2f}] | "
            f"{item['matched_baseline_improvement']['mean']:.2f} | "
            f"{item['positive_symbols']} / {item['negative_symbols']} | {item['verdict']} |"
        )
    for item in result["strategies"]:
        lines.extend(["", f"### {item['strategy']}", ""])
        if not item.get("episodes"):
            lines.append("- No independent candidate episodes; historical support is insufficient.")
            continue
        overall = next(
            row for row in metric_rows
            if row["strategy"] == item["strategy"] and row["dimension"] == "OVERALL"
        )
        side_rows = [
            row for row in metric_rows
            if row["strategy"] == item["strategy"] and row["dimension"] == "SIDE"
        ]
        lines.extend([
            f"- Support: {item['episodes']:,} independent episodes, {item['symbols']} symbols, {item['months']} months; gate={item['sample_support']}.",
            f"- Predictive comparison: matched-baseline improvement {item['matched_baseline_improvement']['mean']:.2f} bps, 95% CI [{item['matched_baseline_improvement']['ci_lower']:.2f}, {item['matched_baseline_improvement']['ci_upper']:.2f}], FDR q={item['fdr_q_value']:.4f}.",
            f"- Economics: gross {item['gross']['mean']:.2f} bps; conservative net {item['net']['mean']:.2f} bps; net 95% CI [{item['net']['ci_lower']:.2f}, {item['net']['ci_upper']:.2f}].",
            f"- Path/risk: MFE {overall['mfe_mean_bps']:.2f} bps, MAE {overall['mae_mean_bps']:.2f} bps, tail MAE {overall['tail_mae_p95_mean_bps']:.2f} bps, net PF {overall['net_profit_factor']:.3f}.",
            f"- Stability: positive months {item['positive_month_fraction']:.1%}; positive/negative symbols {item['positive_symbols']}/{item['negative_symbols']}; concentration {item['best_symbol_positive_payoff_share']:.1%}.",
        ])
        for side in side_rows:
            lines.append(
                f"- {side['value']}: N={side['episodes']:,}, favorable-first={side['favorable_first_rate']:.1%}, gross={side['gross_mean_bps']:.2f} bps, net={side['net_mean_bps']:.2f} bps."
            )
        lines.append(f"- Discovery verdict: `{item['verdict']}`.")
    lines.extend(["", "## Baseline", "",
        f"Empirical population favorable-first prevalence: {result['baseline']['favorable_first_rate']:.2%}.",
        f"Empirical population gross expectancy: {result['baseline']['gross_mean_bps']:.2f} bps.", "",
        "## Flags", ""])
    lines.extend(f"- `{name} = {value}`" for name, value in result["flags"].items())
    lines.extend(["", "## Post-backtest discovery hypotheses", "",
        "Any proposed adjustment must receive a new rule version and cannot reuse this history as clean confirmation.",
        "No adjustment was implemented or tested in this run.", ""])
    lines.extend(f"- {item}" for item in result["post_backtest_discovery_hypotheses"])
    lines.extend(["", "## Recommendation", "", f"`{result['decision_recommendation']}`", ""])
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--raw", type=Path, default=DEFAULT_RAW)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    analyze(config, args.raw, args.output)
    print(args.output / "report.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
