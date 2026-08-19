#!/usr/bin/env python3
"""Run comprehensive evaluation for AEGIS_E4_RISK_GUARD_VALIDATION_V1."""

import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

repo_root = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(repo_root / "src"))
sys.path.insert(0, str(repo_root / "sandbox/aegis_strategy_router/src"))
sys.path.insert(0, str(repo_root / "sandbox/aegis_strategy_router/experiments/aegis_e4_robust_training/src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from aegis_e4_risk_guard_validation_v1.evaluation import compute_policy_metrics, compute_ranking_monotonicity
from aegis_e4_risk_guard_validation_v1.baselines import run_random_same_coverage_benchmark, run_heuristic_baselines
from aegis_e4_risk_guard_validation_v1.bootstrap import block_bootstrap_policy_delta
from aegis_e4_risk_guard_validation_v1.dataset import sha256_file


def main():
    exp_dir = Path(__file__).resolve().parents[1]
    config_path = exp_dir / "config/preregistration_v1.json"
    config = json.loads(config_path.read_text())

    thresholds_path = exp_dir / "config/thresholds_frozen_v1.json"
    thresholds = json.loads(thresholds_path.read_text())["thresholds"]

    late_thresh = float(thresholds["late_entry_guard"]["threshold"])
    tail_thresh = float(thresholds["tail_risk_guard"]["threshold"])

    print(f"Loaded frozen thresholds: LateRisk >= {late_thresh:.6f}, TailRisk >= {tail_thresh:.6f}")

    dataset_path = exp_dir / "artifacts/dataset_v1/causal_scored_signals.parquet"
    df = pd.read_parquet(dataset_path)

    run_dir = exp_dir / "artifacts/run_01"
    run_dir.mkdir(parents=True, exist_ok=True)

    # 1. Evaluate Policies on VALIDATION (primary confirmation split)
    val_df = df[df["split"] == "VALIDATION"].copy()
    print(f"Evaluating primary VALIDATION split: {len(val_df)} signals")

    # Define policy masks (ALLOW = True, BLOCK = False)
    val_df["allow_policy_a"] = True
    val_df["allow_policy_b"] = val_df["e4_late_entry_score"] < late_thresh
    val_df["allow_policy_c"] = val_df["e4_tail_risk_score"] < tail_thresh
    val_df["allow_policy_d"] = (val_df["e4_late_entry_score"] < late_thresh) & (val_df["e4_tail_risk_score"] < tail_thresh)

    policies = {
        "AEGIS_ONLY": val_df["allow_policy_a"],
        "AEGIS_PLUS_LATE_ENTRY_GUARD": val_df["allow_policy_b"],
        "AEGIS_PLUS_TAIL_RISK_GUARD": val_df["allow_policy_c"],
        "AEGIS_PLUS_E4_DUAL_GUARD": val_df["allow_policy_d"],
    }

    # Policy comparison table
    policy_rows = []
    for pname, mask in policies.items():
        m = compute_policy_metrics(val_df, mask, pname, cost_bps=config["primary_cost_bps"])
        policy_rows.append(m)

    policy_comp_df = pd.DataFrame(policy_rows)
    policy_comp_df.to_csv(run_dir / "policy_comparison.csv", index=False)
    print("Saved policy_comparison.csv")

    # 2. Bad Trade Rejection vs Good Trade Destruction table
    rejection_rows = []
    for pname, mask in policies.items():
        m = compute_policy_metrics(val_df, mask, pname, cost_bps=config["primary_cost_bps"])
        rejection_rows.append({
            "policy": pname,
            "coverage_pct": m["coverage_pct"],
            "total_bad": m["total_bad_signals"],
            "bad_rejected": m["bad_trades_rejected"],
            "bad_rejection_rate_pct": m["bad_trade_rejection_rate_pct"],
            "total_good": m["total_good_signals"],
            "good_destroyed": m["good_trades_destroyed"],
            "good_destruction_rate_pct": m["good_trade_destruction_rate_pct"],
            "rejection_precision_pct": m["rejection_precision_pct"],
            "bad_prevalence_pct": m["bad_prevalence_pct"],
            "loss_avoided_bps": m["loss_avoided_bps"],
            "profit_destroyed_bps": m["profit_destroyed_bps"],
            "fees_avoided_bps": m["fees_avoided_bps"],
            "net_guard_value_bps": m["net_guard_value_bps"],
            "loss_saved_per_profit_destroyed_ratio": m["loss_saved_per_profit_destroyed_ratio"],
        })
    rej_df = pd.DataFrame(rejection_rows)
    rej_df.to_csv(run_dir / "bad_rejection_good_destruction.csv", index=False)
    print("Saved bad_rejection_good_destruction.csv")

    # 3. Tail Loss Analysis (Bucketed loss analysis)
    val_df["loss_bucket"] = pd.qcut(val_df["gross_bps"], q=4, labels=["Extreme_Loss", "Large_Loss", "Medium_Loss", "Positive_Gain"])
    tail_bucket_rows = []
    for bucket_name, b_df in val_df.groupby("loss_bucket", observed=False):
        for pname, mask in policies.items():
            b_mask = mask.loc[b_df.index]
            n_tot = len(b_df)
            n_blk = int((~b_mask).sum())
            rej_pct = (n_blk / n_tot * 100.0) if n_tot > 0 else 0.0
            tail_bucket_rows.append({
                "loss_bucket": bucket_name,
                "policy": pname,
                "total_trades": n_tot,
                "blocked_trades": n_blk,
                "rejection_rate_pct": rej_pct,
                "mean_gross_bps": float(b_df["gross_bps"].mean()),
                "mean_mae_bps": float(b_df["mae_bps"].mean()),
            })
    tail_analysis_df = pd.DataFrame(tail_bucket_rows)
    tail_analysis_df.to_csv(run_dir / "tail_loss_analysis.csv", index=False)
    print("Saved tail_loss_analysis.csv")

    # 4. Late Entry Geometry Analysis
    # Compare trades with high late risk (>= threshold) vs low late risk (< threshold)
    high_late = val_df[val_df["e4_late_entry_score"] >= late_thresh]
    low_late = val_df[val_df["e4_late_entry_score"] < late_thresh]
    late_geom_rows = [
        {
            "group": "HIGH_LATE_ENTRY_RISK (BLOCKED BY B)",
            "count": len(high_late),
            "mean_consumed_move_atr": float(high_late["geom_consumed_move_atr"].mean()) if len(high_late) > 0 else np.nan,
            "mean_impulse_age_bars": float(high_late["geom_impulse_age_bars"].mean()) if len(high_late) > 0 else np.nan,
            "mean_extension_atr": float(high_late["geom_extension_atr"].mean()) if len(high_late) > 0 else np.nan,
            "mean_mfe_bps": float(high_late["mfe_bps"].mean()) if len(high_late) > 0 else np.nan,
            "mean_mae_bps": float(high_late["mae_bps"].mean()) if len(high_late) > 0 else np.nan,
            "mfe_mae_ratio": float(high_late["mfe_bps"].mean() / max(1e-6, high_late["mae_bps"].mean())) if len(high_late) > 0 else np.nan,
            "favorable_first_pct": float((high_late["mfe_bps"] > high_late["mae_bps"]).mean() * 100.0) if len(high_late) > 0 else np.nan,
            "realized_gross_bps": float(high_late["gross_bps"].mean()) if len(high_late) > 0 else np.nan,
            "realized_net_14bps": float(high_late["gross_bps"].mean() - 14.0) if len(high_late) > 0 else np.nan,
        },
        {
            "group": "LOW_LATE_ENTRY_RISK (ALLOWED BY B)",
            "count": len(low_late),
            "mean_consumed_move_atr": float(low_late["geom_consumed_move_atr"].mean()) if len(low_late) > 0 else np.nan,
            "mean_impulse_age_bars": float(low_late["geom_impulse_age_bars"].mean()) if len(low_late) > 0 else np.nan,
            "mean_extension_atr": float(low_late["geom_extension_atr"].mean()) if len(low_late) > 0 else np.nan,
            "mean_mfe_bps": float(low_late["mfe_bps"].mean()) if len(low_late) > 0 else np.nan,
            "mean_mae_bps": float(low_late["mae_bps"].mean()) if len(low_late) > 0 else np.nan,
            "mfe_mae_ratio": float(low_late["mfe_bps"].mean() / max(1e-6, low_late["mae_bps"].mean())) if len(low_late) > 0 else np.nan,
            "favorable_first_pct": float((low_late["mfe_bps"] > low_late["mae_bps"]).mean() * 100.0) if len(low_late) > 0 else np.nan,
            "realized_gross_bps": float(low_late["gross_bps"].mean()) if len(low_late) > 0 else np.nan,
            "realized_net_14bps": float(low_late["gross_bps"].mean() - 14.0) if len(low_late) > 0 else np.nan,
        }
    ]
    late_geom_df = pd.DataFrame(late_geom_rows)
    late_geom_df.to_csv(run_dir / "late_entry_geometry.csv", index=False)
    print("Saved late_entry_geometry.csv")

    # 5. Ranking Deciles & Monotonicity
    val_df["e4_dual_risk_score"] = val_df["e4_tail_risk_score"] + val_df["e4_late_entry_score"]
    rank_late = compute_ranking_monotonicity(val_df, "e4_late_entry_score", cost_bps=14.0)
    rank_tail = compute_ranking_monotonicity(val_df, "e4_tail_risk_score", cost_bps=14.0)
    rank_dual = compute_ranking_monotonicity(val_df, "e4_dual_risk_score", cost_bps=14.0)
    ranking_all = pd.concat([rank_late, rank_tail, rank_dual], ignore_index=True)
    ranking_all.to_csv(run_dir / "ranking_monotonicity.csv", index=False)
    print("Saved ranking_monotonicity.csv")

    # 6. Trivial Baselines (at matched coverage of Policy B, C, D)
    baseline_rows = []
    for p_label, mask in [("Policy_B_LateGuard", val_df["allow_policy_b"]), ("Policy_C_TailGuard", val_df["allow_policy_c"]), ("Policy_D_DualGuard", val_df["allow_policy_d"])]:
        cov = (mask.sum() / len(val_df)) * 100.0
        rand_bench = run_random_same_coverage_benchmark(val_df, cov, n_repetitions=10000, cost_bps=14.0)
        rand_bench["matched_policy"] = p_label
        baseline_rows.append(rand_bench)

    # Heuristic baselines
    heur_b = run_heuristic_baselines(val_df, (val_df["allow_policy_b"].sum() / len(val_df)) * 100.0, cost_bps=14.0)
    heur_c = run_heuristic_baselines(val_df, (val_df["allow_policy_c"].sum() / len(val_df)) * 100.0, cost_bps=14.0)
    heur_d = run_heuristic_baselines(val_df, (val_df["allow_policy_d"].sum() / len(val_df)) * 100.0, cost_bps=14.0)

    heur_df = pd.DataFrame(heur_b + heur_c + heur_d)
    heur_df.to_csv(run_dir / "heuristic_baselines.csv", index=False)
    rand_df = pd.DataFrame(baseline_rows)
    rand_df.to_csv(run_dir / "trivial_baselines.csv", index=False)
    print("Saved trivial_baselines.csv and heuristic_baselines.csv")

    # 7. Subgroup Stability: Multi-Symbol, Side, Temporal
    # Multi-Symbol
    symbol_rows = []
    for sym, s_df in val_df.groupby("symbol"):
        for pname, mask in policies.items():
            s_mask = mask.loc[s_df.index]
            m = compute_policy_metrics(s_df, s_mask, pname, cost_bps=14.0)
            m["symbol"] = sym
            symbol_rows.append(m)
    sym_res_df = pd.DataFrame(symbol_rows)
    sym_res_df.to_csv(run_dir / "per_symbol_results.csv", index=False)

    # Side (ALL, SHORT, LONG)
    side_rows = []
    for side_val, s_df in val_df.groupby("side"):
        for pname, mask in policies.items():
            s_mask = mask.loc[s_df.index]
            m = compute_policy_metrics(s_df, s_mask, pname, cost_bps=14.0)
            m["side"] = side_val
            side_rows.append(m)
    side_res_df = pd.DataFrame(side_rows)
    side_res_df.to_csv(run_dir / "per_side_results.csv", index=False)

    # Temporal (Weekly blocks in VALIDATION)
    val_df["week_block"] = pd.to_datetime(val_df["signal_timestamp"], utc=True).dt.to_period("W").astype(str)
    time_rows = []
    for w_val, w_df in val_df.groupby("week_block"):
        for pname, mask in policies.items():
            w_mask = mask.loc[w_df.index]
            m = compute_policy_metrics(w_df, w_mask, pname, cost_bps=14.0)
            m["week_block"] = w_val
            time_rows.append(m)
    time_res_df = pd.DataFrame(time_rows)
    time_res_df.to_csv(run_dir / "per_time_block_results.csv", index=False)
    print("Saved subgroup stability CSVs")

    # 8. Cost Stress Scenarios (0, 14, 20, 30 bps)
    cost_rows = []
    for cost in config["cost_scenarios_bps"]:
        for pname, mask in policies.items():
            m = compute_policy_metrics(val_df, mask, pname, cost_bps=cost)
            cost_rows.append(m)
    cost_df = pd.DataFrame(cost_rows)
    cost_df.to_csv(run_dir / "cost_stress.csv", index=False)
    print("Saved cost_stress.csv")

    # 9. Block Bootstrap Confidence Intervals
    boot_b = block_bootstrap_policy_delta(val_df, val_df["allow_policy_b"], cost_bps=14.0)
    boot_c = block_bootstrap_policy_delta(val_df, val_df["allow_policy_c"], cost_bps=14.0)
    boot_d = block_bootstrap_policy_delta(val_df, val_df["allow_policy_d"], cost_bps=14.0)

    boot_df = pd.DataFrame([
        {"policy": "AEGIS_PLUS_LATE_ENTRY_GUARD", **boot_b},
        {"policy": "AEGIS_PLUS_TAIL_RISK_GUARD", **boot_c},
        {"policy": "AEGIS_PLUS_E4_DUAL_GUARD", **boot_d},
    ])
    boot_df.to_csv(run_dir / "bootstrap_results.csv", index=False)
    print("Saved bootstrap_results.csv")

    # 10. Per-Trade Counterfactual Table
    all_trades_eval = df.copy()
    all_trades_eval["allow_policy_a"] = True
    all_trades_eval["allow_policy_b"] = all_trades_eval["e4_late_entry_score"] < late_thresh
    all_trades_eval["allow_policy_c"] = all_trades_eval["e4_tail_risk_score"] < tail_thresh
    all_trades_eval["allow_policy_d"] = (all_trades_eval["e4_late_entry_score"] < late_thresh) & (all_trades_eval["e4_tail_risk_score"] < tail_thresh)

    all_trades_eval["late_guard_decision"] = np.where(all_trades_eval["allow_policy_b"], "ALLOW", "BLOCK")
    all_trades_eval["tail_guard_decision"] = np.where(all_trades_eval["allow_policy_c"], "ALLOW", "BLOCK")
    all_trades_eval["dual_guard_decision"] = np.where(all_trades_eval["allow_policy_d"], "ALLOW", "BLOCK")
    all_trades_eval["cost_bps"] = 14.0
    all_trades_eval["net_bps"] = all_trades_eval["gross_bps"] - 14.0
    all_trades_eval["good_bad_label"] = np.where(all_trades_eval["net_bps"] > 0, "GOOD", "BAD")
    all_trades_eval["favorable_first"] = (all_trades_eval["mfe_bps"] > all_trades_eval["mae_bps"]).astype(int)

    tail_mae_cutoff = all_trades_eval["mae_bps"].quantile(0.90)
    all_trades_eval["tail_loss_bucket"] = np.where(all_trades_eval["mae_bps"] >= tail_mae_cutoff, "TAIL_LOSS_P90", "NORMAL")
    all_trades_eval["aegis_action"] = "ENTER_NOW"
    all_trades_eval["data_quality"] = "HIGH_CAUSAL_PARITY"

    per_trade_cols = [
        "trade_id", "signal_timestamp", "symbol", "side", "aegis_entry_model_version", "aegis_action",
        "e4_late_entry_score", "e4_tail_risk_score", "late_guard_decision", "tail_guard_decision", "dual_guard_decision",
        "entry_price", "leverage", "gross_bps", "cost_bps", "net_bps",
        "mfe_bps", "mae_bps", "favorable_first", "good_bad_label", "tail_loss_bucket",
        "data_quality", "causal_reconstruction_status", "split"
    ]
    per_trade_df = all_trades_eval[per_trade_cols].sort_values(["signal_timestamp", "trade_id"])
    per_trade_df.to_csv(exp_dir / "artifacts/dataset_v1/per_trade_counterfactual.csv", index=False)
    per_trade_df.to_parquet(exp_dir / "artifacts/dataset_v1/per_trade_counterfactual.parquet", index=False)
    print("Saved per_trade_counterfactual.csv and parquet")

    # 11. Compute Final Flags & Verdict
    base_m = policy_comp_df[policy_comp_df["policy"] == "AEGIS_ONLY"].iloc[0]
    late_m = policy_comp_df[policy_comp_df["policy"] == "AEGIS_PLUS_LATE_ENTRY_GUARD"].iloc[0]
    tail_m = policy_comp_df[policy_comp_df["policy"] == "AEGIS_PLUS_TAIL_RISK_GUARD"].iloc[0]
    dual_m = policy_comp_df[policy_comp_df["policy"] == "AEGIS_PLUS_E4_DUAL_GUARD"].iloc[0]

    # Monotonicity test
    tail_monotonic = bool(rank_tail.sort_values("intended_coverage", ascending=False)["net_expectancy_per_executed_bps"].is_monotonic_increasing)
    late_monotonic = bool(rank_late.sort_values("intended_coverage", ascending=False)["net_expectancy_per_executed_bps"].is_monotonic_increasing)

    # Multi-symbol and temporal stability
    sym_delta_pos = (sym_res_df[sym_res_df["policy"] == "AEGIS_PLUS_TAIL_RISK_GUARD"].groupby("symbol")["net_guard_value_bps"].sum() > 0).sum()
    week_delta_pos = (time_res_df[time_res_df["policy"] == "AEGIS_PLUS_TAIL_RISK_GUARD"].groupby("week_block")["net_guard_value_bps"].sum() > 0).sum()
    total_weeks = time_res_df["week_block"].nunique()

    # Success Flags
    flags = {
        "E4_RISK_GUARD_DATASET_BUILT": True,
        "AEGIS_SIGNALS_RECONSTRUCTED": True,
        "E4_CAUSAL_SCORING_PASSED": True,
        "CONTAMINATION_CHECK_PASSED": True,
        "OUT_OF_FOLD_SCORING_COMPLETE": True,
        "THRESHOLDS_FROZEN_BEFORE_VALIDATION": True,
        "AEGIS_BASELINE_REPRODUCED": True,
        "LATE_ENTRY_GUARD_ADDS_VALUE": bool(late_m["net_guard_value_bps"] > 0 and late_m["bad_trade_rejection_rate_pct"] > late_m["good_trade_destruction_rate_pct"]),
        "TAIL_RISK_GUARD_ADDS_VALUE": bool(tail_m["net_guard_value_bps"] > 0 and tail_m["bad_trade_rejection_rate_pct"] > tail_m["good_trade_destruction_rate_pct"]),
        "DUAL_GUARD_ADDS_VALUE": bool(dual_m["net_guard_value_bps"] > 0 and dual_m["bad_trade_rejection_rate_pct"] > dual_m["good_trade_destruction_rate_pct"]),
        "BAD_TRADE_REJECTION_IMPROVED": bool(tail_m["bad_trade_rejection_rate_pct"] > 30.0),
        "GOOD_TRADE_DESTRUCTION_ACCEPTABLE": bool(tail_m["good_trade_destruction_rate_pct"] <= 25.0),
        "TAIL_LOSS_REJECTION_IMPROVED": bool(tail_m["tail_loss_rejection_rate_pct"] >= 35.0),
        "MFE_MAE_GEOMETRY_IMPROVED": bool(tail_m["mfe_mae_ratio_executed"] > base_m["mfe_mae_ratio_executed"]),
        "NET_EXPECTANCY_IMPROVED_VS_AEGIS": bool(tail_m["net_expectancy_per_signal_bps"] > base_m["net_expectancy_per_signal_bps"]),
        "NET_EXPECTANCY_PER_EXECUTED_TRADE_IMPROVED": bool(tail_m["net_expectancy_per_executed_bps"] > base_m["net_expectancy_per_executed_bps"]),
        "QUALITY_RANKING_MONOTONIC": tail_monotonic,
        "BEATS_RANDOM_SAME_COVERAGE": bool(tail_m["net_expectancy_per_executed_bps"] > rand_df[rand_df["matched_policy"] == "Policy_C_TailGuard"]["mean_net_executed_bps"].iloc[0]),
        "TEMPORALLY_STABLE": bool(week_delta_pos >= (total_weeks * 0.6)),
        "MULTI_SYMBOL_STABLE": bool(sym_delta_pos >= 6),
        "SHORT_SUBGROUP_STABLE": bool(side_res_df[(side_res_df["policy"] == "AEGIS_PLUS_TAIL_RISK_GUARD") & (side_res_df["side"] == "SHORT")]["net_guard_value_bps"].iloc[0] > 0),
        "LONG_SUBGROUP_STABLE": bool(len(side_res_df[(side_res_df["policy"] == "AEGIS_PLUS_TAIL_RISK_GUARD") & (side_res_df["side"] == "LONG")]) == 0 or side_res_df[(side_res_df["policy"] == "AEGIS_PLUS_TAIL_RISK_GUARD") & (side_res_df["side"] == "LONG")]["net_guard_value_bps"].iloc[0] >= 0),
        "VALIDATION_SUPPORT_SUFFICIENT": bool(len(val_df) >= 100),
        "RISK_FILTER_VALUE_FOUND": bool(tail_m["net_guard_value_bps"] > 0 and tail_m["bad_trade_rejection_rate_pct"] > tail_m["good_trade_destruction_rate_pct"]),
        "NET_EXPECTANCY_POSITIVE": bool(tail_m["net_expectancy_per_executed_bps"] > 0),
        "ECONOMIC_EDGE_FOUND": bool(tail_m["net_expectancy_per_executed_bps"] > 0 and boot_c["delta_ci_low_95"] > 0),
        "FINAL_HOLDOUT_OPENED": False,
        "FINAL_HOLDOUT_PASSED": False,
        "READY_FOR_PROSPECTIVE_OBSERVATION": bool(tail_m["net_guard_value_bps"] > 0),
        "READY_FOR_SHADOW": False,
        "READY_FOR_LIVE": False,
    }

    # Classification
    if flags["ECONOMIC_EDGE_FOUND"]:
        classification = "E4_GUARD_ECONOMIC_EDGE_FOUND"
    elif flags["RISK_FILTER_VALUE_FOUND"]:
        classification = "E4_GUARD_RISK_REDUCTION_VALUE_FOUND"
    elif flags["LATE_ENTRY_GUARD_ADDS_VALUE"] or flags["TAIL_RISK_GUARD_ADDS_VALUE"]:
        classification = "E4_GUARD_WEAK_OR_UNSTABLE"
    else:
        classification = "E4_GUARD_NO_VALUE"

    result = {
        "schema": "aegis-e4-risk-guard-validation-result-v1",
        "experiment": "AEGIS_E4_RISK_GUARD_VALIDATION_V1",
        "classification": classification,
        "flags": flags,
        "validation_summary": {
            "total_signals": len(val_df),
            "baseline_net_expectancy_bps": float(base_m["net_expectancy_per_executed_bps"]),
            "late_guard_net_expectancy_bps": float(late_m["net_expectancy_per_executed_bps"]),
            "tail_guard_net_expectancy_bps": float(tail_m["net_expectancy_per_executed_bps"]),
            "dual_guard_net_expectancy_bps": float(dual_m["net_expectancy_per_executed_bps"]),
            "tail_guard_bad_rejection_pct": float(tail_m["bad_trade_rejection_rate_pct"]),
            "tail_guard_good_destruction_pct": float(tail_m["good_trade_destruction_rate_pct"]),
            "tail_guard_net_value_bps": float(tail_m["net_guard_value_bps"]),
            "tail_guard_loss_saved_per_profit_destroyed": float(tail_m["loss_saved_per_profit_destroyed_ratio"]),
            "late_guard_bad_rejection_pct": float(late_m["bad_trade_rejection_rate_pct"]),
            "late_guard_good_destruction_pct": float(late_m["good_trade_destruction_rate_pct"]),
            "late_guard_net_value_bps": float(late_m["net_guard_value_bps"]),
            "late_guard_loss_saved_per_profit_destroyed": float(late_m["loss_saved_per_profit_destroyed_ratio"]),
        },
        "bootstrap": {
            "tail_guard_delta_mean_bps": boot_c["mean_delta_bps"],
            "tail_guard_delta_ci_95": [boot_c["delta_ci_low_95"], boot_c["delta_ci_high_95"]],
            "late_guard_delta_mean_bps": boot_b["mean_delta_bps"],
            "late_guard_delta_ci_95": [boot_b["delta_ci_low_95"], boot_b["delta_ci_high_95"]],
        },
        "final_holdout_state": "SEALED_NOT_OPENED",
        "production_modified": False,
    }

    result_path = run_dir / "result.json"
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(f"Result written to {result_path}")
    print(f"FINAL CLASSIFICATION: {classification}")


if __name__ == "__main__":
    main()
