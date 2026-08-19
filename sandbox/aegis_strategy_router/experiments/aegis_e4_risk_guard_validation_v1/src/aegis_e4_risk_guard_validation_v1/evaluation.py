"""Comprehensive evaluation and metric calculations for E4 Risk Guard Validation."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


def compute_policy_metrics(
    df: pd.DataFrame,
    allowed_mask: pd.Series,
    policy_name: str,
    cost_bps: float = 14.0,
) -> dict[str, Any]:
    n_total = len(df)
    n_exec = int(allowed_mask.sum())
    n_block = n_total - n_exec
    cov = (n_exec / n_total) * 100.0 if n_total > 0 else 0.0

    # Economic outcomes per trade
    # gross_bps is unleveraged return in bps
    # net_bps = gross_bps - cost_bps
    df_eval = df.copy()
    df_eval["net_bps"] = df_eval["gross_bps"] - cost_bps
    df_eval["is_bad"] = df_eval["net_bps"] <= 0
    df_eval["is_good"] = df_eval["net_bps"] > 0

    # Identify tail loss bucket (worst 10% of trades by net_bps or MAE >= P90)
    mae_p90 = df_eval["mae_bps"].quantile(0.90)
    df_eval["is_tail_loss"] = (df_eval["net_bps"] <= df_eval["net_bps"].quantile(0.10)) | (df_eval["mae_bps"] >= mae_p90)

    total_bad = int(df_eval["is_bad"].sum())
    total_good = int(df_eval["is_good"].sum())
    total_tail = int(df_eval["is_tail_loss"].sum())

    blocked_df = df_eval[~allowed_mask]
    exec_df = df_eval[allowed_mask]

    bad_rej_count = int(blocked_df["is_bad"].sum())
    good_dest_count = int(blocked_df["is_good"].sum())
    tail_rej_count = int(blocked_df["is_tail_loss"].sum())

    bad_rej_rate = (bad_rej_count / total_bad * 100.0) if total_bad > 0 else 0.0
    good_dest_rate = (good_dest_count / total_good * 100.0) if total_good > 0 else 0.0
    tail_rej_rate = (tail_rej_count / total_tail * 100.0) if total_tail > 0 else 0.0

    rej_prec = (bad_rej_count / n_block * 100.0) if n_block > 0 else 0.0
    bad_prev = (total_bad / n_total * 100.0) if n_total > 0 else 0.0

    # Economic sums on blocked trades
    # Losses avoided: sum of absolute negative net_bps on blocked bad trades
    loss_avoided_bps = float(blocked_df.loc[blocked_df.is_bad, "net_bps"].abs().sum())
    profit_destroyed_bps = float(blocked_df.loc[blocked_df.is_good, "net_bps"].sum())
    fees_avoided_bps = float(n_block * cost_bps)

    # Net guard value: Difference in total net portfolio bps (with guard vs without guard)
    total_net_all = float(df_eval["net_bps"].sum())
    total_net_exec = float(exec_df["net_bps"].sum()) if len(exec_df) > 0 else 0.0
    net_guard_value_bps = total_net_exec - total_net_all

    loss_per_profit_ratio = loss_avoided_bps / max(1e-6, profit_destroyed_bps)

    # Per original signal expectancy (blocked trades contribute 0 bps)
    gross_per_signal = float(exec_df["gross_bps"].sum() / n_total) if n_total > 0 else 0.0
    net_per_signal = float(exec_df["net_bps"].sum() / n_total) if n_total > 0 else 0.0

    # Per executed trade expectancy
    gross_per_exec = float(exec_df["gross_bps"].mean()) if n_exec > 0 else 0.0
    net_per_exec = float(exec_df["net_bps"].mean()) if n_exec > 0 else 0.0

    win_rate = (exec_df["is_good"].mean() * 100.0) if n_exec > 0 else 0.0
    pos_sum = float(exec_df.loc[exec_df.is_good, "net_bps"].sum()) if n_exec > 0 else 0.0
    neg_sum = float(exec_df.loc[exec_df.is_bad, "net_bps"].abs().sum()) if n_exec > 0 else 0.0
    profit_factor = (pos_sum / neg_sum) if neg_sum > 0 else (99.0 if pos_sum > 0 else 1.0)

    # Geometry on executed
    mfe_mean = float(exec_df["mfe_bps"].mean()) if n_exec > 0 else 0.0
    mae_mean = float(exec_df["mae_bps"].mean()) if n_exec > 0 else 0.0
    mfe_mae_ratio = (mfe_mean / max(1e-6, mae_mean)) if n_exec > 0 else 0.0
    tail_mae_p95 = float(exec_df["mae_bps"].quantile(0.95)) if n_exec > 0 else 0.0

    # Favorable first
    fav_first = float((exec_df["mfe_bps"] > exec_df["mae_bps"]).mean() * 100.0) if n_exec > 0 else 0.0

    # Expected shortfall (worst 5%)
    es_5 = float(exec_df.loc[exec_df["net_bps"] <= exec_df["net_bps"].quantile(0.05), "net_bps"].mean()) if n_exec >= 20 else net_per_exec

    # Max Drawdown in cumulative net bps
    cum_net = exec_df["net_bps"].cumsum() if n_exec > 0 else pd.Series([0.0])
    peak = cum_net.cummax()
    dd = peak - cum_net
    max_dd = float(dd.max()) if len(dd) > 0 else 0.0

    return {
        "policy": policy_name,
        "cost_bps": cost_bps,
        "total_signals": n_total,
        "executed_count": n_exec,
        "blocked_count": n_block,
        "coverage_pct": cov,
        "total_bad_signals": total_bad,
        "total_good_signals": total_good,
        "total_tail_loss_signals": total_tail,
        "bad_trades_rejected": bad_rej_count,
        "good_trades_destroyed": good_dest_count,
        "tail_losses_rejected": tail_rej_count,
        "bad_trade_rejection_rate_pct": bad_rej_rate,
        "good_trade_destruction_rate_pct": good_dest_rate,
        "tail_loss_rejection_rate_pct": tail_rej_rate,
        "rejection_precision_pct": rej_prec,
        "bad_prevalence_pct": bad_prev,
        "loss_avoided_bps": loss_avoided_bps,
        "profit_destroyed_bps": profit_destroyed_bps,
        "fees_avoided_bps": fees_avoided_bps,
        "net_guard_value_bps": net_guard_value_bps,
        "loss_saved_per_profit_destroyed_ratio": loss_per_profit_ratio,
        "gross_expectancy_per_signal_bps": gross_per_signal,
        "net_expectancy_per_signal_bps": net_per_signal,
        "gross_expectancy_per_executed_bps": gross_per_exec,
        "net_expectancy_per_executed_bps": net_per_exec,
        "win_rate_executed_pct": win_rate,
        "profit_factor_executed": profit_factor,
        "mfe_mean_executed_bps": mfe_mean,
        "mae_mean_executed_bps": mae_mean,
        "mfe_mae_ratio_executed": mfe_mae_ratio,
        "favorable_first_pct": fav_first,
        "tail_mae_p95_executed_bps": tail_mae_p95,
        "expected_shortfall_5pct_bps": es_5,
        "max_drawdown_bps": max_dd,
    }


def compute_ranking_monotonicity(df: pd.DataFrame, score_column: str, cost_bps: float = 14.0) -> pd.DataFrame:
    rows = []
    # Rank deciles: 100%, 90%, 80%, 70%, 60%, 50%
    for cov in [1.0, 0.9, 0.8, 0.7, 0.6, 0.5]:
        n_keep = max(1, int(math.ceil(len(df) * cov)))
        # Keep lowest risk scores
        selected = df.nsmallest(n_keep, score_column)
        mask = df.index.isin(selected.index)
        metrics = compute_policy_metrics(df, mask, f"TOP_{int(cov*100)}_PCT_QUALITY", cost_bps=cost_bps)
        metrics["intended_coverage"] = cov
        metrics["score_column"] = score_column
        rows.append(metrics)

    res_df = pd.DataFrame(rows)
    return res_df
