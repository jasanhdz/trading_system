"""Vectorized benchmark baselines for E4 Risk Guard Validation."""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Any
from .evaluation import compute_policy_metrics


def run_random_same_coverage_benchmark(
    df: pd.DataFrame,
    target_coverage_pct: float,
    n_repetitions: int = 10000,
    seed: int = 20260819,
    cost_bps: float = 14.0,
) -> dict[str, Any]:
    n_total = len(df)
    n_keep = max(1, int(round(n_total * (target_coverage_pct / 100.0))))
    n_block = n_total - n_keep

    net_bps = df["gross_bps"].to_numpy(float) - cost_bps
    is_bad = net_bps <= 0
    is_good = net_bps > 0
    total_bad = int(is_bad.sum())
    total_good = int(is_good.sum())
    total_net = float(net_bps.sum())

    rng = np.random.default_rng(seed)

    # Vectorized random permutations
    perm_matrix = np.array([rng.permutation(n_total) for _ in range(n_repetitions)])
    keep_indices = perm_matrix[:, :n_keep]
    block_indices = perm_matrix[:, n_keep:]

    # For each repetition, calculate metrics
    # executed net means
    exec_net_means = np.take(net_bps, keep_indices).mean(axis=1)

    # blocked bad counts
    blocked_bad_counts = np.take(is_bad, block_indices).sum(axis=1)
    bad_rej_rates = (blocked_bad_counts / total_bad * 100.0) if total_bad > 0 else np.zeros(n_repetitions)

    # blocked good counts
    blocked_good_counts = np.take(is_good, block_indices).sum(axis=1)
    good_dest_rates = (blocked_good_counts / total_good * 100.0) if total_good > 0 else np.zeros(n_repetitions)

    # net guard value
    exec_net_sums = np.take(net_bps, keep_indices).sum(axis=1)
    net_guard_values = exec_net_sums - total_net

    return {
        "benchmark": f"RANDOM_SAME_COVERAGE_{int(round(target_coverage_pct))}PCT",
        "target_coverage_pct": float(target_coverage_pct),
        "n_repetitions": n_repetitions,
        "mean_net_executed_bps": float(np.mean(exec_net_means)),
        "ci_low_net_executed_bps": float(np.percentile(exec_net_means, 2.5)),
        "ci_high_net_executed_bps": float(np.percentile(exec_net_means, 97.5)),
        "mean_bad_rejection_pct": float(np.mean(bad_rej_rates)),
        "mean_good_destruction_pct": float(np.mean(good_dest_rates)),
        "mean_net_guard_value_bps": float(np.mean(net_guard_values)),
    }


def run_heuristic_baselines(
    df: pd.DataFrame,
    target_coverage_pct: float,
    cost_bps: float = 14.0,
) -> list[dict[str, Any]]:
    n_total = len(df)
    n_keep = max(1, int(round(n_total * (target_coverage_pct / 100.0))))
    results = []

    # 1. Simple Late-Move: block highest geom_consumed_move_atr / prior move
    if "geom_consumed_move_atr" in df:
        keep = df.nsmallest(n_keep, "geom_consumed_move_atr").index
        m = compute_policy_metrics(df, df.index.isin(keep), f"BASELINE_SIMPLE_LATE_MOVE_ATR_{int(round(target_coverage_pct))}COV", cost_bps=cost_bps)
        results.append(m)

    # 2. Simple Volatility: block highest geom_15m_atr_percentile
    if "geom_15m_atr_percentile" in df:
        keep = df.nsmallest(n_keep, "geom_15m_atr_percentile").index
        m = compute_policy_metrics(df, df.index.isin(keep), f"BASELINE_SIMPLE_VOLATILITY_ATR_P96_{int(round(target_coverage_pct))}COV", cost_bps=cost_bps)
        results.append(m)

    # 3. Aegis Confidence: keep highest turbo_score
    if "turbo_score" in df and df["turbo_score"].notna().any():
        keep = df.nlargest(n_keep, "turbo_score").index
        m = compute_policy_metrics(df, df.index.isin(keep), f"BASELINE_AEGIS_TURBO_SCORE_{int(round(target_coverage_pct))}COV", cost_bps=cost_bps)
        results.append(m)

    return results
