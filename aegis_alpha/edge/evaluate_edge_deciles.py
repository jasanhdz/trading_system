#!/usr/bin/env python3
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from aegis_alpha.edge.common import load_model_bundle, profit_factor, safe_float, write_json


def _side_metrics(
    score: np.ndarray,
    actual_return: np.ndarray,
    mfe: np.ndarray,
    mae: np.ndarray,
    pct: float,
) -> dict[str, Any]:
    if len(score) == 0:
        return {}
    cutoff = float(np.quantile(score, 1.0 - pct))
    mask = score >= cutoff
    returns = actual_return[mask]
    return {
        "pct": pct,
        "score_cutoff": cutoff,
        "count": int(mask.sum()),
        "sample_pct": safe_float(np.mean(mask)),
        "score_avg": safe_float(np.mean(score[mask])) if mask.any() else 0.0,
        "avg_return_after_fees": safe_float(np.mean(returns)) if len(returns) else 0.0,
        "median_return_after_fees": safe_float(np.median(returns)) if len(returns) else 0.0,
        "win_rate": safe_float(np.mean(returns > 0.0)) if len(returns) else 0.0,
        "profit_factor": safe_float(profit_factor(returns)) if len(returns) else 0.0,
        "avg_mfe": safe_float(np.mean(mfe[mask])) if mask.any() else 0.0,
        "avg_mae": safe_float(np.mean(mae[mask])) if mask.any() else 0.0,
    }


def _combined_policy_metrics(
    long_score: np.ndarray,
    short_score: np.ndarray,
    long_return: np.ndarray,
    short_return: np.ndarray,
    min_edge: float,
    min_gap: float,
) -> dict[str, Any]:
    long_mask = (long_score >= min_edge) & ((long_score - short_score) >= min_gap)
    short_mask = (short_score >= min_edge) & ((short_score - long_score) >= min_gap)
    conflict = long_mask & short_mask
    long_mask[conflict] = False
    short_mask[conflict] = False
    returns = np.concatenate((long_return[long_mask], short_return[short_mask]))
    return {
        "min_edge": min_edge,
        "min_gap": min_gap,
        "long_count": int(long_mask.sum()),
        "short_count": int(short_mask.sum()),
        "entry_count": int(len(returns)),
        "entry_pct": safe_float(len(returns) / max(len(long_score), 1)),
        "avg_return_after_fees": safe_float(np.mean(returns)) if len(returns) else 0.0,
        "median_return_after_fees": safe_float(np.median(returns)) if len(returns) else 0.0,
        "win_rate": safe_float(np.mean(returns > 0.0)) if len(returns) else 0.0,
        "profit_factor": safe_float(profit_factor(returns)) if len(returns) else 0.0,
    }


def evaluate_edge_deciles(
    dataset_path: Path,
    model_path: Path,
    output_path: Path,
    test_pct: float,
    top_pcts: list[float],
) -> Path:
    data = np.load(dataset_path, allow_pickle=True)
    x = data["x"].astype(np.float32)
    split = int(len(x) * (1.0 - test_pct))
    x_test = x[split:]

    long_return = data["long_return"].astype(np.float32)[split:]
    short_return = data["short_return"].astype(np.float32)[split:]
    long_mfe = data["long_mfe"].astype(np.float32)[split:]
    long_mae = data["long_mae"].astype(np.float32)[split:]
    short_mfe = data["short_mfe"].astype(np.float32)[split:]
    short_mae = data["short_mae"].astype(np.float32)[split:]
    timestamps = data["timestamp"].astype(str)[split:]

    bundle = load_model_bundle(model_path)
    long_score = bundle["long_classifier"].predict_proba(x_test)[:, 1]
    short_score = bundle["short_classifier"].predict_proba(x_test)[:, 1]
    expected_long_return = bundle["long_return_regressor"].predict(x_test)
    expected_short_return = bundle["short_return_regressor"].predict(x_test)

    long_reports = [_side_metrics(long_score, long_return, long_mfe, long_mae, pct) for pct in top_pcts]
    short_reports = [_side_metrics(short_score, short_return, short_mfe, short_mae, pct) for pct in top_pcts]
    expected_long_reports = [
        _side_metrics(expected_long_return, long_return, long_mfe, long_mae, pct) for pct in top_pcts
    ]
    expected_short_reports = [
        _side_metrics(expected_short_return, short_return, short_mfe, short_mae, pct) for pct in top_pcts
    ]
    policy_grid = [
        _combined_policy_metrics(long_score, short_score, long_return, short_return, min_edge, min_gap)
        for min_edge in (0.55, 0.60, 0.65, 0.70)
        for min_gap in (0.05, 0.10, 0.15, 0.20)
    ]
    viable_policies = [
        row
        for row in policy_grid
        if row["entry_count"] >= 100
        and row["win_rate"] >= 0.52
        and row["profit_factor"] >= 1.10
        and row["avg_return_after_fees"] > 0.0
    ]
    ranked_groups = {
        "top_by_long_success_score": long_reports,
        "top_by_short_success_score": short_reports,
        "top_by_expected_long_return": expected_long_reports,
        "top_by_expected_short_return": expected_short_reports,
    }
    viable_ranked_slices = [
        {"group": group, **row}
        for group, rows in ranked_groups.items()
        for row in rows
        if row["count"] >= 100
        and row["win_rate"] >= 0.52
        and row["profit_factor"] >= 1.10
        and row["avg_return_after_fees"] > 0.0
    ]

    report = {
        "schema_version": "aegis_edge_decile_report_v1",
        "created_at": datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
        "dataset_path": str(dataset_path),
        "model_path": str(model_path),
        "test_pct": test_pct,
        "test_samples": int(len(x_test)),
        "test_start_timestamp": str(timestamps[0]) if len(timestamps) else None,
        "test_end_timestamp": str(timestamps[-1]) if len(timestamps) else None,
        "baseline": {
            "long_avg_return_after_fees": safe_float(np.mean(long_return)),
            "short_avg_return_after_fees": safe_float(np.mean(short_return)),
            "long_win_rate": safe_float(np.mean(long_return > 0.0)),
            "short_win_rate": safe_float(np.mean(short_return > 0.0)),
            "long_profit_factor": safe_float(profit_factor(long_return)),
            "short_profit_factor": safe_float(profit_factor(short_return)),
        },
        "top_by_long_success_score": long_reports,
        "top_by_short_success_score": short_reports,
        "top_by_expected_long_return": expected_long_reports,
        "top_by_expected_short_return": expected_short_reports,
        "policy_grid": policy_grid,
        "viable_policy_count": len(viable_policies),
        "viable_policies": viable_policies,
        "viable_ranked_slice_count": len(viable_ranked_slices),
        "viable_ranked_slices": viable_ranked_slices,
        "passes_minimum_gate": bool(viable_policies or viable_ranked_slices),
        "minimum_gate": {
            "entry_count": ">=100",
            "win_rate": ">=52%",
            "profit_factor": ">=1.10",
            "avg_return_after_fees": ">0",
            "scope": "policy_grid or ranked_slices",
        },
    }
    write_json(output_path, report)
    print(f"Decile report saved -> {output_path}")
    print(f"Test samples: {len(x_test):,}")
    for side, rows in (("LONG", long_reports), ("SHORT", short_reports)):
        top10 = rows[0]
        print(
            f"{side} top {top10['pct']:.0%}: count={top10['count']:,} "
            f"ret={top10['avg_return_after_fees']:.4%} wr={top10['win_rate']:.1%} "
            f"pf={top10['profit_factor']:.2f}"
        )
    print(f"Passes minimum gate: {report['passes_minimum_gate']}")
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="aegis_alpha/data/processed/edge_dataset_v030.npz")
    parser.add_argument("--model", default="aegis_alpha/models/edge/aegis_edge_model_v030.joblib")
    parser.add_argument("--output", default="aegis_alpha/logs/edge/edge_decile_report_v030.json")
    parser.add_argument("--test-pct", type=float, default=0.20)
    parser.add_argument("--top-pcts", type=float, nargs="+", default=[0.10, 0.05, 0.02])
    args = parser.parse_args()
    evaluate_edge_deciles(Path(args.dataset), Path(args.model), Path(args.output), args.test_pct, args.top_pcts)


if __name__ == "__main__":
    main()
