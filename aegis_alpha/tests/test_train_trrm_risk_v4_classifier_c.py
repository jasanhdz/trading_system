#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pandas as pd

from aegis_alpha.tools.train_trrm_risk_v4_classifier_c import (
    build_feature_matrix,
    build_targets,
    make_walk_forward_split,
    baseline_scores,
    safe_metric_metrics,
    run_training,
)


FIELDS = [
    "timestamp",
    "symbol",
    "timeframe",
    "horizon",
    "close",
    "future_mfe_roe_proxy",
    "future_mae_roe_proxy",
    "net_quality_after_costs",
    "clean_entry_v4",
    "bad_entry_v4",
    "premium_allowed_v4",
    "management_dependent_v4",
    "no_trade_v4",
    "tail_risk_v4",
    "early_mae_v4",
    "squeeze_risk_proxy_v4",
    "qmae_target",
    "volatility_features",
    "trend_features",
    "wick_reclaim_proxies",
    "btc_eth_context",
]


def fixture_rows(n: int = 180) -> list[dict]:
    rows = []
    for i in range(n):
        tail = i % 4 == 0
        rows.append({
            "timestamp": f"2026-07-{1 + i // 24:02d} {i % 24:02d}:00:00",
            "symbol": "ADAUSDT" if i % 2 else "ETHUSDT",
            "timeframe": "5m",
            "horizon": "6" if i % 3 else "12",
            "close": str(1.0 + i * 0.001),
            "future_mfe_roe_proxy": "0.1",
            "future_mae_roe_proxy": "0.14" if tail else "0.02",
            "net_quality_after_costs": "-0.05" if tail else "0.05",
            "clean_entry_v4": "0" if tail else "1",
            "bad_entry_v4": "1" if tail else "0",
            "premium_allowed_v4": "0",
            "management_dependent_v4": "0",
            "no_trade_v4": "1" if tail else "0",
            "tail_risk_v4": "1" if tail else "0",
            "early_mae_v4": "1" if tail else "0",
            "squeeze_risk_proxy_v4": "0",
            "qmae_target": "0.14" if tail else "0.02",
            "volatility_features": '{"ret_std_20": 0.01}',
            "trend_features": '{"ret_12": -0.02}',
            "wick_reclaim_proxies": '{"lower_wick_close_pct": 0.002}',
            "btc_eth_context": "{}",
        })
    return rows


def write_fixture(path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(fixture_rows())


def test_no_leakage_columns_in_features() -> None:
    df = pd.DataFrame(fixture_rows())
    x, info = build_feature_matrix(df)
    assert "future_mae_roe_proxy" not in x.columns
    assert "tail_risk_v4" not in x.columns
    assert "qmae_target" not in x.columns
    assert info["excluded_by_leakage"]


def test_targets_built_correctly() -> None:
    df = pd.DataFrame(fixture_rows())
    targets = build_targets(df)
    assert {"target_tail_risk_v4", "target_bad_entry_v4", "target_union_tail_or_bad"} <= set(targets)
    assert targets["target_union_tail_or_bad"].sum() >= targets["target_tail_risk_v4"].sum()


def test_walk_forward_split_ordered() -> None:
    df = pd.DataFrame(fixture_rows())
    split = make_walk_forward_split(df)
    assert split.train_idx.max() < split.val_idx.max()
    assert len(set(split.train_idx) & set(split.test_idx)) == 0


def test_baseline_metrics() -> None:
    df = pd.DataFrame(fixture_rows())
    scores = baseline_scores(df)
    y = build_targets(df)["target_tail_risk_v4"].to_numpy()
    metrics = safe_metric_metrics(y, scores)
    assert metrics["recall"] > 0.9
    assert metrics["precision"] > 0.9


def test_script_runs_and_reports_decision() -> None:
    with tempfile.TemporaryDirectory() as td:
        dataset = Path(td) / "samples.csv"
        write_fixture(dataset)
        result = run_training(argparse.Namespace(dataset_csv=str(dataset), out_dir=td, model_dir=str(Path(td) / "models"), no_save_model=True))
        assert result["decision"] in {"TRRM_PROMISING_FOR_REVIEW", "BASELINE_NOT_BEATEN", "RESEARCH_NOT_READY", "DATASET_NOT_USABLE", "LEAKAGE_RISK_TOO_HIGH"}
        assert Path(result["outputs"]["md"]).exists()
        assert result["safety_confirmations"]["no_active_manifest"] is True
        assert result["safety_confirmations"]["no_yaml"] is True
        assert result["safety_confirmations"]["no_ts_touched"] is True


if __name__ == "__main__":
    test_no_leakage_columns_in_features()
    test_targets_built_correctly()
    test_walk_forward_split_ordered()
    test_baseline_metrics()
    test_script_runs_and_reports_decision()
    print("test_train_trrm_risk_v4_classifier_c: OK")
