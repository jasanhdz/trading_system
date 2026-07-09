#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from aegis_alpha.tools.audit_risk_v4_qmae_dataset_b import run_audit, simple_separability, threshold_sensitivity


FIELDS = [
    "timestamp",
    "symbol",
    "timeframe",
    "horizon",
    "close",
    "future_mfe_roe_proxy",
    "future_mae_roe_proxy",
    "time_to_mfe",
    "time_to_mae",
    "mfe_before_mae",
    "mfe_mae_ratio",
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
    "q95_mae_target",
    "volatility_features",
    "trend_features",
    "wick_reclaim_proxies",
    "btc_eth_context",
]


def make_rows(n: int = 1200) -> list[dict]:
    rows = []
    for i in range(n):
        tail = i % 5 == 0
        mae = 0.12 if tail else 0.025
        rows.append({
            "timestamp": f"2026-07-09 00:{i % 60:02d}:00",
            "symbol": "ADAUSDT",
            "timeframe": "5m",
            "horizon": "6",
            "close": "1.0",
            "future_mfe_roe_proxy": "0.08",
            "future_mae_roe_proxy": str(mae),
            "time_to_mfe": "4",
            "time_to_mae": "2" if tail else "5",
            "mfe_before_mae": "False" if tail else "True",
            "mfe_mae_ratio": "0.7" if tail else "2.0",
            "net_quality_after_costs": "-0.04" if tail else "0.04",
            "clean_entry_v4": "0" if tail else "1",
            "bad_entry_v4": "1" if tail else "0",
            "premium_allowed_v4": "0",
            "management_dependent_v4": "0",
            "no_trade_v4": "1" if tail else "0",
            "tail_risk_v4": "1" if tail else "0",
            "early_mae_v4": "1" if tail else "0",
            "squeeze_risk_proxy_v4": "0",
            "qmae_target": str(mae),
            "q95_mae_target": "0.12",
            "volatility_features": "{}",
            "trend_features": "{}",
            "wick_reclaim_proxies": "{}",
            "btc_eth_context": "{}",
        })
    return rows


def write_fixture(path: Path) -> None:
    rows = make_rows()
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def test_threshold_sensitivity() -> None:
    rows = make_rows()
    sens = threshold_sensitivity(rows)
    assert sens[0]["rows"] >= sens[-1]["rows"]


def test_simple_separability() -> None:
    sep = simple_separability(make_rows())
    assert sep["simple_rule_recall"] > 0.9
    assert sep["signal_gap_mae_mean"] > 0


def test_run_audit_serializes() -> None:
    with tempfile.TemporaryDirectory() as td:
        csv_path = Path(td) / "samples.csv"
        write_fixture(csv_path)
        result = run_audit(argparse.Namespace(samples_csv=str(csv_path), out_dir=td))
        assert result["status"] in {"CONDITIONAL_GO", "GO", "NO-GO"}
        assert Path(result["outputs"]["md"]).exists()
        assert result["safety_confirmations"]["no_model_training"] is True


if __name__ == "__main__":
    test_threshold_sensitivity()
    test_simple_separability()
    test_run_audit_serializes()
    print("test_audit_risk_v4_qmae_dataset_b: OK")
