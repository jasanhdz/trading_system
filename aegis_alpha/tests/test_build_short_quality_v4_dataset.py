#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

import numpy as np

import sys

sys.path.append(str(Path(__file__).resolve().parents[2]))

import aegis_alpha.tools.build_short_quality_v4_dataset as builder  # noqa: E402


def synthetic_market(symbol: str = "ADAUSDT", n: int = 720) -> builder.MarketFrame:
    ts = np.asarray([f"2026-06-01T00:{i % 60:02d}:00" for i in range(n)], dtype=str)
    base = 100.0 - np.linspace(0.0, 3.0, n) + np.sin(np.arange(n) / 8.0) * 0.15
    close = base.astype(np.float64)
    open_ = np.concatenate((close[:1], close[:-1]))
    high = np.maximum(open_, close) + 0.05
    low = np.minimum(open_, close) - 0.35
    volume = np.ones(n, dtype=np.float64)
    return builder.MarketFrame(symbol=symbol, timestamps=ts, open=open_, high=high, low=low, close=close, volume=volume)


def test_classification_promising() -> None:
    rows = []
    for i in range(1000):
        clean = i < 50
        rows.append({
            "net_quality_after_costs": 0.20 if clean else 0.02,
            "mfe_roe_proxy": 0.14 if clean else 0.04,
            "mae_roe_proxy": 0.025 if clean else 0.08,
            "mfe_mae_ratio": 5.0 if clean else 0.5,
            "time_to_mfe": 2,
            "short_clean_entry_v4": int(clean),
            "short_bad_entry_v4": int(not clean),
            "short_premium_allowed_v4": int(i < 20),
            "short_management_dependent_v4": 0,
            "short_no_trade_v4": int(not clean),
        })
    summary = builder.label_quality_rows(rows, "ADAUSDT", 12)
    assert summary["classification"] == "V4_LABEL_PROMISING"


def test_classification_too_strict_and_too_loose() -> None:
    strict_rows = [{"short_clean_entry_v4": 0, "short_bad_entry_v4": 1, "short_premium_allowed_v4": 0, "short_management_dependent_v4": 0, "short_no_trade_v4": 1, "net_quality_after_costs": 0.0, "mfe_roe_proxy": 0.01, "mae_roe_proxy": 0.02, "mfe_mae_ratio": 0.5} for _ in range(1000)]
    loose_rows = []
    for i in range(1000):
        loose_rows.append({"short_clean_entry_v4": int(i < 250), "short_bad_entry_v4": 0, "short_premium_allowed_v4": int(i < 80), "short_management_dependent_v4": 0, "short_no_trade_v4": 0, "net_quality_after_costs": 0.1, "mfe_roe_proxy": 0.12, "mae_roe_proxy": 0.02, "mfe_mae_ratio": 6.0})
    assert builder.label_quality_rows(strict_rows, "ADAUSDT", 12)["classification"] == "V4_LABEL_TOO_STRICT"
    assert builder.label_quality_rows(loose_rows, "ADAUSDT", 12)["classification"] == "V4_LABEL_TOO_LOOSE"


def test_live_overlap_detects_big_loss_blocked() -> None:
    market = synthetic_market()
    labels = [[{"entry_index": 10, "short_clean_entry_v4": 0, "short_bad_entry_v4": 1, "short_premium_allowed_v4": 0, "short_management_dependent_v4": 1}]]
    trade = {
        "trade_id": "t1",
        "symbol": "ADAUSDT",
        "opened_at": "2026-06-01T00:10:00Z",
        "net_pnl_estimated": -2.0,
        "winner": False,
        "bucket": "premium",
    }
    rows = builder.live_overlap_rows([trade], {("ADAUSDT", 12): labels[0]}, {"ADAUSDT": market})
    assert rows[0]["v4_would_block"] is True
    assert rows[0]["premium_loser_blocked"] is True


def test_random_baseline_no_fails_and_serializes() -> None:
    rows = [{"net_quality_after_costs": i / 1000, "mfe_roe_proxy": 0.1, "mae_roe_proxy": 0.02, "short_clean_entry_v4": int(i > 50)} for i in range(100)]
    out = builder.random_baseline_for_rows(rows, n=20)
    assert out["random_count"] == 20
    json.dumps(out, default=str)


def test_active_path_rejected_for_matrix_outputs() -> None:
    try:
        builder.assert_research_only_output(Path("/tmp/models/turbo/ABC/active/matrix.npy"))
    except ValueError:
        return
    raise AssertionError("active path should be rejected")


def test_builder_generates_labels_for_synthetic_ada_and_no_matrix(monkeypatch=None) -> None:
    original_load_market = builder.load_market
    original_feature_diag = builder.build_feature_diagnostics
    original_live = builder.load_live_quality
    try:
        builder.load_market = lambda symbol, lookback_days, db_path=builder.DB_PATH, end=None: synthetic_market(symbol, 720)
        builder.build_feature_diagnostics = lambda market, steps, context: {"feature_count": 4, "feature_schema_hash": "abc", "feature_set": "combined_v3"}
        builder.load_live_quality = lambda args: []
        with tempfile.TemporaryDirectory() as td:
            args = argparse.Namespace(
                symbols="ADAUSDT",
                lookback_days=30,
                horizons="12",
                out_dir=td,
                no_save_matrix=True,
                save_summary=True,
                include_phase_o_comparison=True,
                include_live_trade_overlap=True,
                include_random_baseline=True,
                fast=True,
            )
            payload = builder.build_dataset(args)
            paths = builder.write_outputs(payload, Path(td))
            assert payload["summary"]
            assert payload["summary"][0]["symbol"] == "ADAUSDT"
            assert not list(Path(td).glob("*.npy"))
            assert Path(paths["json"]).exists()
    finally:
        builder.load_market = original_load_market
        builder.build_feature_diagnostics = original_feature_diag
        builder.load_live_quality = original_live


if __name__ == "__main__":
    test_classification_promising()
    test_classification_too_strict_and_too_loose()
    test_live_overlap_detects_big_loss_blocked()
    test_random_baseline_no_fails_and_serializes()
    test_active_path_rejected_for_matrix_outputs()
    test_builder_generates_labels_for_synthetic_ada_and_no_matrix()
    print("build_short_quality_v4_dataset tests passed")
