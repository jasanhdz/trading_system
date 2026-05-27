#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from aegis_alpha.tools.profile_failed_alpha_symbols import (
    bucketize_score,
    compute_alternative_outcomes,
    compute_family_scores,
    diagnose_symbol_alpha_profile,
    run,
)


def assert_true(value: bool, message: str) -> None:
    if not value:
        raise AssertionError(message)


def row(family: str, **updates: object) -> dict[str, object]:
    result: dict[str, object] = {
        "alpha_family": family,
        "bucket": "EXTREME",
        "count": 120,
        "horizon_candles": 12,
        "hit5": 0.25,
        "hit8": 0.20,
        "avg_trade_quality": 0.08,
        "net_quality_proxy_after_cost": 0.07,
        "danger_rate": 0.20,
        "reversal_hit5": 0.10,
        "saved_pnl_proxy": 0.02,
    }
    result.update(updates)
    return result


def market(count: int = 80) -> SimpleNamespace:
    close = np.linspace(100.0, 98.0, count, dtype=np.float32)
    high = close + 0.3
    low = close - 0.3
    timestamps = np.asarray([f"2026-05-01T00:{idx:02d}:00" for idx in range(count)])
    features = np.ones((count, 21), dtype=np.float32)
    features[:, 3] = np.linspace(1.0, 2.0, count)
    return SimpleNamespace(
        cfg=SimpleNamespace(risk=SimpleNamespace(total_fee=0.0004)),
        high=high,
        low=low,
        close=close,
        timestamps=timestamps,
        features=features,
    )


def test_bucketize_score() -> None:
    buckets = bucketize_score(np.asarray([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]))
    assert_true(set(buckets.tolist()) == {"LOW", "MEDIUM", "HIGH", "EXTREME"}, "four score buckets")


def test_diagnosis_families() -> None:
    breakdown = diagnose_symbol_alpha_profile("LINKUSDT", "SHORT", [row("breakdown_continuation_short")])
    assert_true(breakdown["best_alpha_family"] == "breakdown_continuation_short", "continuation is detected")
    assert_true(breakdown["confidence"] == "MODERATE", "discovery confidence is capped before lockbox")
    assert_true(breakdown["validation_status"] == "REQUIRES_LOCKBOX_VALIDATION", "discovery requires lockbox")
    fake = diagnose_symbol_alpha_profile(
        "LINKUSDT",
        "SHORT",
        [row("fake_breakdown_reversal", avg_trade_quality=-0.05, net_quality_proxy_after_cost=-0.06, hit5=0.10, reversal_hit5=0.30)],
    )
    assert_true(fake["best_alpha_family"] == "fake_breakdown_reversal", "reversal is detected")
    lower = diagnose_symbol_alpha_profile(
        "SOLUSDT",
        "SHORT",
        [row("momentum_burst", hit5=0.30, hit8=0.14)],
    )
    assert_true(lower["best_alpha_family"] == "momentum_burst_lower_target", "lower target is detected")
    avoid = diagnose_symbol_alpha_profile(
        "XRPUSDT",
        "SHORT",
        [
            row("breakdown_continuation_short", avg_trade_quality=-0.08, net_quality_proxy_after_cost=-0.09),
            row("avoid_only", avg_trade_quality=-0.05, net_quality_proxy_after_cost=-0.06, saved_pnl_proxy=0.06),
        ],
    )
    assert_true(avoid["best_alpha_family"] == "avoid_only_or_no_trade", "avoid only is detected")


def test_outcome_path_and_feature_scores_are_causal_and_finite() -> None:
    subject = market()
    steps = np.asarray([40, 45], dtype=np.int64)
    outcomes = compute_alternative_outcomes(subject, steps, "SHORT", 12, 0.001)
    assert_true("hit3_before_minus2" in outcomes, "alternate outcomes calculated")
    assert_true(np.all(np.isfinite(outcomes["trade_quality"])), "quality finite")
    feature_dataset = {
        "X": np.asarray([[0.0, 0.8], [0.3, 0.2]], dtype=np.float32),
        "feature_names": np.asarray(["short_breakdown_followthrough_3", "short_failed_breakdown_risk_12"]),
    }
    scores = compute_family_scores(feature_dataset, "SHORT")
    before = scores["breakdown_continuation_short"].copy()
    subject.close[60:] *= 5.0
    after = compute_family_scores(feature_dataset, "SHORT")["breakdown_continuation_short"]
    assert_true(np.allclose(before, after), "feature score consumes current feature rows only, not future candles")
    assert_true(np.all(np.isfinite(after)), "family scores finite")


def test_run_serializes_and_symbol_error_does_not_abort() -> None:
    with tempfile.TemporaryDirectory() as temp:
        args = argparse.Namespace(
            symbols="LINKUSDT,SOLUSDT",
            side="SHORT",
            lookback_days=30,
            horizons="12",
            out_dir=temp,
            fee_bps=8.0,
            slippage_bps=3.0,
            fast=True,
            include_longs=False,
            include_confirmed_controls=False,
            include_repaired_controls=False,
        )
        subject = market()
        dataset = {"step": np.asarray([40, 45], dtype=np.int64)}
        feature_dataset = {
            "X": np.asarray([[0.2], [0.4]], dtype=np.float32),
            "feature_names": np.asarray(["short_breakdown_followthrough_3"]),
            "feature_diagnostics": {"v3": {"cross_symbol_context_available": False}},
        }

        def load_market(_path: str, *, symbol_override: str) -> object:
            if symbol_override == "SOLUSDT":
                raise RuntimeError("missing symbol")
            return subject

        with patch("aegis_alpha.tools.profile_failed_alpha_symbols.load_signal_market", side_effect=load_market):
            with patch("aegis_alpha.tools.profile_failed_alpha_symbols.build_recent_dataset", return_value={"dataset": dataset}):
                with patch("aegis_alpha.tools.profile_failed_alpha_symbols.apply_feature_set", return_value=feature_dataset):
                    report = run(args)
        assert_true(len(report["profiles"]) == 1, "healthy symbol is retained")
        assert_true(len(report["errors"]) == 1, "bad symbol is recorded without abort")
        assert_true(Path(report["paths"]["json"]).exists(), "JSON report serializes")
        assert_true(Path(report["paths"]["summary_csv"]).exists(), "CSV report serializes")
        assert_true(report["models_trained"] is False, "no models trained")
        assert_true(report["model_artifacts_written"] is False, "no active or shadow models written")
        assert_true("LINKUSDT" in json.dumps(report), "result is JSON compatible")


def run_all() -> None:
    test_bucketize_score()
    test_diagnosis_families()
    test_outcome_path_and_feature_scores_are_causal_and_finite()
    test_run_serializes_and_symbol_error_does_not_abort()
    print("manual_profile_failed_alpha_symbols_tests_passed")


if __name__ == "__main__":
    run_all()
