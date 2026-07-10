#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from aegis_alpha.tools.train_trrm_honest_e2 import (
    TARGET,
    baseline_scores,
    budget_metrics,
    dedupe_dataset,
    decision,
    eligible_features,
    evaluate_baseline_set,
    make_split,
    parse_args,
    rejection_budget_table,
    run_e2,
    standard_metrics,
    target_values,
    threshold_for_budget,
)


def fixture(n: int = 720) -> pd.DataFrame:
    rows = []
    symbols = ["BTCUSDT", "ETHUSDT", "ADAUSDT", "SOLUSDT", "SUIUSDT", "XRPUSDT"]
    start = pd.Timestamp("2025-07-09T00:00:00Z")
    for i in range(n):
        horizon = [6, 12, 24][i % 3]
        ts = start + pd.Timedelta(minutes=720 * i)
        vol = (i % 17) / 17.0
        rebound = 1.0 if i % 13 == 0 else 0.0
        target = int(vol > 0.74 or rebound > 0)
        row = {
            "id.symbol": symbols[i % len(symbols)],
            "id.timestamp": str(ts),
            "id.timeframe": "5m",
            "id.horizon": horizon,
            "reference.close": 10.0 + i,
            "feature.close": 10.0 + i,
            TARGET: target,
            "target.bad_entry_v4": int(i % 11 == 0),
            "target.early_mae_v4": int(i % 13 == 0),
            "future_eval.future_mae_roe_proxy": 0.35 if target else 0.04,
            "label.clean_entry_v4": int(not target),
        }
        for j in range(120):
            row[f"feature.synthetic_{j:03d}"] = ((i + j) % 31) / 31.0
        row["feature.atr_proxy_24"] = vol
        row["feature.rolling_range_mean_24"] = vol * 0.8
        row["feature.rolling_range_std_24"] = vol * 0.2
        row["feature.rebound_risk_proxy"] = rebound
        row["feature.squeeze_risk_proxy_causal"] = 1.0 if i % 19 == 0 else 0.0
        row["feature.ema_slope_6"] = 0.01
        row["feature.ema_slope_12"] = 0.02
        row["feature.ema_slope_24"] = 0.03
        row["feature.ema_slope_48"] = 0.04
        rows.append(row)
    return pd.DataFrame(rows)


def test_features_exclude_leakage_and_allow_horizon_metadata() -> None:
    df = fixture()
    features, excluded, manual = eligible_features(df)
    assert TARGET not in features
    assert "feature.close" not in features
    assert "feature.ema_slope_6" in features
    assert any(r["column"] == "feature.close" for r in excluded)
    assert isinstance(manual, list)
    assert "id.symbol" not in features


def test_frozen_target_and_split_embargo() -> None:
    df = fixture()
    y = target_values(df, TARGET)
    split = make_split(df, embargo_minutes=120)
    assert y.sum() > 0
    assert len(split.train_idx) > 0
    assert len(split.validation_idx) > 0
    assert len(split.lockbox_idx) > 0
    assert split.overlap_ok is True
    assert split.rows_embargoed >= 0


def test_duplicate_handling() -> None:
    df = fixture(20)
    exact = pd.concat([df, df.iloc[[0]]], ignore_index=True)
    deduped, info = dedupe_dataset(exact)
    assert len(deduped) == len(df)
    assert info["exact_duplicates_removed"] == 1
    bad = exact.copy()
    bad.loc[len(bad) - 1, TARGET] = 1 - int(bad.loc[len(bad) - 1, TARGET])
    try:
        dedupe_dataset(bad)
    except ValueError as exc:
        assert "DATASET_INTEGRITY_ERROR" in str(exc)
    else:
        raise AssertionError("contradictory duplicate was not rejected")


def test_budget_metrics_are_correct() -> None:
    y = np.array([1, 1, 0, 0, 0, 0])
    score = np.array([0.9, 0.8, 0.7, 0.2, 0.1, 0.0])
    threshold = threshold_for_budget(score, 0.50)
    bm = budget_metrics(y, score, 0.50, threshold)
    assert bm["tail_capture_rate"] == 1.0
    assert round(bm["precision_among_rejected"], 6) == round(2 / 3, 6)
    assert bm["residual_tail_rate"] == 0.0
    assert bm["tail_risk_reduction"] == 1.0
    assert bm["lift_in_rejected_group"] > 1.0
    assert rejection_budget_table(y, score)["30pct"]["rows"] == 6
    assert standard_metrics(y, score, threshold)["false_negatives"] == 0


def test_oracle_and_reject_all_cannot_be_operational_candidates() -> None:
    df = fixture()
    y = target_values(df, TARGET)
    split = make_split(df, embargo_minutes=120)
    scores = baseline_scores(df, y, split)
    assert scores["diagnostic_oracle_upper_bound"]["eligible_for_selection"] is False
    assert scores["reject_all"]["eligible_for_selection"] is False
    evaluated = evaluate_baseline_set(scores, y, split)
    assert evaluated["reject_none"]["validation"]["standard"]["rejection_rate"] == 0.0
    assert evaluated["prevalence_baseline"]["validation"]["standard"]["rejection_rate"] == 0.0
    assert evaluated["reject_all"]["validation"]["standard"]["rejection_rate"] == 1.0


def test_ready_decision_blocks_excessive_rejection() -> None:
    payload = {
        "integrity": {"status": "OK"},
        "leakage_risk": False,
        "split_checks": {"overlap_ok": True},
        "selected_candidate": {"name": "model"},
        "lockbox": {
            "strided_budget_30": {
                "tail_capture_rate": 1.0,
                "tail_risk_reduction": 1.0,
                "rejection_rate": 0.90,
                "retained_rate": 0.10,
                "lift_in_rejected_group": 2.0,
                "residual_tail_rate": 0.0,
            },
            "strided_standard": {"prevalence": 0.05, "pr_auc": 0.20},
        },
        "comparison": {"best_causal_baseline_lockbox_budget_30": {"tail_capture_rate": 0.1, "residual_tail_rate": 0.04}},
        "walk_forward": [{"budget_30": {"lift_in_rejected_group": 2.0}}, {"budget_30": {"lift_in_rejected_group": 2.0}}],
        "ablation": {"status": "MULTI_SIGNAL_RISK_STRUCTURE_CONFIRMED"},
    }
    assert decision(payload)[0] != "TRRM_READY_FOR_PHASE_F_RETROSPECTIVE"


def test_run_e2_fixture_writes_research_artifacts() -> None:
    df = fixture()
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        dense = tmp_path / "dense.csv"
        strided = tmp_path / "strided.csv"
        d2_json = tmp_path / "d2.json"
        df.to_csv(dense, index=False)
        df.iloc[::6].reset_index(drop=True).to_csv(strided, index=False)
        d2_json.write_text(json.dumps({"run": "20260710T051035Z", "decision": "TAIL_TARGET_READY_FOR_E2"}))
        args = parse_args([
            "--dense-csv", str(dense),
            "--strided-csv", str(strided),
            "--d2-report-json", str(d2_json),
            "--output-dir", str(tmp_path),
            "--model-output-dir", str(tmp_path / "aegis_research_models" / "trrm_e2"),
            "--write-models", "false",
            "--max-train-rows", "120",
            "--seed", "7",
        ])
        payload = run_e2(args)
        assert payload["target"]["name"] == TARGET
        assert payload["split_checks"]["lockbox_used_for_selection"] is False
        assert payload["baselines"]["diagnostic_oracle_upper_bound"]["eligible_for_selection"] is False
        assert payload["decision"] in {
            "TRRM_READY_FOR_PHASE_F_RETROSPECTIVE",
            "TRRM_PROMISING_RESEARCH_ONLY",
            "HONEST_CAUSAL_BASELINE_NOT_BEATEN",
            "RESEARCH_NOT_READY",
            "LEAKAGE_RISK_TOO_HIGH",
        }
        assert Path(payload["artifacts"]["markdown"]).exists()
        assert Path(payload["artifacts"]["json"]).exists()
        assert not (tmp_path / "active_manifest.json").exists()


if __name__ == "__main__":
    test_features_exclude_leakage_and_allow_horizon_metadata()
    test_frozen_target_and_split_embargo()
    test_duplicate_handling()
    test_budget_metrics_are_correct()
    test_oracle_and_reject_all_cannot_be_operational_candidates()
    test_ready_decision_blocks_excessive_rejection()
    test_run_e2_fixture_writes_research_artifacts()
    print("test_train_trrm_honest_e2: OK")
