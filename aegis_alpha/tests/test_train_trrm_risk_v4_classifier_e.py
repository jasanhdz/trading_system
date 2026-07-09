#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pandas as pd

from aegis_alpha.tools.train_trrm_risk_v4_classifier_e import (
    baseline_simple_scores,
    build_targets,
    choose_best,
    fit_preprocessors,
    is_leakage_feature,
    make_split,
    run_training,
    select_features,
    threshold_search,
)


def rows(n: int = 180) -> list[dict]:
    out = []
    for i in range(n):
        tail = i % 4 == 0
        out.append({
            "id.timestamp": f"2026-07-{1+i//24:02d} {i%24:02d}:00:00",
            "id.symbol": "ADAUSDT" if i % 2 else "ETHUSDT",
            "id.horizon": "6" if i % 3 else "12",
            "id.timeframe": "5m",
            "feature.atr_proxy_24": 0.08 if tail else 0.01,
            "feature.rolling_range_mean_24": 0.07 if tail else 0.01,
            "feature.rolling_range_std_24": 0.05 if tail else 0.005,
            "feature.rebound_risk_proxy": 1 if tail else 0,
            "feature.squeeze_risk_proxy_causal": 1 if tail else 0,
            "feature.ret_1": -0.01 if tail else 0.002,
            "feature.ret_3": -0.03 if tail else 0.004,
            "feature.ret_6": -0.04 if tail else 0.003,
            "feature.close_vs_ema_12": -0.02 if tail else 0.001,
            "feature.close_position_in_range": 0.2 if tail else 0.7,
            "feature.volume_zscore_24": 2 if tail else 0,
            "feature.trend_strength_proxy_24": 3 if tail else 0.5,
            "feature.chop_proxy_12": 0.2 if tail else 0.7,
            "feature.lower_wick_pct": 0.5 if tail else 0.1,
            "feature.body_to_range": 0.7 if tail else 0.3,
            "feature.distance_to_rolling_low_24": -0.01 if tail else 0.05,
            "feature.distance_to_rolling_high_24": 0.10 if tail else 0.03,
            "feature.btc_ret_6": -0.02 if tail else 0.001,
            "feature.eth_ret_6": -0.02 if tail else 0.001,
            "target.tail_risk_v4": 1 if tail else 0,
            "target.bad_entry_v4": 1 if tail else 0,
            "target.early_mae_v4": 1 if tail else 0,
            "label.clean_entry_v4": 0 if tail else 1,
            "future_eval.future_mae_roe_proxy": 0.15 if tail else 0.02,
        })
    return out


def write_csv(path: Path) -> None:
    data = rows()
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(data[0].keys()))
        writer.writeheader()
        writer.writerows(data)


def test_feature_selection_only_feature_namespace() -> None:
    df = pd.DataFrame(rows())
    features, excluded, manual = select_features(df)
    assert features
    assert all(c.startswith("feature.") for c in features)
    assert not any(c.startswith(("target.", "label.", "future_eval.")) for c in features)
    assert excluded


def test_union_target() -> None:
    df = pd.DataFrame(rows())
    targets = build_targets(df)
    assert "union_tail_or_bad" in targets
    assert targets["union_tail_or_bad"].sum() >= targets["tail_risk_v4"].sum()


def test_temporal_split_order() -> None:
    df = pd.DataFrame(rows())
    split = make_split(df)
    assert split.reliable is True
    assert len(set(split.train_idx) & set(split.test_idx)) == 0


def test_train_only_preprocessing() -> None:
    df = pd.DataFrame(rows())
    features, _, _ = select_features(df)
    split = make_split(df)
    imp, scaler, x = fit_preprocessors(df[features], split.train_idx, scale=True)
    assert hasattr(imp, "medians")
    assert hasattr(scaler, "mean")
    assert len(x) == len(df)


def test_threshold_and_baselines() -> None:
    df = pd.DataFrame(rows())
    y = build_targets(df)["tail_risk_v4"].to_numpy()
    scores = baseline_simple_scores(df)
    search = threshold_search(y, scores)
    assert "recall_095" in search


def test_ema_slope_is_causal_not_leakage() -> None:
    leak, reason = is_leakage_feature("feature.ema_slope_12")
    assert leak is False, f"ema_slope wrongly flagged: {reason}"
    assert is_leakage_feature("feature.sl_price")[0] is True
    assert is_leakage_feature("target.tail_risk_v4")[0] is True


def test_choose_best_ignores_diagnostic_oracle_baseline() -> None:
    perfect = {"recall": 1.0, "precision": 0.99, "rejection_rate": 0.5, "f1": 0.99}
    weaker = {"recall": 0.96, "precision": 0.60, "rejection_rate": 0.70, "f1": 0.74}
    result = {
        "baselines": {
            "baseline_simple_label_derived": {"test": perfect},
            "causal_volatility_baseline": {"test": {"recall": 0.40, "precision": 0.55, "rejection_rate": 0.35, "f1": 0.46}},
            "reject_all": {"test": {"recall": 1.0, "precision": 0.48, "rejection_rate": 1.0, "f1": 0.65}},
        },
        "models": {"gradient_boosting": {"test": weaker}},
    }
    assert choose_best(result) == "gradient_boosting"


def test_split_embargo_drops_boundary_rows() -> None:
    df = pd.DataFrame(rows())
    split_no_embargo = make_split(df, embargo_minutes=0)
    split_embargo = make_split(df, embargo_minutes=120)
    assert split_embargo.embargo_minutes == 120
    assert len(split_embargo.val_idx) < len(split_no_embargo.val_idx)
    assert len(split_embargo.test_idx) < len(split_no_embargo.test_idx)
    assert split_embargo.embargo_dropped_val >= 1
    assert split_embargo.embargo_dropped_test >= 1
    assert len(set(split_embargo.train_idx) & set(split_embargo.val_idx)) == 0
    ts = pd.to_datetime(df["id.timestamp"], utc=True)
    train_max = ts.loc[split_embargo.train_idx].max()
    assert (ts.loc[split_embargo.val_idx] > train_max + pd.Timedelta(minutes=120)).all()


def test_script_runs_and_outputs() -> None:
    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "dataset.csv"
        write_csv(path)
        result = run_training(argparse.Namespace(input_csv=str(path), output_dir=td, target="all", max_rows=0, symbols="", horizons="", write_models="false", write_report="true", strict_causal="true"))
        assert result["decision"] in {"TRRM_READY_FOR_SHADOW_REVIEW", "TRRM_PROMISING_RESEARCH_ONLY", "BASELINE_NOT_BEATEN", "CAUSAL_BASELINE_NOT_BEATEN", "RESEARCH_NOT_READY", "DATASET_NOT_USABLE", "LEAKAGE_RISK_TOO_HIGH"}
        assert Path(result["outputs"]["md"]).exists()
        assert Path(result["outputs"]["json"]).exists()
        assert result["safety_confirmations"]["no_active_manifest"] is True
        assert result["safety_confirmations"]["no_yaml"] is True
        assert result["safety_confirmations"]["no_ts_touched"] is True
        assert result["model_research_path"] is None


if __name__ == "__main__":
    test_feature_selection_only_feature_namespace()
    test_union_target()
    test_temporal_split_order()
    test_train_only_preprocessing()
    test_threshold_and_baselines()
    test_ema_slope_is_causal_not_leakage()
    test_choose_best_ignores_diagnostic_oracle_baseline()
    test_split_embargo_drops_boundary_rows()
    test_script_runs_and_outputs()
    print("test_train_trrm_risk_v4_classifier_e: OK")
