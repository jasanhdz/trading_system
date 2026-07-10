#!/usr/bin/env python3
from __future__ import annotations

import json
import pickle
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from aegis_alpha.tools.calibrate_trrm_operating_point_e21 import (
    LOCKBOX_START,
    ORIGINAL_THRESHOLD,
    TARGET,
    build_internal_folds,
    decision,
    evaluate_method,
    feature_hash,
    metrics_for_policy,
    opened_lockbox_diagnostic,
    policy_predictions,
    run_e21,
    select_policy,
)


def fixture(n: int = 900) -> pd.DataFrame:
    start = pd.Timestamp("2025-07-09T00:00:00Z")
    rows = []
    for i in range(n):
        horizon = [6, 12, 24][i % 3]
        score_driver = ((i % 30) / 30.0) + (0.18 if horizon == 24 else 0.0)
        y = int(score_driver > 0.78)
        row = {
            "id.symbol": ["BTCUSDT", "ETHUSDT", "ADAUSDT"][i % 3],
            "id.timestamp": str(start + pd.Timedelta(hours=12 * i)),
            "id.timeframe": "5m",
            "id.horizon": horizon,
            TARGET: y,
            "future_eval.future_mae_roe_proxy": 0.35 if y else 0.04,
            "reference.close": 100 + i,
        }
        for j in range(110):
            row[f"feature.f{j:03d}"] = ((i + j) % 37) / 37.0
        row["feature.atr_proxy_24"] = score_driver
        row["feature.ema_slope_6"] = 0.01
        row["feature.ema_slope_12"] = 0.02
        row["feature.ema_slope_24"] = 0.03
        row["feature.ema_slope_48"] = 0.04
        rows.append(row)
    return pd.DataFrame(rows)


class DummyImputer:
    def transform(self, x):
        return x.fillna(0.0)


class DummyModel:
    def predict_proba(self, x):
        s = x["feature.atr_proxy_24"].to_numpy(float)
        s = np.clip(s, 0, 1)
        return np.vstack([1 - s, s]).T


def make_artifacts(tmp: Path, df: pd.DataFrame) -> tuple[Path, Path, Path, Path]:
    dense = tmp / "dense.csv"
    strided = tmp / "strided.csv"
    dense.to_csv if False else None
    df.to_csv(dense, index=False)
    df.iloc[::3].reset_index(drop=True).to_csv(strided, index=False)
    features = [c for c in df.columns if c.startswith("feature.")][:111]
    if "feature.atr_proxy_24" not in features:
        features.append("feature.atr_proxy_24")
    features += ["horizon_6", "horizon_12", "horizon_24"]
    model_dir = tmp / "model"
    model_dir.mkdir()
    pipeline = {
        "model": DummyModel(),
        "imputer": DummyImputer(),
        "scaler": None,
        "calibrator": None,
        "features": features,
        "target": TARGET,
        "threshold": ORIGINAL_THRESHOLD,
    }
    with (model_dir / "selected_pipeline.pkl").open("wb") as f:
        pickle.dump(pipeline, f)
    e2 = {
        "target": {"name": TARGET},
        "split_checks": {"lockbox_used_for_selection": False},
        "selected_candidate": {
            "name": "GLOBAL_CAUSAL_PLUS_HORIZON:random_forest:raw",
            "model": "random_forest",
            "design": "GLOBAL_CAUSAL_PLUS_HORIZON",
            "feature_columns": features,
        },
    }
    report = tmp / "e2.json"
    report.write_text(json.dumps(e2))
    metadata = {
        "target": {"name": TARGET},
        "features": {"eligible_feature_columns": [c for c in features if c.startswith("feature.")]},
        "selected_candidate": e2["selected_candidate"],
        "decision": "TRRM_PROMISING_RESEARCH_ONLY",
    }
    (model_dir / "metadata.json").write_text(json.dumps(metadata))
    val = tmp / "val.csv"
    lock = tmp / "lock.csv"
    df.head(10).to_csv(val, index=False)
    df.tail(10).to_csv(lock, index=False)
    return dense, strided, report, model_dir


def test_metrics_formulas() -> None:
    y = np.array([1, 1, 0, 0, 0])
    s = np.array([0.9, 0.8, 0.7, 0.2, 0.1])
    thr = np.array([0.75] * 5)
    m = metrics_for_policy(fixture(5), np.arange(5), y, s, thr, 0.4)
    assert m["tail_capture_rate"] == 1.0
    assert round(m["precision_among_rejected"], 6) == 1.0
    assert m["residual_tail_rate"] == 0.0
    assert m["tail_risk_reduction"] == 1.0
    assert m["lift"] > 1.0


def test_quantiles_are_calibration_and_past_only() -> None:
    df = fixture()
    score = df["feature.atr_proxy_24"].to_numpy(float)
    folds = build_internal_folds(df, 120, 50, 5)
    fold = folds[0]
    s, thr, _ = policy_predictions(df, fold, score, 0.30, "STATIC_GLOBAL_QUANTILE", 50, 5)
    assert np.allclose(thr, np.quantile(score[fold.calibration_idx], 0.70))
    rs, rthr, _ = policy_predictions(df, fold, score, 0.30, "ROLLING_GLOBAL_QUANTILE_PAST_ONLY", 50, 5, 30)
    assert len(rthr) == len(fold.evaluation_idx)
    assert not np.isnan(rthr).any()
    es, ethr, _ = policy_predictions(df, fold, score, 0.30, "EXPANDING_GLOBAL_QUANTILE_PAST_ONLY", 50, 5)
    assert len(ethr) == len(fold.evaluation_idx)


def test_horizon_conditioned_and_fallback() -> None:
    df = fixture()
    score = df["feature.atr_proxy_24"].to_numpy(float)
    fold = build_internal_folds(df, 120, 50, 5)[0]
    _, thr, meta = policy_predictions(df, fold, score, 0.30, "HORIZON_CONDITIONED_QUANTILE", 10_000, 500)
    assert set(meta["thresholds_by_horizon"]) == {6, 12, 24}
    assert len(set(round(x, 10) for x in meta["thresholds_by_horizon"].values())) == 1


def test_selection_uses_prelockbox_only_and_ready_guards() -> None:
    df = fixture()
    score = df["feature.atr_proxy_24"].to_numpy(float)
    folds = build_internal_folds(df, 120, 50, 5)
    results = [evaluate_method(df, score, folds, "STATIC_GLOBAL_QUANTILE", 0.30, 50, 5)]
    selected = select_policy(results)
    assert selected is not None
    opened = opened_lockbox_diagnostic(df, score, selected, 0.30, 50, 5)
    selected_before = json.dumps(selected, sort_keys=True, default=str)
    assert opened["status"] == "OPENED_LOCKBOX_DIAGNOSTIC_ONLY"
    assert opened["not_used_for_selection"] is True
    assert json.dumps(selected, sort_keys=True, default=str) == selected_before
    bad = dict(selected)
    bad["aggregate"] = dict(selected["aggregate"])
    bad["aggregate"]["max_absolute_budget_error"] = 0.50
    assert decision(bad, opened)[0] != "TRRM_OPERATING_POLICY_READY_FOR_FORWARD_RESEARCH"


def test_run_fixture_outputs_research_only() -> None:
    df = fixture()
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        dense, strided, report, model_dir = make_artifacts(root, df)
        args = [
            "--dense-csv", str(dense),
            "--strided-csv", str(strided),
            "--e2-report-json", str(report),
            "--e2-model-dir", str(model_dir),
            "--validation-predictions", str(root / "val.csv"),
            "--opened-lockbox-predictions", str(root / "lock.csv"),
            "--output-dir", str(root),
            "--model-output-dir", str(root / "models"),
            "--minimum-calibration-rows", "50",
            "--minimum-positive-count", "5",
            "--write-artifacts", "true",
            "--write-report", "true",
        ]
        from aegis_alpha.tools.calibrate_trrm_operating_point_e21 import parse_args, run_e21

        payload = run_e21(parse_args(args))
        assert payload["target"] == TARGET
        assert payload["artifact_integrity"]["status"] == "OK"
        assert payload["lockbox_status"]["used_for_selection"] is False
        assert payload["decision"] in {
            "TRRM_OPERATING_POLICY_READY_FOR_FORWARD_RESEARCH",
            "TRRM_CALIBRATION_PROMISING",
            "STATIC_THRESHOLD_UNSTABLE_ADAPTIVE_POLICY_NEEDED",
            "HORIZON_CONDITIONING_REQUIRED",
            "CALIBRATION_NOT_STABLE",
            "RESEARCH_NOT_READY",
        }
        assert Path(payload["artifacts"]["markdown"]).exists()
        assert Path(payload["artifacts"]["json"]).exists()
        assert not (root / "active_manifest.json").exists()


if __name__ == "__main__":
    test_metrics_formulas()
    test_quantiles_are_calibration_and_past_only()
    test_horizon_conditioned_and_fallback()
    test_selection_uses_prelockbox_only_and_ready_guards()
    test_run_fixture_outputs_research_only()
    print("test_calibrate_trrm_operating_point_e21: OK")
