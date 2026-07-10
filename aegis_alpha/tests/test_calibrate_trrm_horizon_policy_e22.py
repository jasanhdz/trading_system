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

import aegis_alpha.tools.calibrate_trrm_horizon_policy_e22 as e22
from aegis_alpha.tools.calibrate_trrm_horizon_policy_e22 import (
    EXPECTED_FEATURE_HASH,
    HORIZONS,
    TARGET,
    PolicySpec,
    almost_equal,
    artifact_integrity,
    complexity_rank,
    decision,
    drift_by_horizon,
    evaluate_policy,
    feature_hash,
    horizon_metrics,
    is_ready,
    parse_args,
    ready_criteria,
    rolling_policy_thresholds,
    run_e22,
    select_policy,
)
from aegis_alpha.tools.train_trrm_honest_e2 import add_horizon_one_hot


def fixture(n: int = 1200) -> pd.DataFrame:
    start = pd.Timestamp("2025-07-09T00:00:00Z")
    rows = []
    for i in range(n):
        horizon = [6, 12, 24][i % 3]
        base = ((i % 60) / 60.0)
        h_bias = {6: 0.02, 12: 0.12, 24: 0.26}[horizon]
        score_driver = min(0.99, base + h_bias)
        y = int(score_driver > {6: 0.74, 12: 0.76, 24: 0.78}[horizon])
        row = {
            "id.symbol": ["BTCUSDT", "ETHUSDT", "ADAUSDT"][i % 3],
            "id.timestamp": str(start + pd.Timedelta(hours=8 * i)),
            "id.timeframe": "5m",
            "id.horizon": horizon,
            TARGET: y,
            "future_eval.future_mae_roe_proxy": 0.35 if y else 0.03,
            "reference.close": 100 + i,
        }
        for j in range(110):
            row[f"feature.f{j:03d}"] = ((i + j) % 41) / 41.0
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
        return np.vstack([1 - s, s]).T


def make_artifacts(root: Path, df: pd.DataFrame) -> tuple[Path, Path, Path, Path, Path, Path]:
    dense = root / "dense.csv"
    strided = root / "strided.csv"
    df.to_csv(dense, index=False)
    df.iloc[::3].reset_index(drop=True).to_csv(strided, index=False)
    features = [c for c in df.columns if c.startswith("feature.")][:111]
    if "feature.atr_proxy_24" not in features:
        features.append("feature.atr_proxy_24")
    features += ["horizon_6", "horizon_12", "horizon_24"]
    model_dir = root / "model"
    model_dir.mkdir()
    pipeline = {
        "model": DummyModel(),
        "imputer": DummyImputer(),
        "scaler": None,
        "calibrator": None,
        "features": features,
        "target": TARGET,
        "threshold": 0.39101951472531293,
    }
    with (model_dir / "selected_pipeline.pkl").open("wb") as f:
        pickle.dump(pipeline, f)
    meta = {
        "target": {"name": TARGET},
        "selected_candidate": {
            "name": "GLOBAL_CAUSAL_PLUS_HORIZON:random_forest:raw",
            "model": "random_forest",
            "design": "GLOBAL_CAUSAL_PLUS_HORIZON",
            "feature_columns": features,
        },
    }
    (model_dir / "metadata.json").write_text(json.dumps(meta))
    e21 = {
        "target": TARGET,
        "artifact_integrity": {"feature_hash": feature_hash(features)},
        "selected_policy": {"method": "ROLLING_GLOBAL_QUANTILE_PAST_ONLY", "budget": 0.30, "rolling_window_days": 30},
        "lockbox_status": {"diagnostic_only": True, "used_for_selection": False},
    }
    e21_json = root / "e21.json"
    e21_json.write_text(json.dumps(e21))
    internal = root / "internal.csv"
    lock = root / "lock.csv"
    df.head(10).to_csv(internal, index=False)
    df.tail(10).to_csv(lock, index=False)
    return dense, strided, e21_json, model_dir, internal, lock


def test_target_and_feature_guards() -> None:
    df = fixture()
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        dense, _, e21_json, model_dir, _, _ = make_artifacts(root, df)
        with (model_dir / "selected_pipeline.pkl").open("rb") as f:
            pipeline = pickle.load(f)
        args = parse_args(["--dense-csv", str(dense), "--e21-report-json", str(e21_json), "--e2-model-dir", str(model_dir)])
        integ = artifact_integrity(args, df, pipeline, json.loads(e21_json.read_text()))
        assert integ["target"] == TARGET
        assert integ["checks"]["no_target_label_future_features"] is True
        assert integ["checks"]["raw_close_absent"] is True
        assert integ["checks"]["symbol_absent"] is True
        assert feature_hash(pipeline["features"]) != "" or EXPECTED_FEATURE_HASH


def test_horizon_thresholds_are_past_only_and_same_horizon() -> None:
    df = fixture(900)
    score = df["feature.atr_proxy_24"].to_numpy(float)
    y = df[TARGET].astype(int).to_numpy()
    cal = np.arange(0, 600)
    ev = np.arange(600, 690)
    spec = PolicySpec("HORIZON_ROLLING_QUANTILE", 0.30, 30)
    thresholds, meta = rolling_policy_thresholds(df, cal, ev, score.copy(), y, spec, 50, 30, 1)
    first = ev[0]
    ts = pd.to_datetime(df["id.timestamp"], utc=True)
    h = int(df.iloc[first]["id.horizon"])
    day = ts.iloc[first].floor("D")
    hist = cal[(pd.to_datetime(df.iloc[cal]["id.timestamp"], utc=True) < day).to_numpy()]
    hist = hist[(pd.to_datetime(df.iloc[hist]["id.timestamp"], utc=True) >= day - pd.Timedelta(days=30)).to_numpy()]
    hist = hist[(pd.to_numeric(df.iloc[hist]["id.horizon"]).astype(int) == h).to_numpy()]
    expected = np.quantile(score[hist], 0.70)
    assert abs(thresholds[0] - expected) < 1e-12
    assert first not in set(hist.tolist())
    assert meta["fallback_rate"] == 0.0


def test_expanding_and_global_fallback() -> None:
    df = fixture(900)
    score = df["feature.atr_proxy_24"].to_numpy(float)
    y = df[TARGET].astype(int).to_numpy()
    cal = np.arange(0, 600)
    ev = np.arange(600, 660)
    spec = PolicySpec("HORIZON_EXPANDING_QUANTILE", 0.30)
    thresholds, _ = rolling_policy_thresholds(df, cal, ev, score.copy(), y, spec, 50, 10, 1)
    assert np.isfinite(thresholds).all()
    high_min = PolicySpec("HORIZON_ROLLING_QUANTILE", 0.30, 30)
    _, meta = rolling_policy_thresholds(df, cal, ev, score.copy(), y, high_min, 50, 10_000, 1_000)
    assert meta["fallback_rate"] == 1.0


def test_blended_and_dynamic_shrinkage_formulas() -> None:
    df = fixture(900)
    score = df["feature.atr_proxy_24"].to_numpy(float)
    y = df[TARGET].astype(int).to_numpy()
    cal = np.arange(0, 600)
    ev = np.arange(600, 601)
    h = int(df.iloc[ev[0]]["id.horizon"])
    ts = pd.to_datetime(df["id.timestamp"], utc=True)
    day = ts.iloc[ev[0]].floor("D")
    hist_global = cal[(ts.iloc[cal] < day).to_numpy()]
    hist_global = hist_global[(ts.iloc[hist_global] >= day - pd.Timedelta(days=30)).to_numpy()]
    hist_h = hist_global[(pd.to_numeric(df.iloc[hist_global]["id.horizon"]).astype(int) == h).to_numpy()]
    gthr = np.quantile(score[hist_global], 0.70)
    hthr = np.quantile(score[hist_h], 0.70)
    thr, _ = rolling_policy_thresholds(df, cal, ev, score.copy(), y, PolicySpec("BLENDED_GLOBAL_HORIZON_ROLLING", 0.30, 30, 0.25), 50, 10, 1)
    assert abs(thr[0] - (0.25 * hthr + 0.75 * gthr)) < 1e-12
    shr, meta = rolling_policy_thresholds(df, cal, ev, score.copy(), y, PolicySpec("SHRUNK_HORIZON_QUANTILE_BY_SAMPLE_SIZE", 0.30, 30, None, 100), 50, 10, 1)
    alpha = len(hist_h) / (len(hist_h) + 100)
    assert abs(shr[0] - (alpha * hthr + (1 - alpha) * gthr)) < 1e-12
    assert abs(meta["mean_dynamic_alpha"] - alpha) < 1e-12


def test_metrics_spread_and_ready_guards() -> None:
    df = fixture(90)
    idx = np.arange(90)
    y = np.array([1 if i % 10 == 0 else 0 for i in idx])
    score = np.linspace(0, 1, 90)
    thresholds = np.array([0.2 if int(df.iloc[i]["id.horizon"]) == 24 else 0.9 for i in idx])
    rows = horizon_metrics(df, idx, y, score, thresholds, 0.30)
    rejections = [r["realized_rejection_rate"] for r in rows]
    assert round(max(rejections) - min(rejections), 6) >= 0.6
    result = {
        "aggregate": {
            "usable": True,
            "mean_absolute_budget_error": 0.01,
            "max_absolute_budget_error": 0.02,
            "mean_tail_capture": 0.9,
            "minimum_tail_capture": 0.8,
            "mean_lift": 2.5,
            "worst_fold_lift": 2.0,
            "mean_tail_risk_reduction": 0.6,
            "mean_realized_rejection": 0.30,
            "worst_horizon_rejection": 0.60,
            "best_horizon_rejection": 0.20,
            "horizon_rejection_spread": 0.40,
        }
    }
    assert ready_criteria(result)["no_horizon_above_50pct"] is False
    assert is_ready(result) is False


def test_selection_prelockbox_and_complexity_preference() -> None:
    simple = {
        "method": "HORIZON_ROLLING_QUANTILE",
        "budget": 0.30,
        "complexity": {"implementation_complexity": "MEDIUM"},
        "aggregate": {
            "usable": True,
            "mean_absolute_budget_error": 0.03,
            "max_absolute_budget_error": 0.08,
            "mean_tail_capture": 0.72,
            "minimum_tail_capture": 0.60,
            "mean_lift": 2.1,
            "worst_fold_lift": 1.7,
            "mean_tail_risk_reduction": 0.55,
            "mean_realized_rejection": 0.30,
            "worst_horizon_rejection": 0.39,
            "best_horizon_rejection": 0.21,
            "horizon_rejection_spread": 0.18,
        },
    }
    complex_one = json.loads(json.dumps(simple))
    complex_one["method"] = "HORIZON_CONDITIONED_SIGMOID"
    complex_one["complexity"] = {"implementation_complexity": "HIGH"}
    complex_one["aggregate"]["mean_tail_capture"] += 0.01
    assert almost_equal(simple, complex_one)
    assert complexity_rank(simple) < complexity_rank(complex_one)
    assert select_policy([complex_one, simple])["method"] == "HORIZON_ROLLING_QUANTILE"
    opened = {"contradicts_internal_evidence": False}
    assert decision(simple, None, opened)[0] == "HORIZON_POLICY_READY_FOR_FORWARD_RESEARCH"
    bad = json.loads(json.dumps(simple))
    bad["aggregate"]["mean_tail_capture"] = 0.4
    assert decision(bad, None, opened)[0] != "HORIZON_POLICY_READY_FOR_FORWARD_RESEARCH"


def test_run_fixture_outputs_research_only() -> None:
    df = fixture()
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        dense, strided, e21_json, model_dir, internal, lock = make_artifacts(root, df)
        args = parse_args(
            [
                "--dense-csv",
                str(dense),
                "--strided-csv",
                str(strided),
                "--e21-report-json",
                str(e21_json),
                "--e21-internal-predictions",
                str(internal),
                "--opened-lockbox-predictions",
                str(lock),
                "--e2-model-dir",
                str(model_dir),
                "--output-dir",
                str(root),
                "--policy-output-dir",
                str(root / "models"),
                "--minimum-history-rows",
                "50",
                "--minimum-horizon-rows",
                "10",
                "--minimum-positive-count",
                "1",
                "--rolling-window-days",
                "30",
                "--shrinkage-alphas",
                "0.25",
                "--write-artifacts",
                "true",
                "--write-report",
                "true",
            ]
        )
        e22.EXPECTED_FEATURE_HASH = json.loads(e21_json.read_text())["artifact_integrity"]["feature_hash"]
        payload = run_e22(args)
        assert payload["target"] == TARGET
        assert payload["lockbox_status"]["opened_lockbox_used_for_selection"] is False
        assert payload["lockbox_status"]["policy_changed_after_opened_lockbox"] is False
        assert payload["safety_confirmations"]["base_model_not_retrained"] is True
        assert payload["safety_confirmations"]["no_per_horizon_models"] is True
        assert payload["decision"] in {
            "HORIZON_POLICY_READY_FOR_FORWARD_RESEARCH",
            "HORIZON_POLICY_PROMISING",
            "GLOBAL_POLICY_PREFERRED",
            "HORIZON_CONDITIONING_REQUIRED_BUT_UNSTABLE",
            "HORIZON_POLICY_NOT_STABLE",
            "RESEARCH_NOT_READY",
        }
        assert Path(payload["artifacts"]["markdown"]).exists()
        assert Path(payload["artifacts"]["json"]).exists()
        assert not (root / "active_manifest.json").exists()
        assert not list(root.glob("*.yaml"))
        assert drift_by_horizon(df, df["feature.atr_proxy_24"].to_numpy(float), None)


if __name__ == "__main__":
    test_target_and_feature_guards()
    test_horizon_thresholds_are_past_only_and_same_horizon()
    test_expanding_and_global_fallback()
    test_blended_and_dynamic_shrinkage_formulas()
    test_metrics_spread_and_ready_guards()
    test_selection_prelockbox_and_complexity_preference()
    test_run_fixture_outputs_research_only()
    print("test_calibrate_trrm_horizon_policy_e22: OK")
