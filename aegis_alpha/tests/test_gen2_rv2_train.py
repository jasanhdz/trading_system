#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import pandas as pd

import aegis_alpha.tools.gen2_rv2_train as rv2


def make_dense(tmp: Path, n: int = 3000, seed: int = 7) -> Path:
    rng = np.random.default_rng(seed)
    ts = pd.date_range("2025-01-01", periods=n, freq="30min")
    risk = rng.random(n)
    noise = rng.random(n)
    y = (risk + 0.3 * noise > 1.05).astype(int)
    mae = np.clip(0.05 + 0.5 * risk + 0.1 * rng.random(n), 0, 1.5)
    df = pd.DataFrame({
        "id.symbol": np.where(np.arange(n) % 2 == 0, "BTCUSDT", "ADAUSDT"),
        "id.timestamp": ts.astype(str),
        "id.horizon": np.tile([6, 12, 24], n)[:n],
        "target.tail_risk_roe_030": y,
        "future_eval.future_mae_roe_proxy": mae,
        "feature.risk_signal": risk + rng.normal(0, 0.05, n),
        "feature.noise_a": rng.random(n),
        "feature.noise_b": rng.random(n),
        "feature.atr_proxy_24": 0.01 + 0.02 * risk,
    })
    path = tmp / "dense.csv"
    df.to_csv(path, index=False)
    return path


def write_lockbox(tmp: Path, dev_end: str = "2025-02-20 23:59:59") -> Path:
    path = tmp / "GEN2_LOCKBOX_MANIFEST.json"
    path.write_text(json.dumps({
        "historical_development_period": {"end": dev_end},
        "semi_blind_historical_confirmation": {"start": "2025-02-21 00:00:00"},
        "allowed_query_count_per_candidate": 1,
        "current_query_count": 0,
        "query_log": [],
    }))
    return path


def small_candidates() -> dict:
    from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
    from sklearn.linear_model import LogisticRegression

    return {
        "random_forest": (True, lambda: RandomForestClassifier(n_estimators=30, max_depth=6, random_state=42, n_jobs=1)),
        "hist_gradient_boosting": (True, lambda: HistGradientBoostingClassifier(max_iter=40, random_state=42)),
        "logistic_reference": (False, lambda: LogisticRegression(max_iter=500, solver="liblinear")),
    }


def test_full_training_run() -> None:
    with tempfile.TemporaryDirectory() as t:
        tmp = Path(t)
        dense = make_dense(tmp)
        rv2.LOCKBOX_MANIFEST_PATH = write_lockbox(tmp)
        rv2.RV2_ROOT = tmp / "rv2"
        original = rv2.trrm_candidates
        rv2.trrm_candidates = small_candidates
        try:
            payload = rv2.run_training(rv2.parse_args(["--dataset-csv", str(dense)]))
        finally:
            rv2.trrm_candidates = original
        # development split respects lockbox boundary
        assert payload["development_rows"] < 3000
        assert payload["semi_blind_rows_not_touched_here"] > 0
        # folds are temporal, embargoed and non-empty
        for f in payload["folds"]:
            assert f["train_rows"] > 0 and f["eval_rows"] > 0
        # a tree model with real signal must win over the fixture
        assert payload["trrm_winner"] in {"random_forest", "hist_gradient_boosting"}
        agg = payload["trrm_results"][payload["trrm_winner"]]["aggregate"]
        assert agg["h1_all_folds_lift_ge_2x"] is True
        assert agg["worst_pr_auc"] > 2 * agg["mean_prevalence"]
        # logistic reference never eligible as winner
        assert payload["trrm_results"]["logistic_reference"]["aggregate"]["eligible_winner"] is False
        # QMAE conformal coverage computed on every fold
        for fold in payload["qmae_results"]["folds"]:
            assert 0.0 <= fold["q90"]["conformal_coverage"] <= 1.0
            assert "conformal_adjustment" in fold["q90"]
        # frozen candidate exists with hashes
        assert Path(payload["frozen_candidate"]["pickle"]).exists()
        assert len(payload["frozen_candidate"]["pickle_sha256"]) == 64
        assert payload["gen1_artifacts_reused"] is False


def test_selection_prefers_stability() -> None:
    results = {
        "lucky": {"aggregate": {"eligible_winner": True, "h1_all_folds_lift_ge_2x": True, "worst_pr_auc": 0.20, "mean_pr_auc": 0.50, "std_pr_auc": 0.20, "total_train_seconds": 1}},
        "stable": {"aggregate": {"eligible_winner": True, "h1_all_folds_lift_ge_2x": True, "worst_pr_auc": 0.38, "mean_pr_auc": 0.42, "std_pr_auc": 0.03, "total_train_seconds": 5}},
        "reference": {"aggregate": {"eligible_winner": False, "h1_all_folds_lift_ge_2x": True, "worst_pr_auc": 0.99, "mean_pr_auc": 0.99, "std_pr_auc": 0.0, "total_train_seconds": 1}},
    }
    assert rv2.select_winner(results) == "stable"


if __name__ == "__main__":
    test_selection_prefers_stability()
    test_full_training_run()
    print("test_gen2_rv2_train: OK")
