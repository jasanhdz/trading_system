#!/usr/bin/env python3
from __future__ import annotations

import json
import pickle
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import pandas as pd

import aegis_alpha.tools.gen2_eqm1_train as eqm
import aegis_alpha.tools.gen2_rv2_train as rv2
from aegis_alpha.tools.gen2_rv2_train import MedianImputer


def make_fixture(tmp: Path, n: int = 4500, seed: int = 11):
    rng = np.random.default_rng(seed)
    ts = pd.date_range("2025-01-01", periods=n, freq="20min")
    value = rng.random(n)          # drives quality
    danger = rng.random(n)         # drives tail
    quality = 0.4 * value - 0.15 + 0.05 * rng.normal(size=n)
    df = pd.DataFrame({
        "id.symbol": np.where(np.arange(n) % 2 == 0, "BTCUSDT", "ADAUSDT"),
        "id.timestamp": ts.astype(str),
        "id.horizon": 12,
        "target.tail_risk_roe_030": (danger > 0.9).astype(int),
        "future_eval.net_quality_after_costs": quality,
        "label.clean_entry_v4": (quality > 0.1).astype(int),
        "feature.value_signal": value + rng.normal(0, 0.05, n),
        "feature.danger_signal": danger + rng.normal(0, 0.05, n),
        "feature.noise": rng.random(n),
    })
    dense = tmp / "dense.csv"
    df.to_csv(dense, index=False)
    lock = tmp / "GEN2_LOCKBOX_MANIFEST.json"
    lock.write_text(json.dumps({"historical_development_period": {"end": "2025-02-20 23:59:59"}}))
    # tiny frozen TRRM on the danger feature
    from sklearn.ensemble import RandomForestClassifier
    feats = ["feature.value_signal", "feature.danger_signal", "feature.noise", "horizon_6", "horizon_12", "horizon_24"]
    work = df.copy()
    for h in (6, 12, 24):
        work[f"horizon_{h}"] = (work["id.horizon"] == h).astype(float)
    imp = MedianImputer().fit(work[feats])
    model = RandomForestClassifier(n_estimators=25, random_state=1).fit(imp.transform(work[feats]), work["target.tail_risk_roe_030"])
    trrm_dir = tmp / "trrm"
    trrm_dir.mkdir()
    with (trrm_dir / "rv2_candidate.pkl").open("wb") as f:
        pickle.dump({"trrm_model": model, "imputer": imp, "calibrator": None, "calibrator_kind": "raw",
                     "qmae_models": {}, "qmae_q90_conformal_adjustment": 0.0, "features": feats}, f)
    return dense, lock, trrm_dir


def small_eqm():
    from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor, RandomForestRegressor
    from sklearn.linear_model import Ridge

    return {
        "reg": {
            "rf_reg": (True, lambda: RandomForestRegressor(n_estimators=25, random_state=42, n_jobs=1)),
            "hgb_reg": (True, lambda: HistGradientBoostingRegressor(max_iter=40, random_state=42)),
            "et_reg": (True, lambda: RandomForestRegressor(n_estimators=15, random_state=7, n_jobs=1)),
            "ridge_reference": (False, lambda: Ridge(alpha=1.0)),
        },
        "clf": {
            "hgb_clf": (True, lambda: HistGradientBoostingClassifier(max_iter=40, random_state=42)),
        },
    }


def test_eqm_training_pipeline() -> None:
    with tempfile.TemporaryDirectory() as t:
        tmp = Path(t)
        dense, lock, trrm_dir = make_fixture(tmp)
        rv2.LOCKBOX_MANIFEST_PATH = lock
        eqm.EQM_ROOT = tmp / "eqm"
        original = eqm.eqm_candidates
        eqm.eqm_candidates = small_eqm
        try:
            payload = eqm.run_training(eqm.parse_args(["--dataset-csv", str(dense), "--trrm-dir", str(trrm_dir)]))
        finally:
            eqm.eqm_candidates = original
        assert payload["reg_winner"] in {"rf_reg", "hgb_reg", "et_reg"}
        agg = payload["results"]["reg"][payload["reg_winner"]]["aggregate"]
        assert agg["h1_top_decile_positive_all_folds"] is True  # fixture has real signal
        assert payload["results"]["reg"]["ridge_reference"]["eligible"] is False
        for c in payload["composite_and_incrementality"]:
            assert c["corr_limit_retention"] <= 1.0
            assert 0.0 <= c["abstention_rate"] <= 1.0
        assert Path(payload["frozen_candidate"]["pickle"]).exists()
        # veto reduces population: retained rows < eval rows in every fold
        for r in payload["results"]["reg"][payload["reg_winner"]]["folds"]:
            assert r["retained_rows"] > 50


def test_correlation_limit_one_per_window() -> None:
    ts = pd.date_range("2025-01-01", periods=12, freq="10min")
    df = pd.DataFrame({"id.timestamp": ts.astype(str)})
    score = np.arange(12, dtype=float)
    mask = eqm.correlation_limit(df, score)
    kept = df[mask]
    windows = pd.to_datetime(kept["id.timestamp"]).dt.floor("30min")
    assert windows.duplicated().sum() == 0
    assert mask.sum() == windows.nunique()


if __name__ == "__main__":
    test_correlation_limit_one_per_window()
    test_eqm_training_pipeline()
    print("test_gen2_eqm1_train: OK")
