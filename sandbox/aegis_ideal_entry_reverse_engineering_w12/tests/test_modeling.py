from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from aegis_ideal_entry_reverse_engineering_w12.experiment import _negative_model_control
from aegis_ideal_entry_reverse_engineering_w12.modeling import FrozenCandidate
from aegis_ideal_entry_reverse_engineering_w12.modeling import ranked_metrics, temporal_negative_mask


def test_validation_thresholds_freeze_prospective_selection() -> None:
    labels = np.array([0, 0, 1, 1, 1], dtype=bool)
    scores = np.array([0.1, 0.2, 0.8, 0.9, 1.0])
    gross = np.array([-20, -20, 30, 30, 30], dtype=float)
    times = pd.Series(pd.date_range("2022-01-01", periods=5, freq="15min", tz="UTC"))
    symbols = pd.Series(["BTC"] * 5)
    _, thresholds, _ = ranked_metrics(labels, scores, gross, times, symbols, [1, 2, 5, 10])
    prospective_scores = np.array([thresholds[2] - 0.01, thresholds[2] + 0.01])
    metrics, frozen, _ = ranked_metrics(
        np.array([1, 0]), prospective_scores, np.array([30, -20]),
        times.iloc[:2], symbols.iloc[:2], [1, 2, 5, 10], thresholds,
    )
    assert frozen == thresholds
    assert metrics["top"]["2"]["selected"] == 1


def test_temporal_negative_exclusion() -> None:
    labels = pd.DataFrame({
        "decision_at": pd.date_range("2022-01-01", periods=7, freq="15min", tz="UTC"),
        "majority_ideal": [False, False, False, True, False, False, False],
    })
    keep = temporal_negative_mask(labels, 30)
    assert keep.tolist() == [True, False, False, True, False, False, True]


def test_ranked_metrics_are_deterministic() -> None:
    rng = np.random.default_rng(12)
    labels = rng.integers(0, 2, 100).astype(bool)
    scores = rng.random(100)
    gross = rng.normal(10, 20, 100)
    times = pd.Series(pd.date_range("2022-01-01", periods=100, freq="15min", tz="UTC"))
    symbols = pd.Series(np.where(np.arange(100) % 2, "BTC", "ETH"))
    first = ranked_metrics(labels, scores, gross, times, symbols, [1, 2, 5, 10])
    second = ranked_metrics(labels, scores, gross, times, symbols, [1, 2, 5, 10])
    assert first[0] == second[0]
    assert first[1] == second[1]


def _control_frame(start: str) -> pd.DataFrame:
    rows = []
    for index, decision_at in enumerate(pd.date_range(start, periods=80, freq="15min", tz="UTC")):
        ideal_side = "LONG" if (index // 7) % 2 == 0 else "SHORT"
        for side in ("LONG", "SHORT"):
            zone_best = index % 7 == 0 and side == ideal_side
            rows.append({
                "decision_at": decision_at,
                "symbol": "BTCUSDT",
                "side": side,
                "horizon_minutes": 15,
                "x": np.sin(index / 5),
                "zone_best": zone_best,
                "majority_ideal": zone_best,
                "entry_quality_score": 80.0 if zone_best else 10.0,
                "policy_gross_bps": 30.0 if zone_best else -20.0,
            })
    return pd.DataFrame(rows)


def _logistic() -> Pipeline:
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", LogisticRegression(class_weight="balanced", random_state=12, solver="liblinear")),
    ])


def test_negative_model_controls_are_deterministic_for_two_stage() -> None:
    discovery = _control_frame("2022-01-01")
    validation = _control_frame("2022-02-01")
    prospective = _control_frame("2022-03-01")
    candidate = FrozenCandidate(
        "OPPORTUNITY_THEN_SIDE_15M", "OPPORTUNITY_THEN_SIDE", "DYNAMIC", 15,
        (_logistic(), _logistic()), ("x",), {1: 0.9, 2: 0.8, 5: 0.7, 10: 0.6}, 0.0, 0.0,
    )
    config = {
        "seed": 12,
        "zones": {"negative_exclusion_minutes": 30},
        "negative_controls": {"random_feature_count": 8},
    }
    for random_features in (False, True):
        first = _negative_model_control(
            candidate, discovery, validation, prospective, config,
            random_features=random_features,
        )
        second = _negative_model_control(
            candidate, discovery, validation, prospective, config,
            random_features=random_features,
        )
        np.testing.assert_array_equal(first, second)
        assert len(first) > 0


def test_negative_model_controls_support_direct_classification() -> None:
    discovery = _control_frame("2022-01-01")
    validation = _control_frame("2022-02-01")
    prospective = _control_frame("2022-03-01")
    candidate = FrozenCandidate(
        "LOGISTIC_LONG_15M", "DIRECT_CLASSIFICATION", "LONG", 15,
        _logistic(), ("x",), {1: 0.9, 2: 0.8, 5: 0.7, 10: 0.6}, 0.0, 0.0,
    )
    config = {
        "seed": 12,
        "zones": {"negative_exclusion_minutes": 30},
        "negative_controls": {"random_feature_count": 8},
    }
    shuffled = _negative_model_control(
        candidate, discovery, validation, prospective, config,
        random_features=False,
    )
    random_features = _negative_model_control(
        candidate, discovery, validation, prospective, config,
        random_features=True,
    )
    assert len(shuffled) > 0
    assert len(random_features) > 0
