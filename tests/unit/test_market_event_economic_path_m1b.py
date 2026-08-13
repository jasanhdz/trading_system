import numpy as np
import pandas as pd
import pytest

from aegis.research.market_event_economic_path_m1b import (
    FEATURE_NAMES,
    M1BContractError,
    M1BPolicy,
    apply_policy,
    calibrate_probability,
    feature_row,
    predict_models,
    train_models,
)


def _feature_source():
    return {
        "ret_3": 0.001,
        "ret_12": -0.002,
        "ret_60": 0.003,
        "flow_1": 0.1,
        "flow_3": -0.2,
        "volume_ratio": 1.1,
        "compression": 0.4,
        "breakout_up": 0.002,
        "breakout_down": -0.001,
        "mark_spot_basis": 0.0002,
        "basis_change_15m": -0.0001,
        "basis_zscore_7d": 0.3,
        "funding_rate": 0.0001,
        "funding_age_hours": 2.0,
        "direction_score": 0.01,
        "realized_volatility_1h": 0.005,
        "liquidity_ratio_1h": 1.2,
        "btc_return_1h": -0.004,
        "cross_symbol_breadth_1h": 0.2,
        "utc_hour_sin": 0.5,
        "utc_hour_cos": -0.5,
        "weekday_sin": 0.25,
        "weekday_cos": 0.75,
    }


def _training_rows(count=240, seed=181001):
    random = np.random.default_rng(seed)
    features = random.normal(size=(count, len(FEATURE_NAMES)))
    rows = pd.DataFrame(features, columns=FEATURE_NAMES)
    signal = features[:, 0] - 0.5 * features[:, 3]
    rows["positive_protected_net"] = signal > 0
    rows["mae_fraction"] = np.abs(features[:, 1]) * 0.002
    rows["protected_net_return"] = signal * 0.001
    rows["timestamp_ms"] = np.arange(count, dtype=np.int64) * 60_000
    rows["symbol"] = "BTCUSDT"
    return rows


def test_feature_contract_is_side_aware_and_rejects_stale_funding():
    source = _feature_source()
    long = feature_row(source, "LONG")
    short = feature_row(source, "SHORT")
    assert len(long) == len(FEATURE_NAMES) == 23
    assert short[0] == -long[0]
    assert short[12] == -long[12]
    assert short[18] == -long[18]
    stale = {**source, "funding_age_hours": 12.01}
    with pytest.raises(M1BContractError, match="FUNDING_STALE"):
        feature_row(stale, "LONG")


def test_probability_cannot_be_used_before_calibration_partition_is_fitted():
    train = _training_rows()
    models = train_models(train)
    with pytest.raises(M1BContractError, match="CALIBRATOR_NOT_FITTED"):
        predict_models(models, train.head(10))
    calibration = _training_rows(seed=181002)
    calibrate_probability(models, calibration)
    predicted = predict_models(models, calibration.head(10))
    assert predicted["predicted_positive_probability"].between(0.0, 1.0).all()
    with pytest.raises(M1BContractError, match="CALIBRATOR_ALREADY_FITTED"):
        calibrate_probability(models, calibration)


def test_policy_keeps_one_deterministic_event_per_cross_section():
    rows = pd.DataFrame(
        {
            "timestamp_ms": [1, 1, 2],
            "symbol": ["BTCUSDT", "ETHUSDT", "ADAUSDT"],
            "predicted_positive_probability": [0.8, 0.8, 0.8],
            "predicted_mae_q90": [0.001, 0.001, 0.001],
            "predicted_net_utility": [0.01, 0.02, 0.03],
        }
    )
    policy = M1BPolicy(0.5, 0.01, 0.0, 0.0, 3)
    selected = apply_policy(rows, policy)
    assert selected["symbol"].tolist() == ["ETHUSDT", "ADAUSDT"]
