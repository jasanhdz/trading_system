import numpy as np
import pandas as pd
import pytest

from aegis.research.residual_regime_alpha_b2 import (
    B2ContractError,
    RegimeThresholds,
    add_barrier_outcomes,
    add_causal_residuals,
    assign_regimes,
    contract_hash,
)


def _rows(events=100):
    records = []
    for event in range(events):
        timestamp = event * 4 * 3_600_000
        btc_return = 0.001 * np.sin(event / 4)
        for symbol, beta, alpha in (("BTCUSDT", 1.0, 0.0), ("ADAUSDT", 1.5, 0.0002)):
            future = beta * btc_return + alpha
            for side in ("LONG", "SHORT"):
                records.append({
                    "timestamp_ms": timestamp, "symbol": symbol, "side": side,
                    "return_4h": beta * btc_return, "long_gross": future,
                    "short_gross": -future, "gross_return": future if side == "LONG" else -future,
                    "mae": 0.001, "mfe": 0.002,
                })
    return pd.DataFrame(records)


def test_residual_beta_is_past_only_and_removes_market_component():
    result = add_causal_residuals(_rows())
    latest = result.loc[result.timestamp_ms.eq(result.timestamp_ms.max())]
    assert latest.beta_btc.notna().all()
    assert latest.loc[latest.symbol.eq("ADAUSDT"), "residual_return"].abs().max() < 1e-8


def test_residual_contract_rejects_missing_columns():
    with pytest.raises(B2ContractError, match="COLUMNS_MISSING"):
        add_causal_residuals(pd.DataFrame({"timestamp_ms": [0]}))


def test_regime_assignment_uses_frozen_thresholds():
    events = pd.DataFrame({
        "btc_return_4h": [0.01, -0.01, 0.001],
        "btc_return_24h": [0.02, -0.02, 0.0],
        "median_volatility_24h": [0.03, 0.001, 0.01],
    })
    result = assign_regimes(events, RegimeThresholds(-0.01, 0.01, 0.005, 0.02))
    assert result.regime.tolist() == ["TREND_UP__EXPANSION", "TREND_DOWN__COMPRESSION", "RANGE__NORMAL"]


def test_barrier_same_minute_fails_closed():
    rows = pd.DataFrame({"timestamp_ms": [0], "symbol": ["ADAUSDT"], "side": ["LONG"]})
    minute = pd.DataFrame({
        "open_time": [900_000, 960_000], "open": [100.0, 100.0],
        "high": [101.0, 100.0], "low": [99.0, 100.0],
    })
    result = add_barrier_outcomes(rows, {"ADAUSDT": minute}, 1)
    assert result.iloc[0].barrier_outcome == "ADVERSE_FIRST_OR_SAME"
    assert not bool(result.iloc[0].favorable_first)


def test_feature_contract_hash_is_order_sensitive():
    assert contract_hash(("a", "b")) != contract_hash(("b", "a"))
