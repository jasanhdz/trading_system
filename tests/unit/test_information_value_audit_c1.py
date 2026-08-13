import numpy as np
import pandas as pd
import pytest

from aegis.research.information_value_audit_c1 import (
    C1ContractError,
    FAMILIES,
    canonical_features,
    contract_hash,
    feature_names,
)


def _source(side="LONG"):
    return pd.DataFrame({
        "side": [side], "return_1h": [.01], "return_4h": [.02], "return_24h": [.03],
        "realized_volatility_4h": [.1], "realized_volatility_24h": [.2],
        "distance_sma_4h": [.04], "distance_sma_24h": [.05], "extension_z_24h": [1.],
        "breakout_acceptance_long": [.2], "breakout_acceptance_short": [-.2],
        "wick_rejection_long": [.3], "wick_rejection_short": [-.3],
        "taker_flow_15m": [.4], "taker_flow_1h": [.5], "prior_taker_flow_1h": [.1],
        "volume_persistence_1h": [2.], "mark_spot_basis": [.001], "basis_z_7d": [1.2],
        "basis_convergence_1h": [.0002], "funding_rate": [.0001], "funding_age_hours": [3.],
        "btc_state_return": [.02], "relative_strength_btc_4h": [.01],
        "cross_sectional_return_rank_4h": [.8], "breadth_4h": [.6], "beta_btc": [1.1],
        "common_alt_state": [.015], "utc_hour_sin": [0.], "utc_hour_cos": [1.],
        "weekday_sin": [0.], "weekday_cos": [1.],
    })


def test_canonical_features_are_side_adjusted_without_changing_risk_scale():
    long = canonical_features(_source("LONG")).iloc[0]
    short = canonical_features(_source("SHORT")).iloc[0]
    assert long.side_return_4h == -short.side_return_4h
    assert long.side_taker_flow_1h == -short.side_taker_flow_1h
    assert long.side_cross_sectional_rank_4h == pytest.approx(.8)
    assert short.side_cross_sectional_rank_4h == pytest.approx(.2)
    assert long.realized_volatility_24h == short.realized_volatility_24h


def test_canonical_feature_order_matches_frozen_families():
    expected = tuple(name for names in FAMILIES.values() for name in names)
    assert tuple(canonical_features(_source()).columns) == expected


def test_missing_source_column_fails_closed():
    with pytest.raises(C1ContractError, match="SOURCE_COLUMNS_MISSING"):
        canonical_features(_source().drop(columns="funding_rate"))


def test_nonfinite_source_fails_closed():
    source = _source()
    source.loc[0, "return_1h"] = np.nan
    with pytest.raises(C1ContractError, match="NONFINITE"):
        canonical_features(source)


def test_candidate_contract_is_order_sensitive_and_baseline_first():
    names = feature_names("PRICE_STATE_PLUS_FLOW_ACTIVITY")
    assert names[:len(FAMILIES["PRICE_STATE"])] == FAMILIES["PRICE_STATE"]
    assert contract_hash(names) != contract_hash(tuple(reversed(names)))
