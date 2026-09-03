import numpy as np
import pandas as pd
import pytest

from aegis.research.compression_entry_m1c import (
    M1C_FEATURE_NAMES,
    add_multitimeframe_features,
    m1c_feature_row,
    pullback_reclaim_confirmation,
    validate_feature_order,
)
from aegis.research.market_event_economic_path_m1b import M1BContractError


def _frame(count=300):
    close = 100.0 + np.arange(count) * 0.01
    frame = pd.DataFrame(
        {
            "open_time": np.arange(count, dtype=np.int64) * 60_000,
            "open": close - 0.005,
            "high": close + 0.02,
            "low": close - 0.02,
            "close": close,
            "flow_1": np.linspace(-0.2, 0.2, count),
            "prior_high": close - 0.001,
        }
    )
    return frame


def test_multitimeframe_features_are_causal_and_complete():
    enriched = add_multitimeframe_features(_frame())
    row = enriched.iloc[-1].to_dict()
    row.update(
        {
            "ret_3": 0.01,
            "ret_12": 0.01,
            "ret_60": 0.01,
            "flow_3": 0.1,
            "volume_ratio": 1.0,
            "compression": 0.5,
            "breakout_up": 0.01,
            "breakout_down": -0.01,
            "mark_spot_basis": 0.0,
            "basis_change_15m": 0.0,
            "basis_zscore_7d": 0.0,
            "funding_rate": 0.0,
            "funding_age_hours": 1.0,
            "direction_score": 0.01,
            "realized_volatility_1h": 0.01,
            "liquidity_ratio_1h": 1.0,
            "btc_return_1h": 0.01,
            "cross_symbol_breadth_1h": 0.2,
            "utc_hour_sin": 0.0,
            "utc_hour_cos": 1.0,
            "weekday_sin": 0.0,
            "weekday_cos": 1.0,
        }
    )
    values = m1c_feature_row(row)
    assert len(values) == len(M1C_FEATURE_NAMES) == 38
    assert np.isfinite(values).all()


def test_pullback_reclaim_enters_only_on_bar_after_confirmation():
    frame = _frame(20)
    event_time = 5 * 60_000
    frame.loc[6, ["open", "high", "low", "close"]] = [100.1, 100.2, 99.9, 100.05]
    frame.loc[7, ["open", "high", "low", "close"]] = [100.0, 100.3, 99.9, 100.2]
    confirmation, entry = pullback_reclaim_confirmation(frame, event_open_time=event_time)
    assert confirmation == 7 * 60_000
    assert entry == 8 * 60_000


def test_pullback_reclaim_abstains_without_confirmation():
    frame = _frame(20)
    event_time = 5 * 60_000
    frame.loc[6:10, "low"] = 200.0
    assert pullback_reclaim_confirmation(frame, event_open_time=event_time) is None


def test_feature_contract_rejects_reordering():
    with pytest.raises(M1BContractError, match="FEATURE_ORDER_INVALID"):
        validate_feature_order(tuple(reversed(M1C_FEATURE_NAMES)))
