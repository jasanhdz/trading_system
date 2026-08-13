import numpy as np
import pandas as pd

from aegis.research.opportunity_atlas_b1 import (
    EVENT_FEATURES,
    build_event_targets,
    event_features,
    feature_contract_hash,
    symbol_side_rows,
)


def _panel(symbols=11):
    names = ["BTCUSDT"] + [f"S{i}USDT" for i in range(symbols - 1)]
    return pd.DataFrame({
        "timestamp_ms": [0] * symbols, "symbol": names, "return_1h": np.linspace(-.01, .01, symbols),
        "return_4h": np.linspace(-.02, .02, symbols), "return_24h": np.linspace(-.03, .03, symbols),
        "btc_return_4h": .001, "breadth_4h": .2, "realized_volatility_24h": .01,
        "volume_persistence_1h": 1.2, "taker_flow_1h": .1, "mark_spot_basis": .001,
        "funding_rate": .0001, "utc_hour_sin": 0., "utc_hour_cos": 1.,
        "weekday_sin": 0., "weekday_cos": 1.,
    })


def test_event_features_require_complete_cross_symbol_cluster():
    assert len(event_features(_panel())) == 1
    assert event_features(_panel(10)).empty


def test_event_feature_contract_is_order_sensitive():
    assert feature_contract_hash(EVENT_FEATURES) != feature_contract_hash(tuple(reversed(EVENT_FEATURES)))


def test_event_target_separates_opportunity_and_direction():
    events = event_features(_panel())
    rows = pd.DataFrame({
        "timestamp_ms": [0, 0], "symbol": ["BTCUSDT", "S0USDT"],
        "long_gross": [.001, -.01], "short_gross": [-.001, .01],
    })
    target = build_event_targets(events, rows).iloc[0]
    assert bool(target.opportunity)
    assert target.best_side == "SHORT"
    assert target.best_symbol == "S0USDT"


def test_symbol_side_rows_preserve_both_directions():
    source = pd.DataFrame({
        "timestamp_ms": [0], "symbol": ["BTCUSDT"],
        "long_gross": [.01], "short_gross": [-.01],
        "long_mae": [.002], "short_mae": [.01], "long_mfe": [.01], "short_mfe": [.002],
    })
    rows = symbol_side_rows(source)
    assert rows.side.tolist() == ["LONG", "SHORT"]
    assert rows.gross_return.tolist() == [.01, -.01]
