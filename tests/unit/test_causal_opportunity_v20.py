from __future__ import annotations

from copy import deepcopy

import yaml
import pytest

from aegis.research.causal_opportunity_v20 import (
    OpportunityFamily,
    OpportunityV20Error,
    classify_opportunities,
    economic_summary,
    opportunity_record,
    side_adjusted_flow,
    viability,
)
from aegis.research.decomposed_entry_v9 import V9_FEATURE_NAMES
from aegis.research.feature_information_v14 import TAKER_FLOW_FEATURE_NAMES


def config():
    return yaml.safe_load(open("config/experiments/aegis_opportunity_dataset_v20.yaml"))


def source(side="LONG"):
    features = {name: 0.0 for name in V9_FEATURE_NAMES}
    features.update(
        {
            "side_ret_3": 0.01,
            "side_ret_12": 0.02,
            "side_trend_stack": 1.0,
            "timeframe_alignment_score": 1.0,
            "volume_ratio_6_24": 1.2,
        }
    )
    flow = {name: 0.2 for name in TAKER_FLOW_FEATURE_NAMES}
    flow["market_taker_breadth_6"] = 0.7
    return {
        "timestamp": "2026-01-01T00:00:00+00:00",
        "symbol": "BTCUSDT",
        "side": side,
        "entry_price": 100.0,
        "v9_features": [features[name] for name in V9_FEATURE_NAMES],
        "v14_taker_flow_feature_names": list(TAKER_FLOW_FEATURE_NAMES),
        "v14_taker_flow_features": [flow[name] for name in TAKER_FLOW_FEATURE_NAMES],
        "v11_causal_regime": "TREND_UP_LOW_VOL",
        "v11_clean_entry_label": True,
        "protection_profiles": {
            "CURRENT_TS": {
                "worst_net_return": 0.004,
                "worst_exit_reason": "TRAILING_STOP",
                "worst_bars_held": 5,
                "break_even_armed": True,
                "trailing_armed": True,
                "path_spread": 0.0,
            }
        },
        "v10_contract_outcomes": {
            "ROE_10_H12": {"realized_utility": 0.003, "outcome": "FAVORABLE_FIRST"}
        },
        "mae_fraction": 0.001,
        "mfe_fraction": 0.005,
        "time_underwater_bars": 1,
    }


def test_directional_flow_is_oriented_for_long_and_short():
    long = side_adjusted_flow(source("LONG"))
    short_source = source("SHORT")
    short = side_adjusted_flow(short_source)
    assert long["side_taker_imbalance_6"] == 0.2
    assert short["side_taker_imbalance_6"] == -0.2
    assert long["side_market_taker_breadth_6"] == 0.7
    assert abs(short["side_market_taker_breadth_6"] - 0.3) < 1e-12


def test_frozen_trend_rule_detects_and_serializes_a_causal_opportunity():
    row = source()
    families = classify_opportunities(row, config())
    assert OpportunityFamily.TREND_CONTINUATION in families
    result = opportunity_record(row, OpportunityFamily.TREND_CONTINUATION)
    assert result["feature_schema"] == "V9_176_PLUS_SIDE_ADJUSTED_V14_TAKER_FLOW_10"
    assert len(result["features"]) == 186
    assert result["protected_net_return"] == 0.004
    assert result["exchange_mutations"] == 0


def test_opposite_flow_prevents_the_same_long_opportunity():
    row = source()
    index = list(TAKER_FLOW_FEATURE_NAMES).index("taker_imbalance_6")
    row["v14_taker_flow_features"][index] = -0.2
    assert OpportunityFamily.TREND_CONTINUATION not in classify_opportunities(row, config())


def test_viability_requires_economic_and_temporal_evidence():
    base = opportunity_record(source(), OpportunityFamily.TREND_CONTINUATION)
    rows = []
    for month in range(1, 7):
        for index in range(10):
            row = deepcopy(base)
            row["timestamp"] = f"2026-{month:02d}-{index + 1:02d}T00:00:00+00:00"
            rows.append(row)
    result = viability(rows, config()["viability_gate"])
    assert result["passed"] is True
    assert economic_summary(rows)["protected_win_rate"] == 1.0


def test_negative_family_is_not_eligible_for_modeling():
    base = opportunity_record(source(), OpportunityFamily.TREND_CONTINUATION)
    rows = []
    for index in range(60):
        row = deepcopy(base)
        row["timestamp"] = f"2026-01-{index % 28 + 1:02d}T{index % 24:02d}:00:00+00:00"
        row["protected_net_return"] = -0.004
        row["contract_utility"] = -0.003
        rows.append(row)
    assert viability(rows, config()["viability_gate"])["passed"] is False


def test_contract_fails_on_reordered_flow_or_inconsistent_duplicate_v9_feature():
    row = source()
    row["v14_taker_flow_feature_names"] = list(reversed(TAKER_FLOW_FEATURE_NAMES))
    with pytest.raises(OpportunityV20Error, match="taker-flow order mismatch"):
        classify_opportunities(row, config())

    row = source()
    duplicate_positions = [
        index for index, name in enumerate(V9_FEATURE_NAMES) if name == "volume_ratio_6_24"
    ]
    row["v9_features"][duplicate_positions[-1]] = 9.0
    with pytest.raises(OpportunityV20Error, match="inconsistent duplicated V9 feature"):
        classify_opportunities(row, config())
