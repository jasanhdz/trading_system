from __future__ import annotations

from copy import deepcopy

import pytest
import yaml

from aegis.research.alpha_laboratory_v21 import (
    AlphaLaboratoryV21Error,
    AlphaStrategy,
    apply_event_spacing,
    candidate_record,
    classify_timestamp,
    gate_assessment,
    partition_name,
    prepare_row,
)
from aegis.research.decomposed_entry_v9 import V9_FEATURE_NAMES
from aegis.research.feature_information_v14 import TAKER_FLOW_FEATURE_NAMES


def config():
    with open("config/experiments/aegis_alpha_laboratory_v21.yaml", encoding="utf-8") as source:
        return yaml.safe_load(source)


def source(symbol="BTCUSDT", side="LONG", timestamp="2026-03-01T00:00:00+00:00"):
    features = {name: 0.0 for name in V9_FEATURE_NAMES}
    features.update({
        "side_rolling_4h_return": 0.02,
        "side_ret_3": 0.01,
        "side_ret_1": 0.005,
        "side_acceleration": 0.002,
        "volume_ratio_6_24": 1.3,
        "favorable_wick_fraction": 0.3,
        "adverse_wick_fraction": 0.1,
        "soft_archetype_BREAKOUT": 0.8,
        "favorable_close_location": 0.8,
        "range_expansion": 1.2,
    })
    flow = {name: 0.2 for name in TAKER_FLOW_FEATURE_NAMES}
    flow["market_taker_breadth_6"] = 0.7
    profile = {
        "worst_net_return": 0.004,
        "worst_exit_reason": "TRAILING_STOP",
        "worst_bars_held": 5,
        "break_even_armed": True,
        "trailing_armed": True,
        "path_spread": 0.0,
    }
    return {
        "timestamp": timestamp,
        "symbol": symbol,
        "side": side,
        "entry_price": 100.0,
        "v9_features": [features[name] for name in V9_FEATURE_NAMES],
        "v14_taker_flow_feature_names": list(TAKER_FLOW_FEATURE_NAMES),
        "v14_taker_flow_features": [flow[name] for name in TAKER_FLOW_FEATURE_NAMES],
        "protection_profiles": {
            "CURRENT_TS": profile,
            "LOCK_AT_5_ROE": profile,
            "LOCK_AT_10_ROE": profile,
        },
        "v10_contract_outcomes": {"ROE_10_H12": {"realized_utility": 0.003}},
        "mae_fraction": 0.001,
        "mfe_fraction": 0.005,
        "time_underwater_bars": 1,
    }


def prepared(symbol="BTCUSDT", side="LONG", timestamp="2026-03-01T00:00:00+00:00"):
    return prepare_row(source(symbol, side, timestamp), funding_rate=0.00005, funding_age_ms=1_000)


def test_cross_sectional_momentum_selects_only_top_two_aligned_symbols():
    rows = [prepared(symbol) for symbol in ("ADAUSDT", "BTCUSDT", "ETHUSDT")]
    for index, value in enumerate((0.01, 0.03, 0.02)):
        rows[index]["features"]["side_rolling_4h_return"] = value
    result = classify_timestamp(rows, config())
    assert AlphaStrategy.CROSS_SECTIONAL_MOMENTUM not in result.get(0, ())
    assert AlphaStrategy.CROSS_SECTIONAL_MOMENTUM in result[1]
    assert AlphaStrategy.CROSS_SECTIONAL_MOMENTUM in result[2]


def test_extreme_reversal_and_breakout_require_causal_confirmation_and_funding():
    rows = [prepared(symbol) for symbol in ("ADAUSDT", "BTCUSDT", "ETHUSDT")]
    for index, value in enumerate((-0.04, -0.03, 0.02)):
        rows[index]["features"]["side_rolling_4h_return"] = value
    result = classify_timestamp(rows, config())
    assert AlphaStrategy.EXTREME_REVERSAL in result[0]
    assert AlphaStrategy.BREAKOUT_FLOW_FUNDING in result[0]

    crowded = deepcopy(rows)
    crowded[0]["side_adjusted_funding_rate"] = 0.0002
    assert AlphaStrategy.BREAKOUT_FLOW_FUNDING not in classify_timestamp(crowded, config())[0]


def test_contract_rejects_inconsistent_duplicate_feature_and_out_of_period_row():
    row = source()
    duplicates = [index for index, name in enumerate(V9_FEATURE_NAMES) if name == "volume_ratio_6_24"]
    row["v9_features"][duplicates[-1]] = 9.0
    with pytest.raises(AlphaLaboratoryV21Error, match="inconsistent duplicated"):
        prepare_row(row, funding_rate=0.0, funding_age_ms=0)
    with pytest.raises(AlphaLaboratoryV21Error, match="outside frozen"):
        partition_name("2025-01-01T00:00:00+00:00", config()["temporal_protocol"])


def test_event_spacing_is_per_symbol_side_and_strategy():
    base = candidate_record(prepared(), AlphaStrategy.CROSS_SECTIONAL_MOMENTUM, config())
    duplicate = deepcopy(base)
    duplicate["timestamp_ms"] += 30 * 60_000
    other = deepcopy(duplicate)
    other["symbol"] = "ETHUSDT"
    assert len(apply_event_spacing([base, duplicate, other], 60)) == 2


def test_gate_requires_positive_validation_holdout_and_random_outperformance():
    base = candidate_record(prepared(), AlphaStrategy.CROSS_SECTIONAL_MOMENTUM, config())
    periods = {}
    for partition, count in (("DISCOVERY", 50), ("VALIDATION", 30), ("FINAL_HOLDOUT", 30)):
        values = []
        for index in range(count):
            row = deepcopy(base)
            row["timestamp"] = f"2026-{(index % 6) + 1:02d}-{(index % 27) + 1:02d}T00:00:00+00:00"
            values.append(row)
        periods[partition] = values
    random_rows = [{**row, "protected_net_return": -0.001} for row in periods["FINAL_HOLDOUT"]]
    result = gate_assessment(periods, random_rows, config()["gate"])
    assert result["passed"] is True
    periods["FINAL_HOLDOUT"][0]["protected_net_return"] = -1.0
    assert gate_assessment(periods, random_rows, config()["gate"])["passed"] is False

