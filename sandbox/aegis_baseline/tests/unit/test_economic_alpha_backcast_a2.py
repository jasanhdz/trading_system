import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from aegis.research.economic_alpha_backcast_a2 import (
    A1_RESULT_SHA256,
    A2ContractError,
    clustered_bootstrap,
    fixed_horizon_outcomes,
    frozen_scales,
    load_frozen_a1_contract,
)


def test_a1_authority_hash_is_fail_closed(tmp_path: Path):
    path = tmp_path / "result.json"
    path.write_text("{}\n")
    with pytest.raises(A2ContractError, match="AUTHORITY_HASH_MISMATCH"):
        load_frozen_a1_contract(path)
    assert len(A1_RESULT_SHA256) == 64


def test_frozen_scales_reconstruct_exact_contract():
    payload = {
        "scales": {
            "LONG": {
                "component": {
                    "median": 2.0,
                    "iqr": 1.0,
                    "lower": 0.0,
                    "upper": 3.0,
                    "scale_method": "IQR",
                }
            }
        }
    }
    scale = frozen_scales(payload, "LONG")["component"]
    assert scale.apply(pd.Series([4.0])).iloc[0] == 1.0


def test_fixed_horizon_outcome_uses_next_state_open_and_direction():
    times = np.arange(0, 76, dtype=np.int64) * 60_000
    minute = pd.DataFrame(
        {
            "open_time": times,
            "open": np.linspace(100.0, 101.0, len(times)),
            "high": np.linspace(100.1, 101.1, len(times)),
            "low": np.linspace(99.9, 100.9, len(times)),
        }
    )
    event = pd.DataFrame(
        {
            "timestamp_ms": [0],
            "state_close_ms": [899_999],
            "symbol": ["BTCUSDT"],
            "side": ["LONG"],
            "mechanism": ["TREND_ACCEPTANCE"],
            "score": [3.0],
        }
    )
    funding = pd.DataFrame({"funding_time": [0], "funding_rate": [0.0]})
    result = fixed_horizon_outcomes(
        event, {"BTCUSDT": minute}, {"BTCUSDT": funding}, 60
    )
    assert len(result) == 1
    assert result.iloc[0]["entry_time"] == 900_000
    assert result.iloc[0]["gross_return"] > 0.0
    assert result.iloc[0]["net_primary_14bps"] < result.iloc[0]["gross_return"]


def test_clustered_bootstrap_is_reproducible():
    rows = pd.DataFrame(
        {
            "timestamp_ms": [0, 0, 86_400_000, 172_800_000],
            "net_primary_14bps": [0.01, -0.005, 0.02, -0.001],
        }
    )
    assert clustered_bootstrap(rows, repetitions=50) == clustered_bootstrap(
        rows, repetitions=50
    )
