from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from aegis_e4.dataset import build_targets


EXPERIMENT = Path(__file__).resolve().parents[1]


def _config() -> dict:
    return json.loads((EXPERIMENT / "config/preregistration_v1.json").read_text())


def test_same_minute_dual_barrier_is_adverse_first() -> None:
    start = pd.Timestamp("2023-09-01T00:00:00Z")
    candles = pd.DataFrame({
        "open_time_ms": ((start.value // 1_000_000) + np.arange(61) * 60_000).astype("int64"),
        "open": np.repeat(100.0, 61), "high": np.repeat(100.0, 61),
        "low": np.repeat(100.0, 61), "close": np.repeat(100.0, 61),
    })
    candles.loc[1, ["high", "low"]] = [101.0, 99.0]
    rows = pd.DataFrame({
        "row_id": ["long"], "label_state": ["LABELED"], "side": ["LONG"],
        "decision_at": [start + pd.Timedelta(minutes=1)],
        "feature__base__tf15m__atr_pct_bps": [100.0],
    })
    result = build_targets(candles, rows, _config()).iloc[0]
    assert result.target__adverse_first == 1
    assert result.target__favorable_first == 0


def test_episode_contract_is_hourly_and_holdout_is_sealed() -> None:
    manifest_path = EXPERIMENT / "artifacts/dataset_v1/dataset_manifest.json"
    if not manifest_path.exists():
        return
    manifest = json.loads(manifest_path.read_text())
    assert manifest["decision_cadence_minutes"] == 5
    assert manifest["effective_symbol_episodes"] < manifest["market_states"]
    assert manifest["sealed_holdout_targets_built"] is False
    sealed = pd.read_parquet(EXPERIMENT / "artifacts/dataset_v1/final_holdout_features_sealed.parquet")
    assert not any(column.startswith("target__") for column in sealed.columns)
