#!/usr/bin/env python3
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import aegis_alpha.tools.refresh_trrm_causal_features_f02 as mod


def test_overlap_metrics_and_incremental_rows_are_label_free() -> None:
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        feature_cols = ["feature.ret_1", "feature.atr_proxy_24"]
        expected = pd.DataFrame(
            [
                {"id.symbol": "BTCUSDT", "id.timestamp": "2026-07-01 00:00:00", "id.timeframe": "5m", "id.horizon": 6, "feature.ret_1": 0.1, "feature.atr_proxy_24": 0.2},
                {"id.symbol": "BTCUSDT", "id.timestamp": "2026-07-01 00:00:00", "id.timeframe": "5m", "id.horizon": 12, "feature.ret_1": 0.1, "feature.atr_proxy_24": 0.2},
            ]
        )
        rebuilt = expected.copy()
        comp = mod.compare_overlap(rebuilt, expected, feature_cols)
        assert comp["passes"] is True
        assert comp["row_match_rate"] == 1.0
        assert comp["feature_value_match_rate"] == 1.0
        expected.loc[0, "feature.ret_1"] = 0.5
        bad = mod.compare_overlap(rebuilt, expected, feature_cols)
        assert bad["passes"] is False

        rows = pd.DataFrame(
            [
                {"id.symbol": "BTCUSDT", "id.timestamp": "2026-07-09 00:00:00", "id.timeframe": "5m", "id.horizon": 6, "feature.ret_1": 0.1, "feature.atr_proxy_24": 0.2},
            ]
        )
        out = root / "features.csv"
        rows.to_csv(out, index=False)
        assert mod.forbidden_columns(rows) == []
        rows["target.tail_risk_roe_030"] = 0
        assert "target.tail_risk_roe_030" in mod.forbidden_columns(rows)


if __name__ == "__main__":
    test_overlap_metrics_and_incremental_rows_are_label_free()
    print("test_refresh_trrm_causal_features_f02: OK")
