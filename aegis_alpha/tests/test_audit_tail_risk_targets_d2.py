#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pandas as pd

from aegis_alpha.tools.audit_tail_risk_targets_d2 import (
    TARGET_CANDIDATES,
    add_tail_targets,
    audit_targets,
    honest_baselines,
    make_temporal_split,
    select_target,
)
from aegis_alpha.tools.build_trrm_causal_feature_dataset_d import is_leakage_column


def fixture(n: int = 360) -> pd.DataFrame:
    rows = []
    for i in range(n):
        mae = 0.30 if i % 12 == 0 else (0.22 if i % 7 == 0 else 0.05)
        atr = 0.005 + (i % 5) * 0.0005
        rows.append({
            "id.symbol": "ADAUSDT" if i % 2 else "ETHUSDT",
            "id.timestamp": str(pd.Timestamp("2025-01-01") + pd.Timedelta(minutes=5 * i)),
            "id.timeframe": "5m",
            "id.horizon": 6 if i % 3 else 24,
            "feature.atr_proxy_24": atr,
            "feature.rolling_range_mean_24": atr * 1.1,
            "feature.rolling_range_std_24": atr * 0.3,
            "feature.rebound_risk_proxy": int(i % 12 == 0),
            "feature.squeeze_risk_proxy_causal": int(i % 10 == 0),
            "feature.ema_slope_6": 0.01,
            "feature.ema_slope_12": 0.01,
            "feature.ema_slope_24": 0.01,
            "feature.ema_slope_48": 0.01,
            "future_eval.future_mae_roe_proxy": mae,
            "future_eval.future_mfe_roe_proxy": 0.08,
            "future_eval.net_quality_after_costs": -mae / 2,
            "target.bad_entry_v4": int(mae >= 0.20),
            "target.early_mae_v4": int(i % 12 == 0),
            "label.clean_entry_v4": int(mae < 0.10),
            "label.management_dependent_v4": int(0.10 <= mae < 0.25),
        })
    return pd.DataFrame(rows)


def test_targets_and_atr_are_built() -> None:
    df = add_tail_targets(fixture())
    for col in TARGET_CANDIDATES:
        assert col in df
    assert "future_eval.mae_atr_units" in df


def test_features_do_not_include_future_and_slope_survives() -> None:
    assert is_leakage_column("feature.future_mae_roe_proxy")[0] is True
    assert is_leakage_column("feature.close")[0] is False
    assert is_leakage_column("feature.ema_slope_6")[0] is False


def test_oracle_is_diagnostic_only_and_not_best() -> None:
    df = add_tail_targets(fixture())
    split = make_temporal_split(df)
    base = honest_baselines(df, "target.tail_risk_roe_025", split)
    oracle = base["items"]["diagnostic_oracle_upper_bound"]
    assert oracle["eligible_for_selection"] is False
    assert oracle["eligible_for_promotion"] is False
    assert base["best_baseline"] != "diagnostic_oracle_upper_bound"


def test_lockbox_not_used_for_selection_and_embargo() -> None:
    df = add_tail_targets(fixture())
    split = make_temporal_split(df, embargo_minutes=120)
    stats = {c: {"global_rate": df[c].mean(), "positives": int(df[c].sum()), "by_horizon": {}, "symbol_rate_max_min_ratio": 1.0} for c in TARGET_CANDIDATES}
    selected = select_target(df, stats, split)
    assert selected["used_lockbox_for_selection"] is False
    assert split["rows_embargoed"] > 0
    assert split["timestamp_overlap_check"] is True


def test_audit_report_has_decision_inputs() -> None:
    out = audit_targets(fixture(), embargo_minutes=120)
    assert out["selected_candidate"]
    assert out["baselines"]["items"]["diagnostic_oracle_upper_bound"]["causal"] is False
    assert out["overlap_diagnostics"]["strided_rows"] < out["overlap_diagnostics"]["dense_rows"]


if __name__ == "__main__":
    test_targets_and_atr_are_built()
    test_features_do_not_include_future_and_slope_survives()
    test_oracle_is_diagnostic_only_and_not_best()
    test_lockbox_not_used_for_selection_and_embargo()
    test_audit_report_has_decision_inputs()
    print("test_audit_tail_risk_targets_d2: OK")
