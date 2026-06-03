#!/usr/bin/env python3
import json
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
import sys
sys.path.insert(0, str(ROOT))

from aegis_alpha.tools.train_long_alpha_candidates_c import (
    TrainConfig,
    _metrics_for_classifier,
    _regression_metrics,
    assert_research_only_path,
    build_probability_buckets,
    classify_long_model_candidate,
    one_class,
    save_model_bundle,
    select_long_family_features,
    temporal_split_indices,
    write_reports,
)


def base_row(**updates):
    row = {
        "symbol": "ETHUSDT",
        "side": "LONG",
        "alpha_family": "slow_trend_long",
        "target_name": "long_hit3_before_minus2",
        "horizon_candles": 24,
        "model_status": "RAW",
        "train_samples": 1200,
        "validation_samples": 400,
        "test_samples": 400,
        "feature_count": 8,
        "hit_auc": 0.56,
        "hit_top_decile_hit_lift": 0.05,
        "quality_corr": 0.04,
        "quality_lift": 0.01,
        "net_quality_lift_after_costs": 0.01,
        "hit_lift": 0.04,
        "p90_mae_delta": 0.0,
        "stop_rate_delta": 0.0,
        "selected_fraction": 0.12,
        "time_to_target_avg": 10.0,
    }
    row.update(updates)
    return row


def main():
    train, val, test = temporal_split_indices(100)
    assert train[0] == 0 and train[-1] == 59
    assert val[0] == 60 and val[-1] == 79
    assert test[0] == 80 and test[-1] == 99

    assert_research_only_path(ROOT / "aegis_alpha/models/research/long_alpha/ETHUSDT")
    for bad in [
        ROOT / "aegis_alpha/models/turbo/ETHUSDT/active/x.joblib",
        ROOT / "aegis_alpha/models/turbo/ETHUSDT/active_manifest.json",
        ROOT / "aegis_alpha/models/turbo/phase_o_short_manifest.json",
    ]:
        try:
            assert_research_only_path(bad)
            raise AssertionError(f"expected bad path rejection for {bad}")
        except ValueError:
            pass

    assert one_class(np.array([1, 1, 1]))
    assert not one_class(np.array([0, 1, 1]))

    cls = _metrics_for_classifier(np.array([0, 1, 0, 1]), np.array([0.5, 0.5, 0.5, 0.5]))
    assert "roc_auc" in cls and "average_precision" in cls
    reg = _regression_metrics(np.array([1.0, 2.0, 3.0]), np.array([1.2, 1.9, 3.1]))
    assert reg["mae"] is not None and reg["rmse"] is not None

    buckets = build_probability_buckets(
        np.linspace(0, 1, 50),
        np.array([0, 1] * 25),
        np.linspace(-0.1, 0.1, 50),
        np.array([0, 1] * 25),
        np.array([1, 0] * 25),
        symbol="ETHUSDT",
        family="slow_trend_long",
        target_name="long_hit3_before_minus2",
        horizon=24,
        bucket_source="hit_probability",
    )
    assert buckets and sum(r["count"] for r in buckets) == 50

    assert classify_long_model_candidate(base_row()) == "LONG_MODEL_PROMISING"
    assert classify_long_model_candidate(base_row(net_quality_lift_after_costs=-0.01)) == "LONG_MODEL_FAILED"
    assert classify_long_model_candidate(base_row(alpha_family="momentum_burst_long", horizon_candles=6, time_to_target_avg=5.5)) != "LONG_MODEL_PROMISING"
    assert classify_long_model_candidate(base_row(feature_count=0)) == "INSUFFICIENT_DATA"

    fake_frame = {"return_3": None, "return_6": None, "close_location_24": None}
    names, missing, proxies = select_long_family_features(fake_frame, "slow_trend_long")
    assert "return_6" in names
    assert any(p.startswith("return_30m->") for p in proxies)
    assert missing

    out = Path(tempfile.mkdtemp())
    files, count = save_model_bundle(
        out / "models" / "research" / "long_alpha" / "ETHUSDT",
        {"long_hit_classifier": None},
        {
            "schema_version": "test",
            "research_only": True,
            "symbol": "ETHUSDT",
            "side": "LONG",
        },
        save_models=False,
    )
    assert files == [] and count == 0
    meta = json.loads((out / "models" / "research" / "long_alpha" / "ETHUSDT" / "metadata.json").read_text())
    assert meta["research_only"] is True

    args = type("Args", (), {
        "out_dir": str(out),
        "symbols": "ETHUSDT",
        "families": "",
        "family_group": "core",
        "db_path": "x",
        "model_dir": str(out / "models/research/long_alpha"),
        "lookback_days": 1,
        "feature_mode": "selected_family",
        "min_train_samples": 1,
        "min_test_samples": 1,
        "fast": True,
        "no_save_models": True,
        "save_models": False,
    })()
    row = base_row(model_status="LONG_MODEL_PROMISING", model_reason="ok", saved_model_count=0, model_bundle_path=str(out))
    paths = write_reports([row], buckets, args)
    assert Path(paths["json"]).exists()
    assert json.loads(Path(paths["json"]).read_text())["schema_version"]
    assert Path(paths["all_configs"]).exists()

    _ = TrainConfig("ETHUSDT", "slow_trend_long", "long_hit3_before_minus2", 24)
    print("PASS test_train_long_alpha_candidates_c")


if __name__ == "__main__":
    main()
