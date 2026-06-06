#!/usr/bin/env python3
import json
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
import sys
sys.path.insert(0, str(ROOT))

from aegis_alpha.tools.profile_long_training_performance import (
    LongResearchCache,
    SectionTimer,
    classify_bottleneck,
    detect_gpu,
    estimate_experiment_cost,
    write_reports,
)


def main():
    timer = SectionTimer()
    with timer.section("feature_build_base_time", fold=1):
        time.sleep(0.001)
    assert timer.sections and timer.sections[0]["section"] == "feature_build_base_time"
    assert timer.total("feature_build") > 0

    cost = estimate_experiment_cost(symbols=1, targets=4, horizons=1, repair_modes=6, folds=4, models_per_fold=4, avg_time_per_model_fit=0.5, avg_time_per_config=2.0)
    assert cost["total_model_fits"] == 384
    assert cost["estimated_total_minutes"] > 0
    assert cost["config_count"] == 24
    assert cost["estimated_config_minutes"] == 0.8

    assert classify_bottleneck({"total_wall_time": 100, "train_hit_classifier_time": 25, "train_quality_regressor_time": 25}) == "BOTTLENECK_SKLEARN_FIT"
    assert classify_bottleneck({"total_wall_time": 100, "feature_build_base_time": 20, "feature_build_long_c_time": 15}) == "BOTTLENECK_FEATURE_BUILD"
    assert classify_bottleneck({"total_wall_time": 100, "load_candles_symbol_time": 20, "load_btc_context_time": 12}) == "BOTTLENECK_SQLITE_IO"
    assert classify_bottleneck({"total_wall_time": 100}, swap_used_bytes=1) == "BOTTLENECK_MEMORY_PRESSURE"

    gpu = detect_gpu()
    assert "rocm_smi" in gpu and "torch" in gpu

    out = Path(tempfile.mkdtemp())
    profile = {
        "schema_version": "test",
        "created_at": "now",
        "mode": "RESEARCH_ONLY",
        "safety": {},
        "args": {"symbol": "AVAXUSDT", "family": "micro_roe_momentum_long", "target": "long_roe12_before_minus8", "horizon": 6},
        "timings": {"total_wall_time": 1.0},
        "timing_sections": [{"section": "total", "seconds": 1.0}],
        "result": {"total_model_fit_count": 4, "total_prediction_count": 100, "model_fit_rows": [{"fold": 1, "model": "hit", "fit": True}]},
        "gpu": {"torch": {"cuda_is_available": False, "hip": None}},
        "bottleneck": "BOTTLENECK_SKLEARN_FIT",
        "cost_estimates": [cost],
        "recommendations": ["cache features"],
    }
    paths = write_reports(profile, out)
    assert Path(paths["json"]).exists()
    assert json.loads(Path(paths["json"]).read_text())["schema_version"] == "test"
    assert Path(paths["timing_sections"]).exists()
    cached_profile = dict(profile)
    cached_profile['args'] = dict(profile['args'], use_cache=True)
    cached_profile['cache_stats'] = LongResearchCache(max_items=4).summary()
    cached_profile['speedup'] = {'repeat': 2, 'cache_hits': 1}
    cached_paths = write_reports(cached_profile, out)
    assert 'cache' in Path(cached_paths['md']).name
    assert Path(cached_paths['cache_stats']).exists()
    assert Path(cached_paths['speedup']).exists()

    try:
        from aegis_alpha.tools.profile_long_training_performance import run_profile
        args = type("Args", (), {
            "out_dir": str(ROOT / "aegis_alpha/models/turbo/ETHUSDT/active"),
            "model_dir": str(out / "models/research"),
        })()
        # Validate the imported guard path is still active without running the heavy profile.
        from aegis_alpha.tools.train_long_alpha_candidates_c import assert_research_only_path
        assert_research_only_path(args.out_dir)
        raise AssertionError("expected active output path rejection")
    except ValueError:
        pass

    print("PASS test_profile_long_training_performance")


if __name__ == "__main__":
    main()
