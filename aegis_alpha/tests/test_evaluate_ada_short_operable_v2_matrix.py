#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from aegis_alpha.tools.evaluate_ada_short_operable_v2_matrix import (
    SIDE,
    SYMBOL,
    classify_ada_short_config,
    rank_ada_short_configs,
    select_reference_configuration,
    validate_research_model_dir,
    write_csv,
)


def assert_true(value: bool, message: str) -> None:
    if not value:
        raise AssertionError(message)


def row(
    *,
    feature_set: str = "operable_v2",
    quality_lift: float = 0.12,
    hit8_lift: float = 0.06,
    latest_quality: float = 0.11,
    corr: float = 0.05,
    recommendation: str = "WALK_FORWARD_PROMISING",
) -> dict[str, object]:
    return {
        "symbol": SYMBOL,
        "side": SIDE,
        "feature_set": feature_set,
        "lookback_days": 30,
        "horizon_candles": 12,
        "fold_count": 4,
        "valid_fold_count": 4,
        "recommendation": recommendation,
        "stability_score": 0.72,
        "decay_score": 0.01,
        "baseline_hit8_mean": 0.13,
        "baseline_quality_mean": 0.03,
        "baseline_danger_mean": 0.22,
        "v2_hit8_auc_mean": 0.59,
        "v2_hit8_auc_min": 0.55,
        "hit8_top_decile_lift_mean": hit8_lift,
        "hit8_top_decile_lift_min": 0.01,
        "latest_fold_hit8_lift": 0.05,
        "v2_quality_corr_mean": corr,
        "v2_quality_corr_min": -0.01,
        "quality_top_decile_lift_mean": quality_lift,
        "quality_top_decile_lift_min": 0.01,
        "latest_fold_quality_lift": latest_quality,
        "latest_fold_quality_p90_mae": 0.007,
        "latest_fold_baseline_p90_mae": 0.007,
        "v2_danger_auc_mean": 0.60,
        "danger_filter_usefulness_mean": 0.15,
    }


def test_strong_configuration_and_ranking() -> None:
    strong = row()
    weak = row(feature_set="base", quality_lift=-0.03, hit8_lift=0.01, latest_quality=-0.04, corr=-0.02)
    ranked = rank_ada_short_configs([weak, strong])
    assert_true(ranked[0]["feature_set"] == "operable_v2", "stable positive configuration should rank first")
    assert_true(classify_ada_short_config(strong) == "ADA_SHORT_STRONG_RESEARCH", "strong criteria")


def test_references_prefer_strong_over_higher_mixed_score() -> None:
    strong = row(feature_set="operable_v2")
    mixed = row(feature_set="operable_v2", quality_lift=0.40, hit8_lift=0.20, latest_quality=0.30, corr=-0.01)
    ranked = rank_ada_short_configs([strong, mixed])
    assert_true(ranked[0]["ada_research_status"] == "ADA_SHORT_MIXED_RESEARCH", "fixture must rank mixed first by score")
    assert_true(select_reference_configuration(ranked)["ada_research_status"] == "ADA_SHORT_STRONG_RESEARCH", "references must use eligible candidate")


def test_mixed_when_hit8_improves_but_quality_does_not() -> None:
    candidate = row(quality_lift=-0.01, hit8_lift=0.05, latest_quality=-0.01, corr=-0.03)
    assert_true(classify_ada_short_config(candidate) == "ADA_SHORT_MIXED_RESEARCH", "mixed classification")


def test_bad_when_quality_and_hit8_are_negative() -> None:
    candidate = row(quality_lift=-0.08, hit8_lift=-0.04, latest_quality=-0.06, corr=-0.05)
    assert_true(classify_ada_short_config(candidate) == "ADA_SHORT_BAD_RESEARCH", "bad classification")


def test_walk_forward_bad_cannot_be_upgraded_to_mixed() -> None:
    candidate = row(
        recommendation="WALK_FORWARD_BAD",
        quality_lift=0.01,
        hit8_lift=-0.05,
        latest_quality=-0.02,
    )
    assert_true(classify_ada_short_config(candidate) == "ADA_SHORT_BAD_RESEARCH", "walk-forward BAD must remain bad")


def test_serialization_and_research_path_guard() -> None:
    payload = {"symbol": SYMBOL, "side": SIDE, "ranking": rank_ada_short_configs([row()])}
    assert_true("ADAUSDT" in json.dumps(payload), "payload should serialize")
    with tempfile.TemporaryDirectory() as temp:
        path = Path(temp) / "summary.csv"
        write_csv(path, payload["ranking"])
        assert_true(path.exists(), "CSV should write for synthetic rows")
    validate_research_model_dir(Path("/tmp/models/research/turbo_v2_ada_matrix"))
    try:
        validate_research_model_dir(Path("/tmp/models/active/turbo_v2_ada_matrix"))
    except ValueError:
        return
    raise AssertionError("active model path must be rejected")


def test_scope_is_ada_short_only() -> None:
    assert_true(SYMBOL == "ADAUSDT", "matrix must remain scoped to ADA")
    assert_true(SIDE == "SHORT", "matrix must remain scoped to SHORT")


def run_all() -> None:
    test_strong_configuration_and_ranking()
    test_references_prefer_strong_over_higher_mixed_score()
    test_mixed_when_hit8_improves_but_quality_does_not()
    test_bad_when_quality_and_hit8_are_negative()
    test_walk_forward_bad_cannot_be_upgraded_to_mixed()
    test_serialization_and_research_path_guard()
    test_scope_is_ada_short_only()
    print("manual_evaluate_ada_short_operable_v2_matrix_tests_passed")


if __name__ == "__main__":
    run_all()
