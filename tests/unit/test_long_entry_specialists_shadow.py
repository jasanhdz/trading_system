from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

from aegis.domain import Candle
from aegis.models import CalibrationMethod, CalibratorSpec
from aegis.research.long_entry_specialist_model_shadow import (
    LongEntrySpecialistModelShadow,
)
from aegis.research.long_entry_specialists_shadow import (
    LONG_SPECIALIST_FEATURE_NAMES,
    classify_long_archetype,
    classify_long_archetype_v2,
    exact_long_path_outcome,
    mirror_short_path_as_long_outcome,
)
from aegis.training.hybrid_directional import calibrator_mapping
from aegis.tree_models import (
    DecisionTree,
    EnsembleAggregation,
    TreeEnsemble,
    TreeNode,
)
from aegis.utils import Sha256HashProvider, canonical_json


def features(**overrides: float):
    values = {name: 0.0 for name in LONG_SPECIALIST_FEATURE_NAMES}
    values["extension_down_proxy"] = 0.0
    values.update(overrides)
    return values


def test_archetypes_use_only_current_causal_structure() -> None:
    continuation = classify_long_archetype(
        features(
            market_breadth_6=0.8,
            market_direction_6=0.01,
            btc_trend_proxy=1.0,
            trend_stack_long=1.0,
            close_vs_ema_12=0.01,
            ema_slope_6=0.01,
            ema_slope_12=0.01,
            ret_6=0.01,
            ret_12=0.02,
            relative_return_6=0.01,
            volume_ratio_6_24=1.2,
        )
    )
    reversal = classify_long_archetype(
        features(
            ret_12=-0.02,
            ret_24=-0.03,
            ret_1=0.005,
            ret_3=0.008,
            momentum_acceleration_3_12=0.01,
            ema_slope_6=0.002,
            close_vs_ema_6=0.001,
            close_position_in_range=0.8,
            lower_wick_fraction=0.4,
            upper_wick_fraction=0.1,
        )
    )
    assert continuation["archetype"] == "TREND_CONTINUATION"
    assert reversal["archetype"] == "CONFIRMED_REVERSAL"
    assert continuation["selection_effect"] == "NONE"
    assert continuation["exchange_mutations"] == 0


def test_short_path_extrema_mirror_exactly_to_long() -> None:
    result = mirror_short_path_as_long_outcome(
        observed={
            "mae_fraction": 0.004,
            "mfe_fraction": 0.001,
            "path_metrics": {"12": {"terminal_short_return": -0.003}},
        },
        label={"time_to_mae": 2, "time_to_mfe": 8},
    )
    assert result["mfe_fraction"] == 0.004
    assert result["mae_fraction"] == 0.001
    assert result["terminal_return_fraction"] == 0.003
    assert result["net_return_after_costs"] == 0.002
    assert result["clean_fast_success"] is True
    assert result["dangerous_entry"] is False


def test_v2_routes_four_long_archetypes_from_causal_features() -> None:
    pullback = classify_long_archetype_v2(
        features(
            trend_stack_long=1.0,
            ema_slope_12=0.003,
            ema_slope_24=0.002,
            ret_12=0.01,
            close_vs_ema_12=0.001,
            ret_1=0.001,
            market_breadth_6=0.7,
            atr_12=0.005,
            distance_to_rolling_high_12=0.01,
        )
    )
    breakout = classify_long_archetype_v2(
        features(
            distance_to_rolling_high_12=-0.001,
            ret_3=0.004,
            ret_6=0.008,
            close_position_in_range=0.85,
            body_to_range=0.7,
            volume_ratio_6_24=1.3,
            market_breadth_6=0.7,
            atr_12=0.005,
        )
    )
    reversal = classify_long_archetype_v2(
        features(
            ret_12=-0.02,
            ret_1=0.003,
            ret_3=0.006,
            momentum_acceleration_3_12=0.02,
            close_vs_ema_6=0.001,
            close_position_in_range=0.8,
            lower_wick_fraction=0.4,
            upper_wick_fraction=0.1,
            distance_to_rolling_high_12=0.02,
            atr_12=0.005,
        )
    )
    rebound = classify_long_archetype_v2(
        features(
            ret_12=-0.03,
            extension_down_proxy=0.02,
            exhaustion_down_proxy=0.03,
            lower_wick_fraction=0.35,
            distance_to_rolling_high_12=0.02,
            atr_12=0.005,
        )
    )
    assert pullback["archetype"] == "TREND_PULLBACK"
    assert breakout["archetype"] == "CONFIRMED_BREAKOUT"
    assert reversal["archetype"] == "CONFIRMED_REVERSAL"
    assert rebound["archetype"] == "SPECULATIVE_REBOUND"
    assert all(
        result["selection_effect"] == "NONE"
        for result in (pullback, breakout, reversal, rebound)
    )


def candle(index: int, *, high: float, low: float, close: float) -> Candle:
    opened = datetime(2026, 1, 1, tzinfo=timezone.utc) + timedelta(minutes=5 * index)
    return Candle(
        open_time=opened,
        close_time=opened + timedelta(minutes=5),
        open=100.0,
        high=high,
        low=low,
        close=close,
        volume=1.0,
        is_closed=True,
        source="TEST",
    )


def test_exact_long_path_uses_real_barrier_order_and_underwater_time() -> None:
    result = exact_long_path_outcome(
        entry_price=100.0,
        future_candles=(
            candle(0, high=100.1, low=99.9, close=99.95),
            candle(1, high=100.4, low=99.85, close=100.3),
            candle(2, high=100.5, low=99.9, close=100.2),
        ),
    )
    assert result["barrier_order"] == "FAVORABLE_FIRST"
    assert result["first_favorable_bar"] == 2
    assert result["first_adverse_bar"] is None
    assert result["clean_fast_success"] is True
    assert result["dangerous_entry"] is False
    assert result["time_underwater_bars"] == 1
    assert (
        result["source_semantics"] == "ACTUAL_FUTURE_OHLC_PATH_NOT_DIRECTIONAL_MIRROR"
    )


def test_same_bar_barrier_order_fails_closed_as_dangerous() -> None:
    result = exact_long_path_outcome(
        entry_price=100.0,
        future_candles=(candle(0, high=100.4, low=99.6, close=100.0),),
    )
    assert result["barrier_order"] == "SAME_BAR_AMBIGUOUS"
    assert result["same_bar_ambiguity"] is True
    assert result["clean_fast_success"] is False
    assert result["dangerous_entry"] is True


def write_artifact(path: Path) -> None:
    tree = TreeEnsemble(
        ensemble_id="long-specialist-test",
        schema_version="aegis-tree-ensemble-v1",
        feature_names=LONG_SPECIALIST_FEATURE_NAMES,
        aggregation=EnsembleAggregation.ADDITIVE_LOGIT,
        base_value=0.0,
        trees=(DecisionTree((TreeNode(0, 0.0, 0, 0, 0.0, True),)),),
        content_hash="",
    ).to_payload()
    calibrator = calibrator_mapping(
        CalibratorSpec(CalibrationMethod.IDENTITY, 0.0, 0.25, 10)
    )
    payload = {
        "schema_id": "aegis-long-entry-specialists-shadow-validation-v1",
        "mode": "SHADOW",
        "feature_names": list(LONG_SPECIALIST_FEATURE_NAMES),
        "shared_danger_model": {
            "model": tree,
            "calibrator": calibrator,
            "maximum_probability": 0.6,
        },
        "specialists": {
            name: {
                "status": "TRAINED_SHADOW_ONLY",
                "model": tree,
                "calibrator": calibrator,
                "success_threshold": 0.4,
                "validation_pass": False,
            }
            for name in ("TREND_CONTINUATION", "CONFIRMED_REVERSAL")
        },
        "validation_pass": False,
        "live_selection_effect": "NONE",
        "exchange_authority": False,
        "exchange_mutations": 0,
    }
    payload["content_hash"] = Sha256HashProvider().digest_value(payload)
    path.write_text(canonical_json(payload) + "\n", encoding="utf-8")


def test_unvalidated_runtime_score_cannot_become_an_entry(tmp_path: Path) -> None:
    path = tmp_path / "long-specialists.json"
    write_artifact(path)
    scorer = LongEntrySpecialistModelShadow(path)
    result = scorer.assess(
        features(
            market_breadth_6=0.8,
            market_direction_6=0.01,
            btc_trend_proxy=1.0,
            trend_stack_long=1.0,
            close_vs_ema_12=0.01,
            ema_slope_6=0.01,
            ema_slope_12=0.01,
            ret_6=0.01,
            ret_12=0.02,
            relative_return_6=0.01,
            volume_ratio_6_24=1.2,
        )
    )
    assert result["status"] == "RESEARCH_SCORE_NOT_VALIDATED"
    assert result["success_probability"] == 0.5
    assert result["counterfactual_pass"] is False
    assert result["counterfactual_action"] == "OBSERVE_ONLY"
    assert result["selection_effect"] == "NONE"
    assert result["exchange_authority"] is False
    assert result["exchange_mutations"] == 0
