from __future__ import annotations

from train_regime_aware_directional_v6_shadow import (
    _derive_policy,
    _fit_regime_router,
    _regime_probabilities,
)


def test_no_calibration_policy_when_frequency_is_insufficient() -> None:
    config = {
        "policy_search": {
            "score_quantiles": [0.5],
            "maximum_mae_quantiles": [0.5],
            "maximum_time_quantiles": [0.5],
            "maximum_reversal_quantiles": [0.5],
            "maximum_selected_per_timestamp_grid": [1],
        },
        "validation": {"minimum_calibration_selections": 40},
    }
    rows = [
        {
            "timestamp": f"2026-01-01T00:{index:02d}:00+00:00",
            "symbol": "BTCUSDT",
            "quality_score": 0.8,
            "mae_q90": 0.003,
            "time_to_advantage": 0.2,
            "early_reversal_probability": 0.1,
            "full_lifecycle_worst_net_return": 0.002,
            "mae_fraction": 0.003,
            "time_underwater_bars": 2,
            "protectable_advantage": True,
            "target_before_stop": True,
            "early_reversal": False,
            "full_lifecycle_worst_bars_held": 3,
        }
        for index in range(10)
    ]
    assert _derive_policy(rows, config) is None


def test_regime_router_produces_normalized_three_class_probabilities() -> None:
    def rows(start: int, count: int) -> list[dict[str, object]]:
        result = []
        classes = ("BEARISH", "NEUTRAL", "BULLISH")
        for index in range(start, start + count):
            identity = index % 3
            feature = -1.0 if identity == 0 else 0.0 if identity == 1 else 1.0
            result.append(
                {
                    "timestamp": f"2026-01-{index // 1440 + 1:02d}T"
                    f"{(index // 60) % 24:02d}:{index % 60:02d}:00+00:00",
                    "regime_router_features": (feature,) * 16,
                    "realized_global_regime": classes[identity],
                }
            )
        return result

    router = _fit_regime_router(rows(0, 300), rows(300, 120), 7)
    assert router is not None
    probabilities = _regime_probabilities(rows(500, 3), router)
    assert len(probabilities) == 3
    assert all(abs(sum(value.values()) - 1.0) < 1e-12 for value in probabilities)
    assert all(
        set(value) == {"BULLISH", "NEUTRAL", "BEARISH"} for value in probabilities
    )
