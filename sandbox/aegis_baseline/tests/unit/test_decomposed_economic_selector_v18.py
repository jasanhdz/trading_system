from aegis.research.decomposed_economic_selector_v18 import (
    economic_metrics,
    offline_side_gate,
    select_candidates,
)


POLICY = {
    "minimum_clean_probability": 0.5,
    "maximum_danger_probability": 0.25,
    "maximum_mae_fraction": 0.005,
    "minimum_expected_utility": 0.0,
}


def row(symbol: str, utility: float, expected: float, **changes: float):
    value = {
        "timestamp": "2026-01-01T00:00:00Z",
        "symbol": symbol,
        "clean_probability": 0.7,
        "danger_probability": 0.1,
        "mae_q90": 0.003,
        "expected_utility": expected,
        "actual_utility": utility,
        "mae_fraction": 0.002,
        "mfe_fraction": 0.006,
    }
    value.update(changes)
    return value


def test_selects_at_most_one_and_does_not_relax_any_gate():
    selected = select_candidates(
        [
            row("BTCUSDT", 0.01, 0.01),
            row("ETHUSDT", 0.02, 0.02),
            row("ADAUSDT", 0.03, 0.03, danger_probability=0.251),
        ],
        POLICY,
    )
    assert [item["symbol"] for item in selected] == ["ETHUSDT"]


def test_economic_metrics_include_loss_tail_and_drawdown():
    first = row("BTCUSDT", 0.02, 0.01)
    second = {**row("ETHUSDT", -0.01, 0.01), "timestamp": "2026-01-01T00:05:00Z"}
    metrics = economic_metrics([first, second])
    assert metrics["net_expectancy"] == 0.005
    assert metrics["profit_factor"] == 2.0
    assert metrics["maximum_drawdown_fraction"] == 0.01
    assert metrics["cvar_05"] == -0.01


def test_gate_fails_when_uncertainty_does_not_exclude_zero():
    metrics = {
        "count": 100,
        "net_expectancy": 0.001,
        "profit_factor": 1.5,
        "cvar_05": -0.003,
        "mean_mae_fraction": 0.003,
    }
    gate = {
        "minimum_selected_per_direction": 100,
        "require_mean_net_expectancy_ci_lower_gt": 0.0,
        "require_profit_factor_ci_lower_gt": 1.0,
        "require_cvar_05_gt": -0.004,
        "require_mean_mae_lte": 0.004,
        "require_positive_validation_thirds": 2,
    }
    result = offline_side_gate(
        metrics,
        {"net_expectancy_95": [-0.001, 0.003], "profit_factor_95": [1.1, 2.0]},
        [{"metrics": {"net_expectancy": 0.001}}] * 3,
        gate,
        random_expectancy=0.0,
    )
    assert result["passed"] is False
    assert result["checks"]["expectancy_ci"] is False


def test_preregistered_partitions_are_disjoint_and_holdout_is_sealed():
    root = Path(__file__).resolve().parents[2]
    config = yaml.safe_load(
        (root / "config/experiments/aegis_v18_preregistered.yaml").read_text()
    )
    partitions = config["partitions"]
    assert partitions["train"]["end"] < partitions["validation"]["start"]
    assert partitions["validation"]["end"] < partitions["final_holdout"]["start"]
    assert partitions["final_holdout"]["status"] == "SEALED_NOT_AVAILABLE"
    assert partitions["final_holdout"]["access_count"] == 0
    assert config["models"]["search_space"] == "NONE"
    assert config["anti_overfitting"]["repeated_holdout_testing"] == "PROHIBITED"
from pathlib import Path

import yaml
